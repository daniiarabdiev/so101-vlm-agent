"""Minimal RunPod Pod lifecycle for Run 3: create, inspect, ssh, delete (+404 check), deadline watchdog.

Every create/delete is appended to run3/budget/pods.jsonl. Actual spend is taken from the
RunPod billing API (see `billing`), not from reservations.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

API = "https://api.runpod.io"
KEY_FILE = Path.home() / ".config/runpod/so101-vlm-api-key"
SSH_PUB = Path.home() / ".ssh/id_ed25519.pub"
LOG = Path(__file__).resolve().parent / "budget" / "pods.jsonl"
# Run 2 proven public image (vLLM 0.29.0); see run2/workers/provider_completion.md
VLLM_IMAGE = "runpod/worker-v1-vllm@sha256:fd9e5c55c996361aad2543d96d9d85ca055625ee5d1fcefa2d213141deb45e17"
START = (
    "(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync "
    "> /tmp/apt.log 2>&1); mkdir -p /root/.ssh /run/sshd; echo \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; "
    "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd -D -e > /tmp/sshd.log 2>&1 & "
    "sleep infinity"
)


def _key() -> str:
    return KEY_FILE.read_text().strip()


def api(method: str, path: str, **kw) -> httpx.Response:
    return httpx.request(method, API + path, headers={"Authorization": f"Bearer {_key()}"}, timeout=60, **kw)


def log(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"time_unix": time.time(), **row}, sort_keys=True) + "\n")


def create(name: str, gpu: str, disk: int = 200, image: str = VLLM_IMAGE, cloud: str = "SECURE",
           ports=("22/tcp", "8000/http", "8001/http", "8002/http"), datacenters=None, min_cuda=None) -> dict:
    body = {"name": name, "image": image, "gpu": {"id": gpu, "count": 1}, "disk": disk, "cloud": cloud,
            "ports": list(ports), "env": {"PUBLIC_KEY": SSH_PUB.read_text().strip(), "NVIDIA_DRIVER_CAPABILITIES": "all"},
            "entrypoint": ["/bin/bash", "-c"], "cmd": [START]}
    if datacenters:
        body["dataCenterIds"] = list(datacenters)
    if min_cuda:
        body["gpu"]["minCudaVersion"] = min_cuda
    r = api("POST", "/v2/pods", json=body)
    if r.status_code not in (200, 201):
        log({"event": "create_failed", "name": name, "gpu": gpu, "status": r.status_code, "body": r.text[:500]})
        raise RuntimeError(f"create failed {r.status_code}: {r.text[:300]}")
    pod = r.json()
    log({"event": "created", "pod_id": pod["id"], "name": name, "gpu": gpu, "cost_per_hr": pod.get("cost"),
         "datacenter": pod.get("dataCenterId"), "createdAt": pod.get("createdAt"), "image": image})
    return pod


def get(pod_id: str) -> httpx.Response:
    return api("GET", f"/v2/pods/{pod_id}")


def ssh_target(pod_id: str) -> tuple[str, int] | None:
    r = get(pod_id)
    if r.status_code != 200:
        return None
    pod = r.json()
    direct = ((pod.get("ssh") or {}).get("direct")) or None
    if direct:
        return direct["host"], int(direct["port"])
    for p in (pod.get("runtime") or {}).get("ports") or pod.get("portMappings") or []:
        if isinstance(p, dict) and p.get("private") == 22 and p.get("public") and p.get("ip"):
            return p["ip"], int(p["public"])
    return None


def ssh(pod_id: str, command: str, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    host, port = ssh_target(pod_id)
    args = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", f"root@{host}", command]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check)


def rsync_to(pod_id: str, src: str, dst: str) -> None:
    host, port = ssh_target(pod_id)
    e = f"ssh -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    subprocess.run(["rsync", "-az", "-e", e, src, f"root@{host}:{dst}"], check=True)


def rsync_from(pod_id: str, src: str, dst: str) -> None:
    host, port = ssh_target(pod_id)
    e = f"ssh -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    subprocess.run(["rsync", "-az", "-e", e, f"root@{host}:{src}", dst], check=True)


def delete(pod_id: str) -> dict:
    attempts = []
    for _ in range(5):
        r = api("DELETE", f"/v2/pods/{pod_id}")
        attempts.append(r.status_code)
        if r.status_code in (200, 202, 204, 404):
            break
        time.sleep(5)
    verify = get(pod_id).status_code
    row = {"event": "deleted", "pod_id": pod_id, "delete_status": attempts, "verify_get_status": verify}
    log(row)
    return row


def billing(start: str, end: str) -> dict:
    r = api("GET", f"/v2/billing?startTime={start}&endTime={end}&bucketSize=day")
    return r.json()


def watchdog(pod_id: str, deadline_unix: float) -> None:
    """Detached safety net: delete the pod at the deadline if it still exists."""
    while time.time() < deadline_unix:
        if get(pod_id).status_code == 404:
            log({"event": "watchdog_exit_pod_gone", "pod_id": pod_id})
            return
        time.sleep(60)
    row = delete(pod_id)
    log({"event": "watchdog_deadline_delete", **row})


def start_watchdog(pod_id: str, hours: float) -> int:
    deadline = time.time() + hours * 3600
    proc = subprocess.Popen([sys.executable, "-m", "run3.pod", "watchdog", pod_id, str(deadline)],
                            cwd=str(Path(__file__).resolve().parents[1]), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    log({"event": "watchdog_started", "pod_id": pod_id, "pid": proc.pid, "deadline_unix": deadline})
    return proc.pid


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("name"); c.add_argument("gpu"); c.add_argument("--disk", type=int, default=200)
    c.add_argument("--hours", type=float, required=True); c.add_argument("--image", default=VLLM_IMAGE)
    c.add_argument("--datacenter", action="append"); c.add_argument("--min-cuda")
    s = sub.add_parser("status"); s.add_argument("pod_id")
    d = sub.add_parser("delete"); d.add_argument("pod_id")
    w = sub.add_parser("watchdog"); w.add_argument("pod_id"); w.add_argument("deadline", type=float)
    x = sub.add_parser("ssh"); x.add_argument("pod_id"); x.add_argument("command")
    sub.add_parser("list")
    a = p.parse_args(argv)
    if a.cmd == "create":
        pod = create(a.name, a.gpu, a.disk, a.image, datacenters=a.datacenter, min_cuda=a.min_cuda)
        pid = start_watchdog(pod["id"], a.hours)
        print(json.dumps({"pod_id": pod["id"], "cost": pod.get("cost"), "dc": pod.get("dataCenterId"), "watchdog_pid": pid}))
    elif a.cmd == "status":
        r = get(a.pod_id)
        pod = r.json() if r.status_code == 200 else {}
        print(r.status_code, json.dumps({k: pod.get(k) for k in ("id", "status", "cost", "dataCenterId", "ssh", "runtime", "gpu")}, default=str)[:2000])
    elif a.cmd == "delete":
        print(json.dumps(delete(a.pod_id)))
    elif a.cmd == "watchdog":
        watchdog(a.pod_id, a.deadline)
    elif a.cmd == "ssh":
        r = ssh(a.pod_id, a.command, check=False)
        print(r.stdout[-5000:], r.stderr[-2000:], sep="\n")
    elif a.cmd == "list":
        print(json.dumps(api("GET", "/v2/pods").json(), default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
