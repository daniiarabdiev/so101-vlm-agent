"""SO-101 arm server: the only process that talks to the servos. Run it in the Hiwonder LeRobot environment:

    ${SO101_PYTHON:-python3} real/arm_server.py            (from vlm_policy/)

It uses the same driver path as jev_arm (so101_twin.hardware.CommandFollower: saved calibration, calibrated joint bounds,
per-command relative caps, torque-preserving stop; never configures, calibrates or disables torque) with jev_arm's speed
caps. The agent (real/run_real.py, in the vlm_policy environment) connects over localhost TCP, one JSON object per line:
  {"op": "hello"} {"op": "read"} {"op": "hold"} {"op": "send", "q": [6]} {"op": "ping"} {"op": "stop"}
Safety, enforced here whatever the client sends:
  - STOP KEY: press Enter in this terminal. The arm holds its current position (torque on) and every later command is
    refused until you type "resume" + Enter. Ctrl-C also holds and exits.
  - watchdog: while a hold is active, if the client sends nothing for 0.5 s, or disconnects, the arm holds position.
  - speed: a goal may move at most (speed cap x time since the last goal); larger jumps are clipped towards it.
Joint units: 5 body joints in degrees, gripper 0-100 %.
"""
from __future__ import annotations

import argparse
import copy
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import os
# The follower driver (so101_twin: CommandFollower, resolve_ports) lives outside this repository; point SO101_TWIN_DIR at it.
TWIN_ROOT = os.environ.get("SO101_TWIN_DIR", "")
TWIN = os.path.join(TWIN_ROOT, "twin")
sys.path.insert(0, TWIN); sys.path.insert(0, TWIN_ROOT)


class Server:
    def __init__(self, twin_config: str, body_deg_s: float, grip_pct_s: float, lead_deg: float = 8.0):
        from resolve_ports import extract_serial_ports
        from so101_twin.hardware import CommandFollower
        cfg = json.loads(Path(twin_config).read_text())
        ports = extract_serial_ports(subprocess.check_output(["ioreg", "-a", "-l", "-p", "IOService", "-r", "-k", "USB Serial Number"]))
        serial = cfg["robot"]["usb_serial"]
        if serial not in ports:
            raise RuntimeError(f"follower USB serial {serial} is not connected")
        import os
        os.environ["F"] = ports[serial]
        rc = copy.deepcopy(cfg)
        rc["robot"]["port_last_resolved"] = ports[serial]
        rc["safety"]["body_slew_deg_per_s"] = body_deg_s
        rc["safety"]["gripper_slew_percent_per_s"] = grip_pct_s
        rc["robot"]["max_relative_target"]["gripper"] = 2.0       # jev_arm gripper_target_lead_percent
        for j in ("shoulder_lift", "elbow_flex"):                  # jev_arm body_target_lead_deg (gravity-loaded joints); 8 deg
            rc["robot"]["max_relative_target"][j] = lead_deg       # could not hold the arm at full stretch (2026-09-27): --lead-deg
        self.rates = (body_deg_s, grip_pct_s)
        self.robot = CommandFollower(rc)
        self.lock = threading.Lock()
        self.stopped, self.holding, self.last_msg, self.last_goal, self.last_goal_t = False, False, time.monotonic(), None, None

    def connect(self):
        self.robot.connect()
        print(f"[arm] connected; joints {self.robot.read()}; speed caps {self.rates}", flush=True)

    # --- safety
    def hold_position(self, why: str):
        with self.lock:
            if self.holding:
                try:
                    self.robot.stop_hold()
                finally:
                    self.holding = False
                print(f"[arm] HOLDING POSITION ({why})", flush=True)

    def stop_key(self):
        for line in sys.stdin:
            if line.strip().lower() == "resume":
                self.stopped = False; print("[arm] resumed: the client must start a new hold", flush=True)
            else:
                self.stopped = True; self.hold_position("stop key"); print("[arm] STOPPED. Type 'resume' + Enter to allow motion again.", flush=True)

    def watchdog(self):
        while True:
            time.sleep(.1)
            if self.holding and time.monotonic() - self.last_msg > .5:
                self.hold_position("client silent for 0.5 s")

    # --- commands
    def handle(self, m: dict) -> dict:
        self.last_msg = time.monotonic()
        op = m.get("op")
        if op == "hello":
            return {"ok": True, "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                    "rates": self.rates}
        if op == "read":
            return {"q": self.robot.read()}
        if op == "ping":
            return {"ok": True, "stopped": self.stopped} if not self.stopped else {"stopped": True, "reason": "stop key"}
        if self.stopped and op in ("hold", "send"):
            return {"stopped": True, "reason": "stop key"}
        if op == "hold":
            with self.lock:
                new = not self.holding
                if new:
                    q = self.robot.begin_hold(); self.holding = True; self.last_goal, self.last_goal_t = list(q), time.monotonic()
                return {"q": self.robot.read(), "new": new}
        if op == "send":
            if not self.holding:
                return {"error": "no active hold (the watchdog or stop key ended it): send hold first"}
            q = [float(v) for v in m["q"]]
            now = time.monotonic(); dt = min(max(now - self.last_goal_t, 1e-3), .2)
            lim = [self.rates[0] * dt * 1.5] * 5 + [self.rates[1] * dt * 1.5]
            q = [lg + max(-l, min(l, v - lg)) for v, lg, l in zip(q, self.last_goal, lim)]
            with self.lock:
                out = self.robot.send(q)
            self.last_goal, self.last_goal_t = list(out), now
            return {"q": out}
        if op == "stop":
            self.hold_position("client request")
            return {"ok": True}
        return {"error": f"unknown op {op!r}"}

    def serve(self, port: int):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port)); srv.listen(1)
        print(f"[arm] listening on 127.0.0.1:{port}. Press Enter at any time to STOP.", flush=True)
        while True:
            conn, _ = srv.accept(); f = conn.makefile("rw")
            print("[arm] client connected", flush=True)
            try:
                for line in f:
                    try:
                        r = self.handle(json.loads(line))
                    except Exception as exc:  # noqa: BLE001
                        self.hold_position(f"error: {exc!r}"); r = {"error": repr(exc)[:300]}
                    f.write(json.dumps(r) + "\n"); f.flush()
            except OSError as exc:   # the client vanished mid-reply (its timeout, a crash): a disconnect, not a server crash
                print(f"[arm] client connection lost: {exc!r}", flush=True)   # (2026-09-27: BrokenPipe killed the server)
            finally:
                self.hold_position("client disconnected"); conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--twin-config", default=f"{TWIN}/config.json")
    ap.add_argument("--body-deg-s", type=float, default=10.0); ap.add_argument("--grip-pct-s", type=float, default=12.0)
    ap.add_argument("--lead-deg", type=float, default=8.0, help="max target lead of shoulder_lift / elbow_flex (servo effort)")
    a = ap.parse_args()
    s = Server(a.twin_config, a.body_deg_s, a.grip_pct_s, a.lead_deg)
    s.connect()
    threading.Thread(target=s.stop_key, daemon=True).start()
    threading.Thread(target=s.watchdog, daemon=True).start()
    try:
        s.serve(a.port)
    except KeyboardInterrupt:
        pass
    finally:
        s.hold_position("exit")
        s.robot.close()
        print("[arm] closed (torque unchanged: the arm keeps holding its position)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
