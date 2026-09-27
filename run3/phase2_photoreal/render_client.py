"""Client for the Blender render server (through an SSH tunnel to the Pod)."""
from __future__ import annotations

import io
import os
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from run3 import pod
from run3.phase2_photoreal import mjexport

KEY = Path(os.environ.get("RUN3_KEY_FILE", Path.home() / ".config/runpod/so101-run3-vllm-key"))
ID_COLORS = {"object": (255, 0, 0), "container": (0, 255, 0), "gripper": (0, 0, 255), "arm": (255, 255, 0),
             "distractor": (255, 0, 255)}


def open_tunnel(pod_id: str, local_port: int = 18002, remote_port: int = 8002) -> subprocess.Popen:
    host, port = pod.ssh_target(pod_id)
    proc = subprocess.Popen(["ssh", "-N", "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                             "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
                             "-L", f"{local_port}:localhost:{remote_port}", f"root@{host}"], stdin=subprocess.DEVNULL)
    time.sleep(3)
    return proc


class Renderer:
    def __init__(self, base: str | None = None):
        port = os.environ.get("RUN3_RENDER_PORT", "8002")
        base = base or (f"http://localhost:{port}" if os.environ.get("RUN3_ON_POD") else "http://localhost:18002")
        self.base = base
        self.client = httpx.Client(timeout=600, headers={"Authorization": f"Bearer {KEY.read_text().strip()}"})
        self.scene_key = None
        self.static_sha = None

    def set_scene(self, model, config: dict, force: bool = False) -> float:
        static = mjexport.static_scene(model)
        key = (config["scene_key"], static["static_sha256"])
        if key == self.scene_key and not force:
            return 0.0
        r = self.client.post(self.base + "/scene", json={"static": static, "config": config})
        r.raise_for_status()
        self.scene_key, self._model, self._config = key, model, config
        return float(r.json()["build_s"])

    def render(self, model, data, mode: str = "photo", samples: int = 64, seed: int = 0,
               width: int = 448, height: int = 448) -> tuple[Image.Image, float, float]:
        state = mjexport.frame_state(model, data, "overhead", width, height)
        t0 = time.time()
        body = {**state, "mode": mode, "samples": samples, "seed": seed, "scene_key": self.scene_key[0] if self.scene_key else None}
        r = self.client.post(self.base + "/render", json=body)
        if r.status_code == 409:  # another client replaced the server's scene: rebuild ours and retry once
            self.set_scene(self._model, self._config, force=True)
            r = self.client.post(self.base + "/render", json=body)
        if r.status_code != 200:
            raise RuntimeError(r.text[:2000])
        return Image.open(io.BytesIO(r.content)).convert("RGB"), float(r.headers.get("X-Render-Seconds", "nan")), time.time() - t0


def id_masks(id_image: Image.Image) -> dict[str, np.ndarray]:
    a = np.asarray(id_image).astype(int)
    out = {}
    for role, rgb in ID_COLORS.items():
        out[role] = (np.abs(a - np.asarray(rgb)).sum(-1) < 60)
    return out
