"""Real cameras for the agent's three views (top, side, wrist) as 448x448 RGB arrays, like the simulator's renders.
Each camera: an OpenCV AVFoundation index, optional undistortion (real/cam_intrinsics.py output: K, dist), then a centre
square crop (or an explicit crop box) resized to 448. A background thread keeps the newest frame; a frame older than
max_age_s is refused (the jev_arm rule: never act on stale images). Optional rotation (90/180/270 deg clockwise, after
undistortion, before the crop) makes a camera mounted another way round match the simulator's layout; a left-right mirror
makes a side camera on the robot's left look like the simulator's (on its right). Pointing never uses a mirrored view.
With server="http://127.0.0.1:8766" the frames come from real/cam_server.py (run in Terminal.app, which has macOS camera
permission; the Claude app's shell does not); the server enforces the same freshness rule.
Usage in code: RealCameras({"top": 2, "side": 0, "wrist": 1}, undistort={"top": "real/cal/top_intrinsics.npz"}).frames(views)
List devices: ffmpeg -f avfoundation -list_devices true -i ""
"""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


class Cam:
    ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def __init__(self, index: int, undistort: str | None = None, crop: tuple | None = None, size: int = 448,
                 server: str | None = None, rot: int = 0, mirror: bool = False, expect: tuple | None = None):
        self.index, self.size, self.crop, self.server, self.rot = index, size, crop, server, int(rot) % 360
        self.mirror, self.expect = mirror, expect
        if self.rot and self.rot not in self.ROT:
            raise ValueError("rotation must be 0, 90, 180 or 270")
        self.maps = None
        if undistort:
            d = np.load(undistort); self.K, self.dist = d["K"], d["dist"]
        else:
            self.K = None
        self.lock, self.frame, self.t, self.err = threading.Lock(), None, 0.0, None
        self.stop = threading.Event()
        if server:
            self.cap = None; self._fetch(1.0)   # fail now, not mid-episode, if the server or camera is missing
        else:
            self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open camera {index} (no camera permission? use real/cam_server.py)")
            threading.Thread(target=self._loop, daemon=True).start()

    def _fetch(self, max_age_s: float, wait_s: float = 25.0) -> np.ndarray:
        """Newest frame from the camera server; while a camera reconnects or the server restarts itself (real/cam_server.py),
        keep asking for up to wait_s before failing (2026-09-27: the wrist camera's drops ended runs at the first step)."""
        t0 = time.monotonic()
        while True:
            try:
                with urllib.request.urlopen(f"{self.server}/frame/{self.index}?max_age={max_age_s}", timeout=6) as r:
                    data = r.read()
                return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            except urllib.error.HTTPError as exc:
                err = RuntimeError(f"camera {self.index}: {exc.read().decode(errors='replace')[:200]}")
            except OSError as exc:
                err = RuntimeError(f"camera server {self.server} unreachable: {exc}")
            if time.monotonic() - t0 > wait_s:
                raise err
            time.sleep(.5)

    def _loop(self):
        while not self.stop.is_set():
            ok, f = self.cap.read()
            if not ok:
                self.err = f"camera {self.index} read failed"; return
            with self.lock:
                self.frame, self.t = f, time.monotonic()

    def get(self, max_age_s: float = .4) -> np.ndarray:
        f = self._fetch(max_age_s) if self.server else self._newest(max_age_s)
        return self.process(f)

    def _newest(self, max_age_s: float) -> np.ndarray:
        t0 = time.monotonic()
        while True:
            if self.err:
                raise RuntimeError(self.err)
            with self.lock:
                f, t = self.frame, self.t
            if f is not None and time.monotonic() - t <= max_age_s:
                break
            if time.monotonic() - t0 > 3:
                raise RuntimeError(f"camera {self.index}: no fresh frame")
            time.sleep(.02)
        return f

    def process(self, f: np.ndarray) -> np.ndarray:
        """Raw BGR frame -> the agent's RGB view: undistort, rotate, square crop, resize."""
        if getattr(self, "expect", None) and (f.shape[1], f.shape[0]) != tuple(self.expect):
            raise RuntimeError(f"camera {self.index} delivers {f.shape[1]}x{f.shape[0]}, calibrated at {self.expect[0]}x{self.expect[1]}: "
                               "another app may have switched its format; close it (the camera server re-requests the size)")
        if self.K is not None:
            if self.maps is None:
                h, w = f.shape[:2]
                self.maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_16SC2)
            f = cv2.remap(f, *self.maps, cv2.INTER_LINEAR)
        if self.rot:
            f = cv2.rotate(f, self.ROT[self.rot])
        if getattr(self, "mirror", False):
            f = cv2.flip(f, 1)
        h, w = f.shape[:2]
        if self.crop:
            x0, y0, x1, y1 = self.crop
        else:
            s = min(h, w); x0, y0 = (w - s) // 2, (h - s) // 2; x1, y1 = x0 + s, y0 + s
        f = cv2.resize(f[y0:y1, x0:x1], (self.size, self.size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    def close(self):
        self.stop.set()
        if self.cap is not None:
            time.sleep(.1); self.cap.release()


class RealCameras:
    def __init__(self, indices: dict, undistort: dict | None = None, crops: dict | None = None, server: str | None = None,
                 rot: dict | None = None, mirror: tuple = (), expect: dict | None = None):
        undistort, crops, rot, expect = undistort or {}, crops or {}, rot or {}, expect or {}
        if "top" in mirror:
            raise ValueError("the top view is used for pointing: never mirror it")
        self.cams = {v: Cam(i, undistort.get(v), crops.get(v), server=server, rot=int(rot.get(v, 0)), mirror=v in mirror, expect=expect.get(v))
                     for v, i in indices.items()}

    def frames(self, views) -> dict:
        return {v: self.cams[v].get() for v in views}

    def close(self):
        for c in self.cams.values():
            c.close()
