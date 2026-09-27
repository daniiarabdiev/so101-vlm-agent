"""Camera server: owns the USB / Continuity cameras and serves their newest frames on localhost, so programs started from
an app without macOS camera permission (the Claude app's shell) can still read them. Run it in a terminal app that has
camera permission (Terminal.app asks once: click Allow), from vlm_policy/:

    .venv/bin/python -m real.cam_server 0,1,2,3          (OpenCV AVFoundation indices)

OpenCV's index order is NOT ffmpeg's device-list order (2026-09-26: OpenCV 2 was the MacBook camera while ffmpeg listed
an icspring there), so cameras are identified by their images, never by name. Indices shift when a camera (the iPhone)
comes or goes: after any replug, restart this server and look at every view again. A camera that drops is reopened at the
same index only if it delivers its previous frame size again; until then its frames are errors and the agent stops.
GET /list                      -> {"<index>": {"width", "height", "age_s", "error"}}
GET /frame/<index>?max_age=0.4 -> PNG of the newest raw frame (header X-Age: seconds since capture); 503 if no frame
                                  that fresh arrives within 3 s (the jev_arm rule: never act on stale images)
Clients: real/cameras.py with server="http://127.0.0.1:8766" (run_real.py / selfcal_real.py --cam-server).
Ctrl-C stops it. It never touches the arm.
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2



class Source:
    def __init__(self, index: int, size: tuple | None = None):
        self.index, self.size = index, size
        t0 = time.monotonic()
        while True:   # the first run triggers macOS's camera prompt: keep retrying while the user clicks Allow
            self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if self.cap.isOpened():
                if size:   # e.g. 4:3 on an iPhone: the full sensor height, a wider view than 16:9 video
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0]); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
                break
            if time.monotonic() - t0 > 120:
                raise RuntimeError(f"cannot open camera {index}: camera permission for this terminal app?")
            print(f"[cam] waiting for camera {index}: if macOS asks for camera access, click Allow", flush=True)
            time.sleep(3)
        self.lock, self.frame, self.t, self.err = threading.Lock(), None, 0.0, None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last_fix = 0.0
        while True:
            ok, f = self.cap.read()
            if ok and self.size and (f.shape[1], f.shape[0]) != tuple(self.size) and time.monotonic() - last_fix > 2:
                # another app opening the camera can switch its format (2026-09-26: the iPhone fell back from 4:3 to
                # 16:9 mid-session and every pointed position was centimetres off): ask again; clients check the size too
                print(f"[cam] camera {self.index} is {f.shape[1]}x{f.shape[0]}, requesting {self.size[0]}x{self.size[1]} again", flush=True)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0]); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1]); last_fix = time.monotonic()
            if ok and f.max() > 0:   # all-black frames (e.g. an iPhone that has not started streaming) never count
                with self.lock:
                    self.frame, self.t, self.err = f, time.monotonic(), None
                continue
            if ok:
                continue
            # dropped (the iPhone link, a loose USB cable): reopen the same index, but accept it only if it delivers the frame
            # size it had before (the iPhone's 1920x1440 is unique here), so a renumbered camera is never served in its place
            shape = None if self.frame is None else self.frame.shape
            self.err = f"camera {self.index} disconnected; reconnecting"; print(f"[cam] {self.err}", flush=True); self.cap.release()
            tries = 0
            while True:
                time.sleep(2); tries += 1
                if tries > 3:   # 2026-09-27: a dropped USB camera never came back to this process (reopening the same index
                    # failed for 27 min) while a fresh process opened it at once: restart the whole server in place (same
                    # process and terminal, so the camera permission holds); clients retry their fetches meanwhile
                    print(f"[cam] camera {self.index} did not come back: restarting the camera server", flush=True)
                    import os, sys
                    os.execv(sys.executable, [sys.executable, "-m", "real.cam_server", *sys.argv[1:]])
                cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
                if not cap.isOpened():
                    continue
                if self.size:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                good = False
                for _ in range(30):
                    ok, f = cap.read()
                    if ok and f.max() > 0:
                        good = shape is None or f.shape == shape; break
                if good:
                    self.cap = cap; print(f"[cam] camera {self.index} reconnected ({f.shape[1]}x{f.shape[0]})", flush=True); break
                cap.release()

    def newest(self, max_age: float):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3:
            with self.lock:
                f, t = self.frame, self.t
            if f is not None and time.monotonic() - t <= max_age:
                return f, time.monotonic() - t
            time.sleep(.01)
        return None, self.err or f"camera {self.index}: no fresh frame"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("indices", nargs="?", default="1,2,3"); ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--size", default="", help="index=WxH,... requested capture size (e.g. 3=1920x1440)")
    a = ap.parse_args()
    sizes = {int(k): tuple(int(v) for v in wh.split("x")) for k, wh in (kv.split("=") for kv in a.size.split(",") if kv)}

    def _watchdog(srcs):   # 2026-09-27: reopening a dropped USB camera can hang inside OpenCV, so the capture thread never gets to
        # its own restart; a fresh process opens it at once. Restart the server in place when any camera's newest frame is
        # older than 8 s (same process and terminal, so the camera permission holds); clients retry their fetches meanwhile.
        import os, sys
        time.sleep(20)
        while True:
            time.sleep(1)
            for i_, c in srcs.items():
                if c.frame is not None and time.monotonic() - c.t > 8.0:
                    print(f"[cam] camera {i_} has had no frame for {time.monotonic() - c.t:.0f} s: restarting the camera server", flush=True)
                    os.execv(sys.executable, [sys.executable, "-m", "real.cam_server", *sys.argv[1:]])
    try:   # Center Stage (Continuity Camera) reframes the view when it sees a person: calibration-breaking (2026-09-27).
        import AVFoundation as AV   # take app control of it and keep it off for this process
        before = bool(AV.AVCaptureDevice.isCenterStageEnabled())
        AV.AVCaptureDevice.setCenterStageControlMode_(1)   # AVCaptureCenterStageControlModeApp
        AV.AVCaptureDevice.setCenterStageEnabled_(False)
        print(f"[cam] Center Stage was {'ON' if before else 'off'}; now app-controlled and "
              f"{'ON' if AV.AVCaptureDevice.isCenterStageEnabled() else 'off'}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[cam] could not control Center Stage: {exc!r}", flush=True)
    def keep_center_stage_off():   # re-assert every 2 s and record the state for clients (real/runs/cam_status.json)
        status = Path(__file__).parent / "runs" / "cam_status.json"
        while True:
            try:
                import AVFoundation as AV
                was = bool(AV.AVCaptureDevice.isCenterStageEnabled()); mode = int(AV.AVCaptureDevice.centerStageControlMode())
                if was or mode != 1:
                    AV.AVCaptureDevice.setCenterStageControlMode_(1); AV.AVCaptureDevice.setCenterStageEnabled_(False)
                status.write_text(json.dumps({"t": time.time(), "center_stage_was_on": was, "control_mode_was": mode,
                                              "center_stage_now": bool(AV.AVCaptureDevice.isCenterStageEnabled())}))
            except Exception as exc:  # noqa: BLE001
                status.write_text(json.dumps({"t": time.time(), "error": repr(exc)[:200]}))
            time.sleep(2)
    threading.Thread(target=keep_center_stage_off, daemon=True).start()
    src = {i: Source(i, sizes.get(i)) for i in (int(v) for v in a.indices.split(","))}
    threading.Thread(target=_watchdog, args=(src,), daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path); parts = u.path.strip("/").split("/")
            if parts == ["list"]:
                now = time.monotonic()
                out = {str(i): {"width": None if s.frame is None else s.frame.shape[1],
                                "height": None if s.frame is None else s.frame.shape[0],
                                "age_s": None if s.frame is None else round(now - s.t, 3), "error": s.err} for i, s in src.items()}
                return self._send(200, json.dumps(out).encode(), "application/json")
            if len(parts) == 2 and parts[0] == "frame" and parts[1].isdigit() and int(parts[1]) in src:
                max_age = float(parse_qs(u.query).get("max_age", ["0.4"])[0])
                f, info = src[int(parts[1])].newest(max_age)
                if f is None:
                    return self._send(503, info.encode(), "text/plain")
                q = parse_qs(u.query)
                if q.get("fmt", ["png"])[0] == "jpg":   # logging (hand sessions): ~10x smaller than PNG
                    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, int(q.get("q", ["92"])[0])])
                    return self._send(200, jpg.tobytes(), "image/jpeg", {"X-Age": f"{info:.3f}"})
                ok, png = cv2.imencode(".png", f, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                return self._send(200, png.tobytes(), "image/png", {"X-Age": f"{info:.3f}"})
            return self._send(404, b"unknown path", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    for i, s in src.items():
        print(f"[cam] opened OpenCV camera {i}", flush=True)
    print(f"[cam] serving on http://127.0.0.1:{a.port} (Ctrl-C stops)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for s in src.values():
            s.cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
