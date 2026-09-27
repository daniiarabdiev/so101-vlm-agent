"""Arm backends with one interface, in real joint units (5 body joints in degrees, gripper %):
  read() -> 6 values        begin_hold() -> (q, new)   send(pose) -> the bounded goal actually written
  wait(dt)                  stop_hold()         close()
SocketBackend  the physical arm, through real/arm_server.py (runs in the Hiwonder LeRobot environment and owns the serial
               bus, the driver's safety clamps, a heartbeat watchdog and the stop key)
SimBackend     the MuJoCo world as a stand-in for dry runs: its position servos follow the same commands at the same rate,
               so the whole adapter (kinematics, speed limits, touch rule, gripper logic, agent loop) runs without hardware
"""
from __future__ import annotations

import json
import socket
import threading
import time

import mujoco
import numpy as np

import run8.env.world  # noqa: F401  (the Run 8 world: calibration marker)
from real.joints import real_to_sim, sim_to_real
from run7.env.world import World


class SocketBackend:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 5.0, heartbeat: bool = True):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.f = self.sock.makefile("rw")
        self.lock, self.last, self.closed = threading.Lock(), time.monotonic(), False
        hello = self._call({"op": "hello"})
        if not hello.get("ok"):
            raise RuntimeError(f"arm server refused: {hello}")
        self.info = hello
        if heartbeat:   # keep the hold (and the gripper's squeeze) alive while the agent waits for the model; the server's
            threading.Thread(target=self._beat, daemon=True).start()   # watchdog still fires if this process dies or hangs

    def _beat(self):
        while not self.closed:
            time.sleep(.1)
            if time.monotonic() - self.last > .2:
                try:
                    self._call({"op": "ping"})
                except Exception:  # noqa: BLE001  (stop key / closed socket: the main thread will see it on its next call)
                    pass

    def _call(self, msg: dict) -> dict:
        with self.lock:
            self.f.write(json.dumps(msg) + "\n"); self.f.flush()
            line = self.f.readline(); self.last = time.monotonic()
        return self._reply(line)

    def _reply(self, line: str) -> dict:
        if not line:
            raise RuntimeError("arm server closed the connection")
        r = json.loads(line)
        if r.get("stopped"):
            raise StopRequested(r.get("reason", "stop"))
        if r.get("error"):
            raise RuntimeError(f"arm server: {r['error']}")
        return r

    def read(self) -> list[float]:
        return self._call({"op": "read"})["q"]

    def begin_hold(self) -> tuple[list[float], bool]:
        """(current joints, whether a new hold was primed: the server's watchdog ends holds while the agent thinks)."""
        r = self._call({"op": "hold"})
        return r["q"], bool(r.get("new", True))

    def send(self, pose) -> list[float]:
        return self._call({"op": "send", "q": [float(v) for v in pose]})["q"]

    def wait(self, dt: float):
        time.sleep(dt)
        self._call({"op": "ping"})  # heartbeat: the server holds position if the client goes silent

    def stop_hold(self):
        try:
            self._call({"op": "stop"})
        except StopRequested:
            pass

    def close(self):
        try:
            self.stop_hold()
        finally:
            self.closed = True
            try:
                self.sock.close()
            except OSError:
                pass


class StopRequested(RuntimeError):
    """The operator pressed the stop key on the arm server: the arm holds its position with torque on."""


class SimBackend:
    """The simulator as the arm. Commands set the position-servo targets; wait() steps the physics."""

    def __init__(self, scene: dict):
        self.world = World(scene); self.world.reset()
        self.armed = False

    def read(self) -> list[float]:
        return sim_to_real(self.world.data.qpos[:6]).tolist()

    def begin_hold(self) -> tuple[list[float], bool]:
        new = not self.armed
        if new:  # like the driver: the current position becomes the goal (gripper included)
            self.world.data.ctrl[:6] = self.world.data.qpos[:6]; self.armed = True
        return self.read(), new

    def send(self, pose) -> list[float]:
        if not self.armed:
            raise RuntimeError("hold not active")
        q = real_to_sim(pose)
        lo, hi = self.world.model.jnt_range[:6, 0], self.world.model.jnt_range[:6, 1]
        q = np.clip(q, lo, hi)
        self.world.data.ctrl[:6] = q; self.world.grip_target = float(q[5])
        return sim_to_real(q).tolist()

    def wait(self, dt: float):
        for _ in range(max(1, round(dt / self.world.model.opt.timestep))):
            mujoco.mj_step(self.world.model, self.world.data)

    def stop_hold(self):
        self.world.data.ctrl[:6] = self.world.data.qpos[:6]; self.armed = False

    def close(self):
        self.world.close()

    # dry-run extras (never available on hardware)
    def full_qpos(self) -> np.ndarray:
        return self.world.data.qpos.copy()
