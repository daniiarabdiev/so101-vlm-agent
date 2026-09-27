"""Set down whatever the gripper holds, then fold to the base pose (between episodes, e.g. after a run ends holding an object).
Lowers at the current spot (or --xy) until the table stops it, opens, lifts and folds.
Usage: .venv/bin/python -m real.put_down [--xy 0.35,-0.05] [--no-fold]
"""
from __future__ import annotations

import argparse
import json

from real.arm import PROFILES, ArmConfig, RealArm
from real.backend import SocketBackend


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--xy", default=""); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--profile", choices=list(PROFILES), default="fast"); ap.add_argument("--table-z", type=float, default=-0.0098)
    ap.add_argument("--no-fold", action="store_true")
    a = ap.parse_args()
    arm = RealArm(SocketBackend(port=a.port), ArmConfig(table_z=a.table_z, carry_z=.18, **PROFILES[a.profile]))
    arm._sync(); p = arm.ee()
    if a.xy:
        x, y = (float(v) for v in a.xy.split(",")); arm.move_xy([x, y]); p = arm.ee()
    d = arm._descend(.01, detect=False, lead=False)   # the object's own height stops the arm first
    arm.open_gripper()
    arm._move([p[0], p[1], max(p[2], .12)])
    print(json.dumps({"set_down_at": [round(float(p[0]), 3), round(float(p[1]), 3)], "lowest_z": round(float(d["z"]), 3)}), flush=True)
    if not a.no_fold:
        arm.fold(); print("[put_down] at the base pose", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
