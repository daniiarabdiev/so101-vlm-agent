"""Contact sheet of a calibration sweep (real/fullcal.py): every pose's top-view overlay (found marker = green circle, the
installed calibration's prediction = red cross; after a fit, --cal adds the new prediction as a blue cross) next to its side
view, in pages of 24, for checking the detections and the arm's poses by eye.
Usage: .venv/bin/python -m real.contact_sheet real/runs/full_<time> [--cal real/cal/<fit>.json] [--json real/cal/full_<time>.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--json", default="")
    ap.add_argument("--cal", default="", help="fitted calibration (<json>_kin.json): its joint offsets + camera give the blue cross")
    ap.add_argument("--per-page", type=int, default=24); ap.add_argument("--thumb", type=int, default=224)
    a = ap.parse_args()
    run = Path(a.run); ovs = sorted(run.glob("ov_*.jpg"))
    pairs = json.loads(Path(a.json).read_text())["pairs"] if a.json else None
    new_px = {}
    if a.cal and pairs:
        import mujoco
        import run8.env.world as RW
        from real.arm import ArmConfig, RealArm
        from real.backend import SimBackend
        from real.joints import OFFSETS_DEG, real_to_sim
        from real.selfcal_real import jaw_tip_local
        from run3.phase2_photoreal import mjexport
        from run7.env.world import sample_scene
        c = json.loads(Path(a.cal).read_text()); cam = c["camera"]; off = np.asarray(c["joint_offsets_deg"], float)
        dm = np.asarray(c.get("marker_shift_mm", [0, 0, 0]), float) / 1000; dm = np.r_[dm, np.zeros(3 - len(dm))]
        arm = RealArm(SimBackend(sample_scene(60000, "place_in")), ArmConfig(table_z=-0.020), mirror=False)
        body = c.get("marker_body", "gripper")
        RW.MARKER_BODY = body; m0 = jaw_tip_local(arm, body, float(np.median([p["q"][5] for p in pairs]))).copy(); RW.MARKER_LOCAL[:] = m0 + dm
        for i, p in enumerate(pairs):
            q = np.asarray(p["q"], float).copy(); q[:5] += off
            arm.kin.data.qpos[:6] = real_to_sim(q) - np.r_[np.radians(OFFSETS_DEG), 0.0]; mujoco.mj_kinematics(arm.kin.model, arm.kin.data)
            new_px[i] = mjexport.project(cam, np.asarray(RW.marker_world(arm.kin)) - [0, 0, -0.020])
    tiles = []
    for ov in ovs:
        i = int(ov.stem.split("_")[1]); top = cv2.imread(str(ov))
        if i in new_px and new_px[i] is not None:
            cv2.drawMarker(top, (int(new_px[i][0]), int(new_px[i][1])), (255, 128, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        side = run / f"side_{i:03d}.jpg"
        s = cv2.resize(cv2.imread(str(side)), (448, 448)) if side.exists() else np.zeros_like(top)
        t = np.hstack([cv2.resize(top, (a.thumb, a.thumb)), cv2.resize(s, (a.thumb, a.thumb))])
        cv2.putText(t, str(i), (a.thumb + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        tiles.append(t)
    out = []
    for k in range(0, len(tiles), a.per_page):
        page = tiles[k:k + a.per_page]
        while len(page) % 4:
            page.append(np.zeros_like(tiles[0]))
        img = np.vstack([np.hstack(page[r:r + 4]) for r in range(0, len(page), 4)])
        p = run / f"sheet_{k // a.per_page:02d}.jpg"; cv2.imwrite(str(p), img); out.append(str(p))
    print(json.dumps({"sheets": out, "poses": len(tiles)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
