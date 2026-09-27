"""Real SO-101 joint units <-> the simulator's joint radians.

Real (driver, use_degrees=True): five body joints in degrees, gripper 0-100 % (hiwonder_follower calibration).
Sim (so101_vlm/assets/so101/so101.xml, the Menagerie SO-101 the agent was trained on): radians.
- Body joints: identity degrees -> radians (both GATE0 and the twin config revision 16 agree; owner-matched poses).
- Gripper: the twin revision-16 mapping q = 0.0191986 * p - 0.416667. Under it an empty close (15.3 %) reads -0.12 rad,
  the sim's closed value, so the agent's jaw-offset model (run8 jaw_offset) and the sim thresholds keep their meaning.
  run8/diag/REAL_SENSING.md: never combine GATE0's gripper mapping with sim radians (every miss would read "holding").
Holding is decided on the raw reading (>= 20 %, fitted on the jev_arm logs), not on radians.
"""
from __future__ import annotations

import math

import numpy as np

GRIP_SCALE, GRIP_OFFSET = 0.019198621771937624, -0.41666741838563637
GRIP_OPEN_PCT, GRIP_CLOSED_PCT = 95.0, 15.0   # open 95 (teleop reached 97, 2026-09-06; the 7 cm ball needs the width), command floor 15
HOLD_MIN_PCT = 20.0                           # REAL_SENSING: empty closes ~15.3 %, a held ball ~50 %

# Measured zero-point offsets of the real body joints (model joint = real reading + offset), from real/kin_cal.py; zeros
# when the file is absent. Every conversion between real degrees and model radians goes through these two functions.
try:
    import json as _json
    from pathlib import Path as _Path
    OFFSETS_DEG = np.asarray(_json.loads((_Path(__file__).parent / "cal" / "joint_offsets.json").read_text())["offsets_deg"], float)
except (OSError, KeyError, ValueError):
    OFFSETS_DEG = np.zeros(5)


def body_model_rad(p5) -> np.ndarray:
    """Real body joints (deg) -> model joint radians."""
    return np.radians(np.asarray(p5, float)[:5] + OFFSETS_DEG)


def body_real_deg(q5) -> np.ndarray:
    """Model joint radians -> real body joint commands (deg)."""
    return np.degrees(np.asarray(q5, float)[:5]) - OFFSETS_DEG


def grip_pct_to_rad(p: float) -> float:
    return GRIP_SCALE * float(p) + GRIP_OFFSET


def grip_rad_to_pct(q: float) -> float:
    return (float(q) - GRIP_OFFSET) / GRIP_SCALE


def real_to_sim(p) -> np.ndarray:
    p = np.asarray(p, float)
    return np.r_[body_model_rad(p[:5]), grip_pct_to_rad(p[5])]


def sim_to_real(q) -> np.ndarray:
    q = np.asarray(q, float)
    return np.r_[body_real_deg(q[:5]), grip_rad_to_pct(q[5])]


def deg(x) -> float:
    return float(math.degrees(x))
