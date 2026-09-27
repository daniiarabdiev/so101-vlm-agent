"""Run 8 world = the Run 7 world plus a calibration marker: a small cyan sphere on the wrist link (the real-arm equivalent is
a cyan sticker at the same spot). It is visual only (no contacts, no mass) and visible to the top camera in every
calibration pose (searched by ray casting over 27 reachable poses; run8/DECISIONS.md #5). Importing this module makes
every Run 7 World in the process carry the marker (the Run 7 build_xml is wrapped), so snapshots stay compatible.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

import run7.env.world as W7
from run7.env.world import *  # noqa: F401,F403

MARKER_BODY = "wrist"
MARKER_LOCAL = np.array([0.0145, 0.010, 0.0448])   # sphere centre in the wrist-link frame (surface at x = 0.0099)
MARKER_RADIUS = 0.008
MARKER_RGB = (0.0, 0.70, 0.75)

_build_xml7 = W7.build_xml


def build_xml(scene: dict, width: int = 448, height: int = 448) -> str:
    root = ET.fromstring(_build_xml7(scene, width, height))
    body = root.find(f".//body[@name='{MARKER_BODY}']")
    ET.SubElement(body, "geom", name="calib_marker", type="sphere", size=f"{MARKER_RADIUS}",
                  pos=" ".join(f"{v:.5f}" for v in MARKER_LOCAL), rgba=f"{MARKER_RGB[0]} {MARKER_RGB[1]} {MARKER_RGB[2]} 1",
                  contype="0", conaffinity="0", mass="0", group="1")
    return ET.tostring(root, encoding="unicode")


W7.build_xml = build_xml  # every World built after this import has the marker


def marker_world(env) -> np.ndarray:
    """The marker's 3D position from forward kinematics (what the robot knows without any camera)."""
    bid = env.model.body(MARKER_BODY).id
    return env.data.xpos[bid] + env.data.xmat[bid].reshape(3, 3) @ MARKER_LOCAL
