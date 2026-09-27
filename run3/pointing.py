"""Molmo2 pointing output parsing (official html-v2 grammar, allenai/molmo2 point_formatter.py)."""
from __future__ import annotations

import re

COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
LEGACY_REGEX = re.compile(r"<point[^>]*x=\"([0-9.]+)\"[^>]*y=\"([0-9.]+)\"")


def parse_points(text: str, width: int, height: int) -> list[tuple[float, float]]:
    """All points in pixel coordinates of the (single) input image; [] when none parse."""
    points = []
    for coord in COORD_REGEX.finditer(text or ""):
        for grp in FRAME_REGEX.finditer(coord.group(1)):
            for m in POINTS_REGEX.finditer(grp.group(2)):
                x, y = float(m.group(2)) / 1000 * width, float(m.group(3)) / 1000 * height
                if 0 <= x <= width and 0 <= y <= height:
                    points.append((x, y))
    if not points:  # Molmo-1 fallback (0-100 percent)
        for m in LEGACY_REGEX.finditer(text or ""):
            x, y = float(m.group(1)) / 100 * width, float(m.group(2)) / 100 * height
            if 0 <= x <= width and 0 <= y <= height:
                points.append((x, y))
    return points
