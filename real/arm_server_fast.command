#!/bin/bash
# Arm server in its own Terminal window (FAST: 60 deg/s, gripper 50 %/s). This window is the STOP window: press Enter here to stop.
cd "$(dirname "$0")/.." && exec ${SO101_PYTHON:-python3} real/arm_server.py --body-deg-s ${BODY:-60} --grip-pct-s ${GRIP:-50} --lead-deg ${LEAD:-14}
