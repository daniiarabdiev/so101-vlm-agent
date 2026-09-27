#!/bin/bash
# Arm server in its own Terminal window (commissioning speed). This window is the STOP window: press Enter here to stop.
cd "$(dirname "$0")/.." && exec ${SO101_PYTHON:-python3} real/arm_server.py --body-deg-s ${BODY:-10} --grip-pct-s ${GRIP:-12} --lead-deg ${LEAD:-14}
