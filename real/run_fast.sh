#!/bin/bash
# One command in the fastest mode (2026-09-27): fastest profile (no settle on travel moves, 100 mm/s descents to calibrated heights), JPEG images to the model; arm server real/arm_server_fast.command (60 deg/s).
cd "$(dirname "$0")/.."
BASE=${MODEL_BASE_URL:?set MODEL_BASE_URL to your vLLM server, e.g. https://<pod-id>-8000.proxy.runpod.net}
OUT=real/runs/ep_$(date +%Y%m%d-%H%M%S)
exec .venv/bin/python -m real.run_real "$1" --backend socket --base $BASE \
  --camera-json real/cal/current_camera.json --cams top=3,side=1,wrist=0 --rotate top=270 --crop top=0:120:1440:1560 \
  --expect top=1920x1440 --mirror side --table-z -0.0098 --carry-z 0.18 --object-z 0.036 --grasp-clear 0.005 --place-clear 0.035 --no-drift-guard \
  --ref-frame real/cal/ref_top_wide.png --profile fastest --jpeg --start-from-rest --out $OUT
