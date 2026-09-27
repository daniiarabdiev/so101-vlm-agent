#!/bin/bash
# One real-arm episode with the day-2 setup (2026-09-27): bash real/run_task.sh "<command>" [pod_id]
# Camera + joint offsets from real/kin_cal.py (held-out 3.3 mm median), wide top view, fast profile, video + ledger.
cd "$(dirname "$0")/.."
BASE=${MODEL_BASE_URL:?set MODEL_BASE_URL to your vLLM server, e.g. https://<pod-id>-8000.proxy.runpod.net}
OUT=real/runs/ep_$(date +%Y%m%d-%H%M%S)
exec .venv/bin/python -m real.run_real "$1" --backend socket --base $BASE \
  --camera-json real/cal/current_camera.json --cams top=3,side=1,wrist=0 --rotate top=270 --crop top=0:120:1440:1560 \
  --expect top=1920x1440 --mirror side --table-z -0.0098 --carry-z 0.18 --object-z 0.036 --grasp-clear 0.005 --no-drift-guard \
  --ref-frame real/cal/ref_top_wide.png --profile fast --start-from-rest --out $OUT
