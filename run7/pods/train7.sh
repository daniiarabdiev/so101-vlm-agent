#!/bin/bash
# usage (training Pod, from /root/vp): setsid nohup bash run7/pods/train7.sh rule|outcome OUT_DIR > /root/train7.log 2>&1 &
# Build the v3 (rule) or v4 (outcome) data from the same states (run7/lora/build_data.py) and continue LoRA v2 on it
# (run7/lora/train.py --init /root/lora_run6; Run 5 recipe otherwise). Needs: run7/cells/dagger (+PNGs), run7/data/*.
labels=$1; out=$2
cd /root/vp
log(){ echo "$(date -u +%H:%M:%S) $*"; }
python3 -c "import peft, fla" 2>/dev/null || pip install -q peft flash-linear-attention > /root/pip_train.log 2>&1
python3 -c "import peft, fla; print('packages ok')" || { log "PACKAGES_FAILED"; exit 1; }
python3 -m run7.lora.build_data run7/data/lora_$labels.jsonl --labels $labels --dagger run7/cells/dagger \
  --dagger-q run7/data/outcome_dagger.jsonl \
  --replay v1 run7/data/replay_v1.jsonl run7/data/images_train run7/data/outcome_replay_v1.jsonl \
  --replay nm run6/data/nm_train.jsonl run7/data/images_train run7/data/outcome_nm_train.jsonl \
  --replay nm7 run7/data/nm7_train.jsonl run7/data/images_train run7/data/outcome_nm7_train.jsonl > /root/build_$labels.log 2>&1 \
  || { log "BUILD_FAILED"; tail -5 /root/build_$labels.log; exit 1; }
log "data $(tail -1 /root/build_$labels.log)"
HF_HOME=/root/hf python3 -u -m run7.lora.train run7/data/lora_$labels.jsonl $out --init /root/lora_run6 > /root/train_$labels.log 2>&1 \
  || { log "TRAIN_FAILED"; tail -5 /root/train_$labels.log; exit 1; }
sha256sum $out/adapter_model.safetensors $out/adapter_config.json
log "TRAIN_DONE"
