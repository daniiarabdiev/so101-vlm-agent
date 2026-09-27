#!/bin/bash
# usage (model Pod): bash run8/pods/start_model8.sh   (Run 8 serving: CUDA graphs + FP8; base + run7-v4; runtime LoRA loading on)
# Same image, revision and core flags as Runs 5-7; changes: no --enforce-eager, --quantization fp8, more sequences (run8 #3).
export HF_HOME=/root/hf VLLM_ENABLE_CUDA_COMPATIBILITY=0 VLLM_API_KEY=$(cat /root/.vllm_key) VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
setsid nohup vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --served-model-name Qwen/Qwen3.8-27B --dtype bfloat16 \
  --host 0.0.0.0 --port 8000 --max-model-len 16384 --max-num-seqs ${SEQS:-48} --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":24}' --enable-prefix-caching --trust-remote-code --return-tokens-as-token-ids \
  --max-logprobs 20 --enable-lora --max-lora-rank 64 --max-loras 4 --quantization fp8 \
  --lora-modules run7-v4=/root/lora_v4 ${EXTRA_LORAS} > /root/vllm8.log 2>&1 < /dev/null &
echo "started vllm run8 (fp8, cuda graphs)"
