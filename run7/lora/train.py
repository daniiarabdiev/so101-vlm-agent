"""Run 7 copy of run5/lora/train.py with one addition: --init ADAPTER continues training an existing LoRA (v3 and v4
start from v2; run7/BRIEF.md D). Everything else is the Run 5 recipe.

LoRA fine-tuning of Qwen3.8-27B on Run 5 simulated SO-101 examples (one A100 80 GB, bf16, gradient checkpointing).

- Base weights frozen, vision tower frozen. LoRA (rank 16, alpha 32, dropout 0.05) on the language model's MLPs in all
  64 layers and the q/k/v/o projections of the 16 full-attention layers. The 48 linear-attention layers get no adapters,
  to stay within what the vLLM LoRA runtime can serve.
- Each example is tokenized exactly like inference: the chat template with add_generation_prompt=True and
  enable_thinking=False (as vLLM renders the request), then the answer tokens + <|im_end|>. Loss on the answer only.
- One epoch, AdamW lr 1e-4, 3% warmup then cosine decay, gradient accumulation 16, examples ordered randomly.
Usage: python -m run5.lora.train DATA.jsonl OUT_DIR [--limit N] [--epochs 1] [--time-probe]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image

MODEL_DIR = glob.glob("/root/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/*/")[0] if glob.glob("/root/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/*/") else None
TARGET = r".*language_model.*\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))"


def encode(processor, ex: dict, device):
    content = [{"type": "text", "text": ex["prompt"]}] + [{"type": "image", "image": Image.open(p).convert("RGB")} for p in ex["images"]]
    msgs = [{"role": "user", "content": content}]
    enc = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                        enable_thinking=False)
    ans = processor.tokenizer(ex["answer"] + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([enc["input_ids"], ans], 1)
    labels = torch.cat([torch.full_like(enc["input_ids"], -100), ans], 1)
    batch = {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device), "labels": labels.to(device)}
    plen = int(enc["input_ids"].shape[1])
    for k, v in enc.items():
        if k in ("input_ids", "attention_mask") or not torch.is_tensor(v):
            continue
        if v.dim() == 2 and v.shape == (1, plen):  # per-token fields (e.g. mm_token_type_ids): answer tokens are text (0)
            v = torch.cat([v, torch.zeros((1, ans.shape[1]), dtype=v.dtype)], 1)
        batch[k] = v.to(device)
    return batch, plen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--time-probe", action="store_true")
    ap.add_argument("--init", type=Path, default=None, help="start from this LoRA adapter (same rank / modules)")
    a = ap.parse_args()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(0); random.seed(0)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.init:
        model = PeftModel.from_pretrained(model, str(a.init), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=.05, target_modules=TARGET, bias="none"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    exs = [json.loads(l) for l in a.data.read_text().splitlines()]
    if a.limit:
        exs = exs[:a.limit]
    steps_total = max(1, int(len(exs) * a.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    n_upd = max(1, steps_total // a.accum); warm = max(1, int(.03 * n_upd))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * .5 * (1 + math.cos(math.pi * min(1.0, u / n_upd))))
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / "train_log.jsonl").open("a")
    model.train(); t0 = time.time(); tokens = 0; run_loss = []
    order = [exs[i % len(exs)] for i in range(steps_total)]
    random.shuffle(order)
    for step, ex in enumerate(order):
        batch, plen = encode(processor, ex, "cuda")
        out = model(**batch)
        (out.loss / a.accum).backward()
        tokens += batch["input_ids"].shape[1]; run_loss.append(float(out.loss))
        if (step + 1) % a.accum == 0 or step + 1 == len(order):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0 or a.time_probe or step + 1 == len(order):
            rec = {"step": step + 1, "of": len(order), "loss": sum(run_loss) / len(run_loss), "lr": sched.get_last_lr()[0],
                   "tokens": tokens, "s": time.time() - t0, "s_per_example": (time.time() - t0) / (step + 1),
                   "prompt_len": plen, "kind": ex["kind"], "trainable": n_train, "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True); run_loss = []
        if a.time_probe and step >= 5:
            break
        if (step + 1) % 1000 == 0:
            model.save_pretrained(a.out / "checkpoint")
    model.save_pretrained(a.out)
    (a.out / "train_meta.json").write_text(json.dumps({"examples": len(exs), "steps": len(order), "rank": a.rank, "lr": a.lr, "accum": a.accum,
                                                       "target_modules": TARGET, "model_dir": MODEL_DIR, "init": str(a.init) if a.init else None, "seconds": time.time() - t0}, indent=1))
    print("TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
