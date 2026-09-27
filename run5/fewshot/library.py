"""Few-shot example library for readout (fixed worked examples shown before every question).

Real-life framing (user, 2026-09-24): examples ship with the product for this rig (SO-101 + top/side/wrist cameras) and
must help on tasks nobody wrote examples for. So the library holds generic *situations* taken from the four tuning
tasks only; the held-out gather task never contributes an example, and the tuning-task test scenes never do either.

  next         6 examples: grasp (aligned), rotate (bar misaligned), move_to_place (holding, not over),
               release (over the destination), open (closed on nothing), done (placed)
  check_over_dest   2 examples (yes / no)
  check_done        2 examples (yes / no)
Every example uses the same prompt builder as the agent (run5.agent.prompts), with its three images.

Build (after the tuning states are rendered): python -m run5.fewshot.library build TUNE_STATES IMAGES OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from run5.agent import prompts as P

NEXT_LABELS = ["grasp", "rotate", "move_to_place", "release", "open", "done"]


def _example(row: dict, images: Path, kind: str, views: list[str]) -> dict:
    subs, k = row["subs"], row["k"]
    text = P.context(row["instruction"], subs, k, row["last"], row["gripper"], views)
    ex = {"sid": row["sid"], "task": row["task"], "text": text, "views": views, "images": [str(images / f"{row['sid']}_{v}.png") for v in views]}
    if kind == "next":
        opts = P.next_step_options(subs[k])
        lab = P.LABELS[list(opts).index(next(l for l in opts if opts[l].startswith(row["gold"] + ":")))]
        return {**ex, "options": opts, "question": P.NEXT_Q, "answer": lab, "gold_step": row["gold"]}
    q = P.check_question(kind[len("check_"):], subs[k])
    return {**ex, "options": P.YES_NO, "question": q, "answer": "A" if row[kind] else "B"}


def build(states: Path, images: Path, out: Path, views_cfg: str = "top+side+wrist", seed: int = 7) -> dict:
    rows = [json.loads(l) for l in states.read_text().splitlines()]
    rows = [r for r in rows if r["task"] != "gather" and all((images / f"{r['sid']}_{v}.png").exists() for v in ("top", "side", "wrist"))]
    rng = np.random.default_rng(seed)
    views = P.VIEWS[views_cfg]
    lib, used_tasks = {"next": [], "check_over_dest": [], "check_done": []}, []
    for g in NEXT_LABELS:  # spread the examples over tasks
        cands = [r for r in rows if r["gold"] == g]
        fresh = [r for r in cands if r["task"] not in used_tasks[-2:]] or cands
        r = fresh[rng.integers(len(fresh))]
        used_tasks.append(r["task"])
        lib["next"].append(_example(r, images, "next", views))
    for kind in ("check_over_dest", "check_done"):
        for truth in (True, False):
            cands = [r for r in rows if r.get(kind) is truth]
            r = cands[rng.integers(len(cands))]
            lib[kind].append(_example(r, images, kind, views))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lib, indent=1))
    return lib


_CACHE: dict = {}


def load_shots(path: Path, views_cfg: str = "top+side+wrist") -> dict:
    """Library -> {kind: [shot]} with PIL images (cached: identical bytes every call keep the prefix cache warm)."""
    key = (str(path), views_cfg)
    if key not in _CACHE:
        lib = json.loads(Path(path).read_text())
        base = Path(path).parent
        shots = {}
        for kind, exs in lib.items():
            shots[kind] = []
            for e in exs:
                imgs = []
                for p in e["images"]:
                    q = Path(p) if Path(p).exists() else base / "images" / Path(p).name
                    imgs.append(Image.open(q).convert("RGB"))
                shots[kind].append({"text": e["text"], "images": imgs, "options": e["options"], "question": e["question"], "answer": e["answer"]})
        _CACHE[key] = shots
    return _CACHE[key]


if __name__ == "__main__":
    if sys.argv[1] == "build":
        lib = build(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        print({k: [(e["task"], e.get("gold_step", e["answer"])) for e in v] for k, v in lib.items()})
