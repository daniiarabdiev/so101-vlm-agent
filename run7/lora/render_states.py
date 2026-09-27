"""Run 7 copy of run5/screen/render_states.py (restores with the Run 7 world). Render the three camera views of saved states photoreal (on a Pod with Blender servers).

Usage: python -m run7.lora.render_states STATES.jsonl OUT_DIR --shard i/n --render-url URL [--no-heldout-looks]
Images: OUT_DIR/<sid>_<view>.png (448 x 448). Existing images are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

from PIL import Image

from run5.env.render import PhotoRenderer, photo_config
from run7.env.world import restore

VIEWS = ("top", "side", "wrist")
SAMPLES = 32  # Run 5 renders everything at 32 Cycles samples + denoising (DECISIONS #15)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--render-url", default="http://localhost:8002")
    ap.add_argument("--no-heldout-looks", action="store_true")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    rows = [json.loads(l) for l in a.states.read_text().splitlines()][i::n]
    a.out.mkdir(parents=True, exist_ok=True)
    r = PhotoRenderer(a.render_url)
    t0, done = time.time(), 0
    for row in rows:
        if all((a.out / f"{row['sid']}_{v}.png").exists() for v in VIEWS):
            continue
        env = restore(row["snapshot"])
        try:
            cfg = photo_config(env.scene, allow_heldout=not a.no_heldout_looks)
            for attempt in range(4):  # a render server can run out of GPU memory transiently: retry, never crash the shard
                try:
                    r.set_scene(env.model, cfg, force=attempt > 0)
                    views, _ = r.views(env.model, env.data, cfg, cameras=VIEWS, samples=SAMPLES,
                                       seed=zlib.crc32(row["sid"].encode()) % 100_000)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(json.dumps({"sid": row["sid"], "attempt": attempt, "error": str(exc)[-300:]}), flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                continue
            for v in VIEWS:
                Image.fromarray(views[v]).save(a.out / f"{row['sid']}_{v}.png")
            (a.out / f"{row['sid']}_look.json").write_text(json.dumps(cfg["factors"], default=str))
        finally:
            env.close()
        done += 1
        if done % 20 == 0:
            print(json.dumps({"done": done, "of": len(rows), "s_per_state": (time.time() - t0) / done}), flush=True)
    print("RENDER_DONE", done, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
