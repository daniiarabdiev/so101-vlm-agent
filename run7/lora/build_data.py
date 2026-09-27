"""LoRA v3 / v4 training data (run7/BRIEF.md D). v3 and v4 use exactly the same states and differ only in labels:
  --labels rule     (v3, tier 1) the oracle's gold step and the simulator's rule truths (Run 6 labelling)
  --labels outcome  (v4, tier 2) the branching labels from run7.lora.outcome: next step = best-scoring step (DAgger states);
                    "over the destination?" = does releasing right here succeed; on replay / twin states a release vs
                    move_to_place label follows that release outcome. "done?" is the simulator's truth in both.
State selection never looks at the label set: DAgger states where the agent's own choice disagreed with either label come
first, then the rest (fixed seed).

Sources (tuning tasks and training seeds only; gather, next_to, in_then_out and every test seed are never used):
  dagger   run7/cells/dagger: T0 rollouts with LoRA v2, seeds 90000+ (PNGs saved per step)
  replay   state files with snapshots, re-rendered without held-out looks: Run 5 v1 states (subset), Run 6 near-miss
           twins (80000+), Run 7 take_out twins (80000+)
  point_on block-top pointing (the agent asks the LoRA for it: run6 DECISIONS #11)
  parse    instruction -> sub-task JSON for the tuning tasks (incl. take_out)
Usage: python -m run7.lora.build_data OUT.jsonl --labels rule|outcome --dagger CELL --dagger-q Q.jsonl
         --replay NAME STATES IMAGES OUTCOMES [--replay ...] [caps]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from run3.phase2_photoreal import mjexport
from run5.agent.models import formatted
from run7.agent import prompts as P
from run7.env.policy import STEPS
from run7.env.world import TUNING_TASKS, sample_scene
from run7.lora.outcome import q_label

VIEWS = P.VIEWS["top+side+wrist"]
TOP_CAM = {"pos": [.20, 0, .68], "xmat": [1, 0, 0, 0, 1, 0, 0, 0, 1], "fovy_deg": 48.0, "width": 448, "height": 448}


def load_jsonl(p: Path) -> dict:
    out = {}
    if p and p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line); out[r["sid"]] = r
    return out


def ex(eid, kind, images, prompt, answer, source):
    return {"id": eid, "kind": kind, "images": images, "prompt": prompt, "answer": answer, "source": source}


def dagger_pool(cell: Path, q: dict, labels: str) -> tuple[list, list, list]:
    """(disagree next, other next, checks) with the chosen labels."""
    dis, oth, chk = [], [], []
    for summ in sorted(cell.glob("*/*/seed_*/summary.json")):
        s = json.loads(summ.read_text())
        if s["task"] not in TUNING_TASKS or not s.get("subtasks_correct"):
            continue
        d = summ.parent; subs = s["subtasks"]
        for line in (d / "episode.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if "text" not in rec or rec["subtask"] >= len(subs) or rec.get("facts") is None or rec.get("gold") is None:
                continue
            imgs = [str(d / f"{rec['step']:03d}_{v}.png") for v in VIEWS]
            if not all(Path(p).exists() for p in imgs):
                continue
            k, f, text = rec["subtask"], rec["facts"], rec["text"]
            sid = f"dagger7|{s['task']}|{s.get('fault') or 'clean'}|{s['seed']}|{rec['step']}"
            qr = q.get(sid, {})
            rule = rec["gold"]
            qlab = q_label(qr["q"], rule, bool(f["satisfied"])) if qr.get("q") else rule  # (recomputed: run7 #13)
            plan = rec.get("plan") or {}
            planned = {"auto_verify": "done", "place_check_first": "release"}.get(plan.get("mode")) or \
                (STEPS[ord(plan["label"]) - 65] if plan.get("label") else None)
            lab = rule if labels == "rule" else qlab
            e = ex(f"{sid}|next", "next", imgs, formatted(text, P.next_step_options(subs[k]), P.NEXT_Q), P.LABELS[STEPS.index(lab)], "dagger7")
            (dis if planned is not None and planned not in (rule, qlab) else oth).append(e)
            truths = {}
            if rec.get("holding") is None:
                truths["done"] = bool(f["satisfied"])
            elif f.get("holding_target"):
                over = f["over_dest"] if labels == "rule" or qr.get("over_outcome") is None else qr["over_outcome"]
                truths["over_dest"] = bool(over)
            for kind, t in truths.items():
                chk.append(ex(f"{sid}|check_{kind}", f"check_{kind}", imgs, formatted(text, P.YES_NO, P.check_question(kind, subs[k])),
                              "A" if t else "B", "dagger7"))
    return dis, oth, chk


def replay_pool(name: str, states: Path, images: Path, outcomes: dict, labels: str) -> tuple[list, list, list]:
    """(next, checks, point_on) from rendered snapshot states."""
    nxt, chk, pon = [], [], []
    for line in states.read_text().splitlines():
        r = json.loads(line)
        if r["task"] not in TUNING_TASKS or r["k"] >= len(r["subs"]):
            continue
        sid, subs, k = r["sid"], r["subs"], r["k"]
        imgs = [str(images / f"{sid}_{v}.png") for v in VIEWS]
        if not all(Path(p).exists() for p in imgs):
            continue
        text = P.context(r["instruction"], subs, k, r["last"], r["gripper"], VIEWS)
        gold, f = r["gold"], r.get("facts") or {}
        oc = (outcomes.get(sid) or {}).get("over_outcome")
        if labels == "outcome" and oc is not None and gold in ("release", "move_to_place") and f.get("holding_target"):
            gold = "release" if oc else "move_to_place"
        nxt.append(ex(f"{name}|{sid}|next", "next", imgs, formatted(text, P.next_step_options(subs[k]), P.NEXT_Q),
                      P.LABELS[STEPS.index(gold)], name))
        for kind in ("over_dest", "done"):
            t = r.get(f"check_{kind}")
            if t is None:
                continue
            if kind == "over_dest" and labels == "outcome" and oc is not None:
                t = oc
            chk.append(ex(f"{name}|{sid}|check_{kind}", f"check_{kind}", imgs, formatted(text, P.YES_NO, P.check_question(kind, subs[k])),
                          "A" if t else "B", name))
        tg = r.get("targets") or {}
        if subs[k]["relation"] == "on" and f.get("holding_target") and tg.get("dest_xy"):
            u, v = mjexport.project(TOP_CAM, [tg["dest_xy"][0], tg["dest_xy"][1], .035])
            if 0 <= u <= 448 and 0 <= v <= 448:
                pon.append(ex(f"{name}|{sid}|point_on", "point", [str(images / f"{sid}_top.png")],
                              P.POINT_ONE.format(noun=P.dest_phrase(subs[k])),
                              json.dumps([{"point_2d": [int(round(u / 448 * 1000)), int(round(v / 448 * 1000))], "label": "target"}]), name))
    return nxt, chk, pon


def parse_examples(n: int, seed0: int = 20_500) -> list[dict]:
    out = []
    for s in range(n):
        task = TUNING_TASKS[s % len(TUNING_TASKS)]
        sc = sample_scene(seed0 + s, task)
        out.append(ex(f"parse-{task}-{seed0 + s}", "parse", [], P.PARSE_PROMPT.format(instruction=sc["instruction"]),
                      json.dumps({"subtasks": P.subtasks_from_scene(sc)}), "parse"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path); ap.add_argument("--labels", choices=["rule", "outcome"], required=True)
    ap.add_argument("--dagger", type=Path, required=True); ap.add_argument("--dagger-q", type=Path, required=True)
    ap.add_argument("--replay", nargs=4, action="append", default=[], metavar=("NAME", "STATES", "IMAGES", "OUTCOMES"))
    ap.add_argument("--dagger-next", type=int, default=1300); ap.add_argument("--dagger-check", type=int, default=1000)
    ap.add_argument("--replay-caps", default="v1:500:300,nm:500:500,nm7:400:400")
    ap.add_argument("--point-on", type=int, default=200); ap.add_argument("--parse", type=int, default=250)
    a = ap.parse_args()
    rng = random.Random(7)
    q = load_jsonl(a.dagger_q)
    dis, oth, dchk = dagger_pool(a.dagger, q, a.labels)
    for lst in (dis, oth, dchk):
        lst.sort(key=lambda e: e["id"]); rng.shuffle(lst)
    chosen = dis[:a.dagger_next] + oth[:max(0, a.dagger_next - len(dis))] + dchk[:a.dagger_check]
    caps = {c.split(":")[0]: tuple(map(int, c.split(":")[1:])) for c in a.replay_caps.split(",")}
    stats = {"dagger_disagree": min(len(dis), a.dagger_next), "dagger_other": min(len(oth), max(0, a.dagger_next - len(dis))),
             "dagger_check": min(len(dchk), a.dagger_check)}
    pons = []
    for name, states, images, outc in a.replay:
        nxt, chk, pon = replay_pool(name, Path(states), Path(images), load_jsonl(Path(outc)), a.labels)
        for lst in (nxt, chk, pon):
            lst.sort(key=lambda e: e["id"]); rng.shuffle(lst)
        cn, cc = caps.get(name, (len(nxt), len(chk)))
        chosen += nxt[:cn] + chk[:cc]; pons += pon
        stats[f"{name}_next"], stats[f"{name}_check"] = min(cn, len(nxt)), min(cc, len(chk))
    pons.sort(key=lambda e: e["id"]); rng.shuffle(pons)
    chosen += pons[:a.point_on] + parse_examples(a.parse)
    stats.update(point_on=min(len(pons), a.point_on), parse=a.parse, labels=a.labels)
    rng.shuffle(chosen)
    stats["total"] = len(chosen)
    stats["answer_letters"] = {}
    for e in chosen:
        if e["kind"] == "next":
            stats["answer_letters"][STEPS[ord(e["answer"]) - 65]] = stats["answer_letters"].get(STEPS[ord(e["answer"]) - 65], 0) + 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(e) + "\n" for e in chosen))
    (a.out.parent / (a.out.stem + "_stats.json")).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
