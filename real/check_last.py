"""Print the latest episode's steps and save its final views (+ live top/side) to <episode>/final_check.jpg."""
import glob, json, os, sys, urllib.request
import cv2, numpy as np
ep = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("real/runs/ep_*"), key=os.path.getmtime)[-1]
print(ep)
if os.path.exists(f"{ep}/progress.jsonl"):
    for l in open(f"{ep}/progress.jsonl"):
        r = json.loads(l); o = r.get("outcome") or {}; loc = r.get("locate") or {}; o = o if isinstance(o, dict) else {"raw": str(o)}
        print(r["step"], r["executed"], "| xy", [round(v, 3) for v in loc.get("xy") or []], "| grip", round(o.get("gripper_pct", -1), 1),
              "hold", o.get("holding"), "|", (r.get("last") or "")[:60])
fs = sorted(glob.glob(f"{ep}/*_top.jpg"))
fr = [cv2.imread(f"{fs[-1][:-8]}_{v}.jpg") for v in ("top", "side", "wrist")] if fs else []
for i in (3, 1):
    d = urllib.request.urlopen(f"http://127.0.0.1:8766/frame/{i}?max_age=1&fmt=jpg", timeout=5); fr.append(cv2.imdecode(np.frombuffer(d.read(), np.uint8), cv2.IMREAD_COLOR))
cv2.imwrite(f"{ep}/final_check.jpg", np.hstack([cv2.resize(f, (300, 300)) for f in fr]))
