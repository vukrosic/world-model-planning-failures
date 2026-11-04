"""Which scores can judge, using published methods only, so designs can't shape the verdict. python3 audit.py TIER CHECK_TIER

moved: share of methods whose score changed by 2 standard errors between an untrained row (--budget untrained) and training. spread: best minus worst method in standard errors of the episode-by-episode gap. lr rho: rank agreement across methods between the two lrs most methods tried. tier rho: rank agreement with the check tier. world rho: across fresh worlds. hidden rho: with the hidden episodes.
Cut-offs, set before any table was read: moved >= 1/2, spread >= 3, lr rho >= .4, tier rho >= .3, world rho >= .4, hidden rho >= .4. With four baselines a rank agreement is coarse, which the README states. Missing checks do not count against a score and print as nan. Scores that pass both ways go in judges.json, which compare.py reads.
"""
import argparse, collections, json, os
import numpy as np
from beat import load, tuned, items
from model import BASELINES
from world import TASKS

HERE = os.path.dirname(os.path.abspath(__file__)); p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("check"); a = p.parse_args(); rows = [r for r in load() if r["method"] in BASELINES and not r["tag"]]; final = load("final_results.jsonl")
def rho(x, y):
    if len(x) < 4: return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else 0.0


def table(T, T2):
    good = []; print(f"tier {T}, check tier {T2}\n| score | moved | spread | lr rho | tier rho | world rho | hidden rho | judges |\n|---|---|---|---|---|---|---|---|")
    for task in TASKS:
        B, _ = tuned(rows, T, task, "steps"); B2, _ = tuned(rows, T2, task, "steps"); W1, _ = tuned(rows, T, task, "steps", world=1); U, _ = tuned(rows, T, task, "untrained"); here = [r for r in rows if r["tier"] == T and r["task"] == task and r["world"] == 0 and r["budget"] == "steps"]; lrs = [l for l, _ in collections.Counter(r["lr"] for r in here).most_common(2)]
        if not B: continue
        at = lambda lr: {r["method"]: r["scores"] for r in here if r["lr"] == lr}; A, C = (at(lrs[0]), at(lrs[1])) if len(lrs) == 2 else ({}, {})
        for k in [k for k in next(iter(B.values()))["scores"] if not k.startswith("tune")]:
            ms = [m for m in B if k in B[m]["scores"]]; v = {m: B[m]["scores"][k] for m in ms}
            if len(ms) < 2: continue
            hi, lo = max(v, key=v.get), min(v, key=v.get); I = {m: items(B[m]) for m in ms}; f = k in I[hi].files and k in I[lo].files; d = I[hi][k] - I[lo][k] if f else np.zeros(1); spread = abs(d.mean()) / max(d.std() / len(d) ** 0.5, 1e-9) if len(d) > 1 else 0.0
            mv = []
            for m in ms:
                if m in U and k in items(U[m]).files and f: e = I[m][k] - items(U[m])[k]; mv.append(abs(e.mean()) / max(e.std() / len(e) ** 0.5, 1e-9) >= 2)
            moved = float(np.mean(mv)) if mv else float("nan"); same = [m for m in A if m in C and k in A[m] and k in C[m]]; lr = rho([A[m][k] for m in same], [C[m][k] for m in same]); both = [m for m in ms if m in B2 and k in B2[m]["scores"]]; tr = rho([v[m] for m in both], [B2[m]["scores"][k] for m in both]); wo = [m for m in ms if m in W1 and k in W1[m]["scores"]]; wr = rho([v[m] for m in wo], [W1[m]["scores"][k] for m in wo])
            F = {r["method"]: r["scores"] for r in final if r["tier"] == T and r["task"] == task and r["method"] in B and r["lr"] == B[r["method"]]["lr"] and k in r["scores"]}; hid = rho([v[m] for m in F], [F[m][k] for m in F]); ok = not moved < 0.5 and spread >= 3 and not lr < 0.4 and not tr < 0.3 and not wr < 0.4 and not hid < 0.4; good += [f"{task}/{k}"] * bool(ok)
            print(f"| {task}/{k} | {moved:.2f} | {spread:.1f} | {lr:.2f} | {tr:.2f} | {wr:.2f} | {hid:.2f} | {'yes' if ok else ''} |")
    print(f"| judges: {len(good)} |\n"); return good


x, y = table(a.tier, a.check), table(a.check, a.tier); both = [k for k in x if k in y]; json.dump(both, open(os.path.join(HERE, "judges.json"), "w"), indent=0); print(f"pass both: {len(both)}\n" + "\n".join(both))
