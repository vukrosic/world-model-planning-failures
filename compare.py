"""Print every score per method on one task and tier, tuned lr per method, with no combined number. python3 compare.py TIER TASK [--budget steps] [--base lewm]

A star means the episode-by-episode gap to the base method survives false-win correction (Benjamini-Hochberg at 5%) across the table. Only scores in judges.json, from audit.py, are shown once it exists.
"""
import argparse, json, math, os
import numpy as np
from beat import load, tuned, items
from model import BASELINES

p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("task"); p.add_argument("--budget", default="steps"); p.add_argument("--base", default="lewm"); a = p.parse_args(); best, tried = tuned(load(), a.tier, a.task, a.budget); order = [m for m in BASELINES if m in best] + sorted(set(best) - set(BASELINES)); I = {m: items(best[m]) for m in order}
keys = [k for k in best[order[0]]["scores"] if not k.startswith("tune") and all(k in best[m]["scores"] for m in order)]; jp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judges.json"); J = json.load(open(jp)) if os.path.exists(jp) else None; live = [k for k in keys if J is None or f"{a.task}/{k}" in J]; tests = []
for k in live:
    for m in order:
        if m != a.base and k in I[m].files and k in I[a.base].files and len(I[m][k]) > 1: d = I[m][k] - I[a.base][k]; tests.append((math.erfc(abs(d.mean()) / max(d.std() / len(d) ** 0.5, 1e-12) / 2 ** 0.5), m, k))
tests.sort(key=lambda t: t[0]); cut = max([i + 1 for i, t in enumerate(tests) if t[0] <= 0.05 * (i + 1) / len(tests)], default=0); star = {(m, k) for _, m, k in tests[:cut]}
print(f"{a.task}, tier {a.tier}, equal {a.budget}, stars against {a.base}: {cut} of {len(tests)}\n| score | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
for k in live: print(f"| {k} | " + " | ".join(f"{best[m]['scores'][k]:.3f}" + ("*" if (m, k) in star else "") for m in order) + " |")
for name, f in (("lr picked", lambda r: f"{r['lr']:g}"), ("settings tried", lambda r: str(len(tried[r["method"]]))), ("knob", lambda r: f"{r['knob']:g}"), ("steps", lambda r: str(r["steps"])), ("test weights", lambda r: str(r["test_weights"])), ("training-only weights", lambda r: str(r["aux_weights"])), ("top slot", lambda r: f"{r['top']} z {r['top_z']:+.1f}"), ("seconds a step", lambda r: f"{r['s_per_step']:.3f}"), ("search seconds llll", lambda r: f"{r['cell_s'].get('llll', 0):.1f}")): print(f"| {name} | " + " | ".join(f(best[m]) for m in order) + " |")
