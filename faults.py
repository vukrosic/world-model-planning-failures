"""Score the attribution and every published diagnostic against planted faults with known answers. python3 faults.py TIER [--task push]

Each row is lewm with one fault, at the lr of the healthy row. Credit gain is the slot's Shapley credit minus its credit with no fault. A diagnostic column shows its value with the fault minus its value with no fault, so a reader can see which diagnostic moves under which fault. Which way each should move is in diag.py. Weaker strengths show how small a fault the test can find.
"""
import argparse
from beat import load, tuned, FAULT_SLOT
from world import TASKS

p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("--task", nargs="*", default=list(TASKS)); a = p.parse_args(); rows = load(); F = load("results_tag.jsonl"); D = ("pred", "zero", "rank", "centroid", "ir", "sr")
print("| task | fault | strength | answer | named | credit gain E P C S | success | " + " | ".join(D) + " |\n|---|---|---|---|---|---|---|" + "---|" * len(D)); right = tot = 0
for t in a.task:
    h = tuned(rows, a.tier, t, "steps")[0].get("lewm")
    if not h: continue
    for r in F:
        if r["tier"] == a.tier and r["task"] == t and r.get("fault") and r["lr"] == h["lr"] and r["world"] == h["world"]:
            s, b = r["scores"], h["scores"]; g = {x: s[f"shap.{x}"] - b[f"shap.{x}"] for x in "EPCS"}; n = max(g, key=g.get); ans = FAULT_SLOT[r["fault"]]
            if r["strength"] == 1: right += n == ans; tot += 1
            print(f"| {t} | {r['fault']} | {r['strength']:g} | {ans} | {n} | " + " ".join(f"{g[x]:+.2f}" for x in "EPCS") + f" | {s['success.llll']:.2f} | " + " | ".join(f"{s.get('diag.' + d, float('nan')) - b.get('diag.' + d, float('nan')):+.2f}" for d in D) + " |")
print(f"named at full strength: {right} of {tot}")
