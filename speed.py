"""Measure what one training step costs each method, compared with lewm. python3 speed.py TIER [--task push] [--seconds 7]

Steps run one at a time on one thread, interleaved across methods, so anything else slowing the machine slows every method alike. The ratio of medians goes to speed.json, which --budget compute reads. frozen spends 3 rounds in 10 on its pretraining step, as it does in training, and takes the mean; the rest take the median of the last half. Search time is not in this ratio, because every method searches the same way; beat.py checks it from cell_s.
"""
import argparse, json, os, time
import numpy as np, torch
import env, model as Mo, world as W

p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("--task", default="push"); p.add_argument("--seconds", type=float, default=7); a = p.parse_args(); os.nice(19); torch.set_num_threads(1); c = env.TIERS[a.tier]; torch.manual_seed(0); ms = list(Mo.METHODS); mods = {m: Mo.build(m, a.task, "pix", c) for m in ms}; aux = {m: Mo.Aux(m, a.task, "pix", c) for m in ms}; opt = {m: torch.optim.Adam(list(mods[m].parameters()) + list(aux[m].parameters()), 1e-4) for m in ms}
X, U = W.collect(a.task, c["batch"], c["sub"] - 1, 0); im = torch.from_numpy(np.stack([W.render(a.task, x) for x in X.astype(np.float64).transpose(1, 0, 2)], 1)); U = torch.from_numpy(U); T = {m: [] for m in ms}; t0 = time.time(); r = 0
while time.time() - t0 < a.seconds or r < 3:
    for m in np.random.default_rng(r).permutation(ms):
        t = time.perf_counter(); l, _ = Mo.loss(mods[m], aux[m], mods[m].obs(im), U, 0 if m == "frozen" and r % 10 < 3 else c["steps"], c["steps"]); opt[m].zero_grad(); l.backward(); opt[m].step(); T[m].append(time.perf_counter() - t)
    r += 1
med = {m: float(np.mean(T[m]) if m == "frozen" else np.median(T[m][len(T[m]) // 2:])) for m in ms}; sp = {m: med[m] / med[Mo.REF] for m in ms}; path = os.path.join(env.HERE, "speed.json"); S = json.load(open(path)) if os.path.exists(path) else {}; S[a.tier] = {m: round(v, 3) for m, v in sp.items()}; S[a.tier]["s_per_step_ref"] = round(med[Mo.REF], 5); json.dump(S, open(path, "w"), indent=1)
if r < 20: print(f"only {r} rounds: treat these ratios as a debug check, and rerun with more --seconds on a quiet machine before any --budget compute run")
print(f"{a.tier}: {r} rounds, {Mo.REF} {med[Mo.REF] * 1000:.0f} ms a step, so training one model takes {med[Mo.REF] * c['steps']:.0f} s")
for m in sorted(ms, key=sp.get): print(f"  {m:8s} {sp[m]:.2f}x  steps at equal compute {int(round(c['steps'] / sp[m]))}")
