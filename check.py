"""Fast checks. python3 check.py [--slow]. Every part runs in seconds at the lowest priority, with no real training. --slow also runs a tiny row twice through the real script for the repeat and resume checks.

A failed check stops with the reason.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np, torch
import world as W, model as Mo, plan as Pl, attrib as At, diag as Di, env

os.nice(19); torch.set_num_threads(1); p = argparse.ArgumentParser(); p.add_argument("--slow", action="store_true"); a = p.parse_args(); T0 = time.time()
def ok(c, msg):
    if not c: sys.exit("FAILED: " + msg)
def done(n): print(f"{n}: ok ({time.time() - T0:.1f} s)")

# Tasks: the wall blocks, the door lets through, the disk moves only on contact, wind moves it alone.
s = np.array([[0.40, 0.10]]); x = s
for _ in range(30): x = W.step("rooms", x, np.array([[1.0, 0.0]]))
ok(x[0, 0] < 0.5 - W.WALL, "the wall let the dot through")
x = np.array([[0.40, 0.50]])
for _ in range(30): x = W.step("rooms", x, np.array([[1.0, 0.0]]))
ok(x[0, 0] > 0.5 + W.WALL, "the door blocks the dot")
s = np.array([[0.1, 0.1, 0.7, 0.7]]); ok(np.allclose(W.step("push", s, np.array([[1.0, 0.0]]))[0, 2:], [0.7, 0.7]), "disk moved without contact")
s = np.array([[0.5, 0.5, 0.5 + W.RA + W.RB - 0.001, 0.5]]); ok(W.step("push", s, np.array([[1.0, 0.0]]))[0, 2] > s[0, 2], "contact did not move the disk")
s = np.array([[0.1, 0.1, 0.7, 0.5]]); ok(np.linalg.norm(W.step("wind", s, np.zeros((1, 2)))[0, 2:] - s[0, 2:]) > 0, "wind does nothing"); done("task rules")

# Episodes: splits never share a start, goals are not already met, the same call gives the same episodes, and the expert can reach them.
for t in W.TASKS:
    E = {x: W.episodes(t, 16, 0, x) for x in ("tune", "report", "final")}; ok(all(np.allclose(W.episodes(t, 16, 0, "report")[o][0], E["report"][o][0]) for o in W.OFFSETS), f"{t} episodes differ between calls")
    for o in W.OFFSETS:
        ok(len(E["report"][o][0]) == 16, f"{t} offset {o} found too few episodes"); ok((W.dist(t, *E["report"][o]) > 2 * W.THR[t]).all(), f"{t} goal met at the start"); A, B, C = [E[x][o][0].round(4) for x in ("tune", "report", "final")]; ok(not ({tuple(r) for r in A} & {tuple(r) for r in B} | {tuple(r) for r in B} & {tuple(r) for r in C}), f"{t} splits share a start")
        s, g = E["report"][o]; x = s.copy(); hit = np.zeros(16, bool)
        for _ in range(env.replans(o) * W.H): x = W.block(t, x, W.expert(t, x, g)); hit |= W.dist(t, x, g) < W.THR[t]
        ok(hit.mean() >= (0.5 if t == "wind" else 0.9), f"{t} expert reaches {hit.mean():.2f} of offset {o} goals, so the budget or the expert is wrong")
done("episodes and experts")

# Renders: the object is drawn, moving it changes the image, the fault removes it, and the state fits one frame.
for t in W.TASKS:
    s, g = W.episodes(t, 4, 0, "report")[25]; ok(W.render(t, s).shape == (4, 3, 32, 32), "render shape"); ok(np.abs(W.render(t, s) - W.render(t, g)).sum((1, 2, 3)).min() > 0, f"{t} goal and start render the same"); m = Mo.build("lewm", t, "pix", dict(w=2, pw=8), mask="blind"); ok(m.obs(s)[:, W.BLIND[t]].abs().sum() == 0, f"{t} blind leaves the object")
    b = Mo.build("broken", t, "pix", dict(w=2, pw=8)); ok(b.obs(s).abs().sum() == 0, "broken still reads the image")
ok(W.render("small", W.episodes("small", 1, 0, "report")[25][0])[0, 1].sum() < 12, "the small object is not small"); done("renders")

# SIGReg: a Gaussian scores low, a collapsed latent scores high.
torch.manual_seed(0); sg = Mo.SIGReg(); g = sg(torch.randn(4, 256, 32)).item(); c = sg(torch.zeros(4, 256, 32) + 0.5).item(); ok(g < c / 5, f"SIGReg {g:.2f} on a Gaussian, {c:.2f} on a collapsed latent"); done("SIGReg")

# Every test-time weight trains, the training-only heads are dropped, and weights are equal.
c = env.TIERS["7s"]; cap = Mo.cap_weights(c, "push")
for me in Mo.METHODS:
    for k in ("pix", "state"):
        m = Mo.build(me, "push", k, c); ax = Mo.Aux(me, "push", k, c); X, U = W.collect("push", 8, 3, 0); im = torch.from_numpy(np.stack([W.render("push", x) for x in X.astype(np.float64).transpose(1, 0, 2)], 1)) if k == "pix" else torch.from_numpy(X); x = m.obs(im); U = torch.from_numpy(U)
        if me == "frozen":
            l, _ = Mo.loss(m, ax, x, U, 0, 10); l.backward(); ok(all(p.grad is not None for p in m.enc.parameters()), "frozen pretraining skips the encoder"); m.zero_grad(set_to_none=True); l, _ = Mo.loss(m, ax, x, U, 9, 10); l.backward(); ok(all(p.grad is None for p in m.enc.parameters()), "frozen encoder gets a gradient after pretraining"); ok(all(p.grad is not None for p in m.pred.parameters()), "frozen predictor gets no gradient"); continue
        l, _ = Mo.loss(m, ax, x, U, 0, 10); l.backward()
        for n, p_ in m.named_parameters(): ok(p_.grad is not None and (me == "broken" and k == "pix" or p_.grad.abs().sum() > 0), f"{me} {k} {n} gets no gradient")
    if me != "wide": ok(Mo.build(me, "push", "pix", c).test_weights() == cap, f"{me} has other test weights")
ok(Mo.build("wide", "push", "pix", c).test_weights() > 1.01 * cap, "wide is inside the cap"); done("gradients and weights")

# Shapley: credits add up, and tables with known answers give them.
n = 5; rng = np.random.default_rng(0); r = {x: rng.random(n) for x in At.CELLS}; phi = At.shapley(r); ok(np.allclose(sum(phi.values()), r["oooo"] - r["llll"]), "credits do not add up")
def table(f): return {x: np.full(n, float(f({s for s, l in zip(At.SLOTS, x) if l == "o"}))) for x in At.CELLS}
phi = At.shapley(table(lambda on: "C" in on)); ok(phi["C"].mean() == 1 and sum(abs(phi[s].mean()) for s in "EPS") == 0, "one slot fix, wrong credit")
phi = At.shapley(table(lambda on: "C" in on and "S" in on)); ok(np.isclose(phi["C"].mean(), 0.5) and np.isclose(phi["S"].mean(), 0.5), "two slots needed together, wrong credit")
phi = At.shapley(table(lambda on: "E" in on or {"P", "C"} <= on)); ok(np.isclose(phi["E"].mean(), 2 / 3) and np.isclose(phi["P"].mean(), 1 / 6), "encoder or predictor and cost, wrong credit")
L = At.labels(table(lambda on: "C" in on)); ok(L["single.C"] == 1 and L["several"] == 0, "labels"); L = At.labels(table(lambda on: "C" in on and "S" in on)); ok(L["several"] == 1, "several"); L = At.labels(table(lambda on: False)); ok(L["none"] == 1, "none"); ok(At.interactions(table(lambda on: "C" in on and "S" in on))["CS"].mean() == 1, "interaction"); done("Shapley")

# Search: finds a minimum, a tie does not pick the planted plan by position, and a tiny budget is worse.
gen = torch.Generator().manual_seed(0); goal = torch.full((4, 5, 2), 0.3); f = lambda A: (A - goal[:, None]).square().sum((-1, -2)); P, _, _ = Pl.cem(f, 4, 5, 200, 6, 20, gen); ok((P - goal).abs().max() < 0.1, "CEM misses a quadratic"); Q, _, _ = Pl.cem(f, 4, 5, 3, 1, 1, gen); ok(f(Q[:, None]).mean() > f(P[:, None]).mean(), "tiny budget is as good")
marked = torch.ones(32, 1, 5, 2) * 0.77; hit = 0
for i in range(32): B, _, _ = Pl.cem(lambda A: torch.zeros(A.shape[:2]), 1, 5, 32, 1, 3, torch.Generator().manual_seed(i), extra=marked[i:i + 1], best=True); hit += (B - 0.77).abs().max() < 1e-4
ok(hit < 8, f"a cost that ties everything picked the planted plan {hit} of 32 times"); done("search")

# NGC: on a U-shaped set of latents, the graph sees that the two tips are far apart, and near points use plain distance.
t = torch.linspace(0, 3.14, 200); nodes = torch.stack([t.cos(), t.sin()], 1); nodes = nodes + 0.001 * torch.randn(200, 2, generator=gen); G = Pl.NGC(nodes); q = torch.tensor([[[1.0, 0.0], [-1.0, 0.05]]]); g0 = torch.tensor([[-1.0, 0.0]]); co = G.cost(q, g0); ok(co[0, 0] > 2.5, f"far along the curve costs {co[0, 0]:.2f}, under the path length"); ok(co[0, 1] < 0.1, "near points do not use plain distance"); ok(G.connected > 0.99, "graph falls apart"); done("graph cost")

# Probe and goal policy never change the model; the probe reads a healthy latent.
m = Mo.build("lewm", "rooms", "state", dict(env.TIERS["7s"], fit=100)); m.requires_grad_(False); w0 = [p.clone() for p in m.parameters()]; X, U = W.collect("rooms", 64, 8, 0); r2 = env.attach(m, "rooms", X, U, dict(env.TIERS["7s"], fit=150), 0); ok(all((x == y).all() for x, y in zip(w0, m.parameters())), "the probe changed the model"); ok(r2 > 0.5, f"probe R2 {r2:.2f} on a random encoder of two numbers"); done("probe")

# Diagnostics: the rank correlation and AUC behave on known inputs.
ok(np.isclose(Di.rank_corr(np.arange(8.0)[None], np.arange(8.0)[None])[0], 1), "rank"); ok(np.isclose(Di.rank_corr(np.arange(8.0)[None], -np.arange(8.0)[None])[0], -1), "rank"); ok(Di.auc(np.array([1.0, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1, "auc"); done("diagnostics")

# The planted faults on a perfect stand-in: with no training at all, the attribution must name each fault's slot, and everything oracle must succeed.
env.TIERS["chk"] = dict(env.BASE, steps=0, batch=2, nt=2, T=4, w=2, pw=8, eps=8, n=48, iters=3, elite=6, mult=4, fit=1, nodes=4)
import argparse as ap
def stand(task, f, st=1.0): return env.row(ap.Namespace(method="lewm", task=task, tier="chk", lr=1e-3, knob=1.0, budget="steps", world=0, fault=f, tag="chk", final=False, stand=True, max_seconds=0, strength=st, shape=0.1))["scores"]
h = stand("push", None); ok(h["success.oooo"] >= 0.9, f"oooo {h['success.oooo']:.2f} on the stand-in"); ok(abs(h["shap.E"]) <= 0.1 and abs(h["shap.P"]) <= max(h["noise.flip"], 0.1), f"a perfect encoder and predictor get credit: {h['shap.E']:.2f} {h['shap.P']:.2f}")
for f, slot in (("blind", "E"), ("noact", "P"), ("flip", "C"), ("tiny", "S")):
    s = stand("push", f); gain = {x: s[f"shap.{x}"] - h[f"shap.{x}"] for x in "EPCS"}; ok(max(gain, key=gain.get) == slot, f"fault {f} named {max(gain, key=gain.get)}, {gain}"); ok(s["success.llll"] <= h["success.llll"], f"fault {f} did not hurt")
done("planted faults on the stand-in")

if a.slow:
    cmd = [sys.executable, "env.py", "--method", "lewmidm", "--task", "rooms", "--tier", "7s", "--tag", "rep"]; run = lambda *x: subprocess.run(cmd + list(x), capture_output=True, text=True)
    path = "results_tag.jsonl"; run(); run(); rows = [json.loads(l) for l in open(path)]; A, B = rows[-2]["scores"], rows[-1]["scores"]; ok(A == B, f"two runs differ: {[k for k in A if A[k] != B[k]][:5]}"); done("repeat")
    x = run("--max-seconds", "0.5"); ok(x.returncode != 0, "--max-seconds did not stop the job"); t = 0
    while run("--max-seconds", "3").returncode != 0 and t < 5: t += 1
    C = [json.loads(l) for l in open(path)][-1]["scores"]; ok(C == A, f"a resumed run differs: {[k for k in A if A[k] != C[k]][:5]}"); done("resume")
    for k in ("results_tag.jsonl",): os.remove(k)
print("all checks passed")
