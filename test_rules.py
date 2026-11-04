"""Test the tests: feed made-up rows through the kill test and the rules, and confirm each verdict. No training, under a second. python3 test_rules.py

Every case names a way a test could pass or fail for the wrong reason.
"""
import itertools, sys, zlib
import numpy as np
import attrib as At, beat, kill
from world import TASKS

N = 90; fails = []; CELLS = At.CELLS + ["llnl", "lllp"]


def ok(name, cond, note=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {note}" if note else ""))
    if not cond: fails.append(name)


def row(rows, m="lewm", task="push", p=0.5, lr=0.001, knob=1.0, tier="5m", budget="steps", w=0, tune=None, cells=None, idle=0.0, rand=0.0, fault="", flip=0.05, state_tune=1.0, r2=0.9, conn=1.0, off=None, weights=1000):
    """cells gives each cell's success rate, default p. Success per episode is drawn with the same uniform numbers across cells, so gaps are paired."""
    r_ = np.random.default_rng(zlib.crc32(repr((m, task, lr, knob, tier, budget, w, fault)).encode())); u = r_.random(N); P = {c: p for c in CELLS}; P.update(cells or {}); it = {f"success.{c}": (u < P[c]).astype(np.float32) for c in CELLS}; it["offset"] = np.repeat([25, 50, 75], N // 3)
    for c, f in (off or {}).items(): it[f"success.{c}"] = np.where(it["offset"] == 75, (u < f).astype(np.float32), it[f"success.{c}"])
    it["success.idle"], it["success.random"] = (u < idle).astype(np.float32), (u < rand).astype(np.float32); v = {c: it[f"success.{c}"] for c in At.CELLS}; phi = At.shapley(v); top, z = At.top_z(phi); s = {f"success.{c}": float(it[f"success.{c}"].mean()) for c in CELLS}
    for o in (25, 50, 75):
        for c in CELLS: s[f"success.{c}.o{o}"] = float(it[f"success.{c}"][it["offset"] == o].mean())
    for k in "EPCS": s[f"shap.{k}"] = float(phi[k].mean()); it[f"shap.{k}"] = phi[k]
    for k, x in At.solo(v).items(): it[f"solo.{k}"] = x
    for k, x in At.interactions(v).items(): it[f"pair.{k}"] = x
    s.update({"tune.dist": 0.5 - p if tune is None else tune, "tune.dist.state": state_tune, "noise.flip": flip, "noise.credit": At.noise_sd(flip, N), "probe.r2": r2, "probe.r2.state": 0.9, "ngc.connected": conn, "success.idle": float(it["success.idle"].mean()), "success.random": float(it["success.random"].mean())})
    x = dict(method=m, task=task, tier=tier, lr=lr, knob=knob, world=w, budget=budget, fault=fault, strength=1, tag="", shape=beat.DEFAULT_SHAPE, state_lr=0.0, stand=False, top=top, top_z=z, scores=s, test_weights=weights, search=dict(n=1), cell_s={"llll": 1.0, "llnl": 1.0, "lllp": 1.0}, v=it); rows.append(x); return x


class Files(dict):
    @property
    def files(s): return list(s)


beat.items = lambda r, hidden=False: Files(r["v"])


def status(rows, name, faults=()): return kill.report(rows, list(faults), say=lambda s: None)[name]
HEALTHY = dict(p=0.4, cells={"llll": 0.4, "oooo": 1.0})


# Holding still is a floor: a task where it succeeds as often as the model has nothing to attribute.
rows = []; row(rows, task="wind", p=0.4, cells=dict(oooo=1.0), idle=0.4, rand=0.3); row(rows, task="push", p=0.4, cells=dict(oooo=1.0))
ok("a model no better than holding still doesn't count in K1", status(rows, "K1 everything learned sits between .1 and .9 and above the no-model floor on at least 2 tasks") == "FAILS")
rows = []; row(rows, task="wind", p=0.4, cells=dict(oooo=1.0), rand=0.05); row(rows, task="push", p=0.4, cells=dict(oooo=1.0))
ok("a model above both floors counts in K1", status(rows, "K1 everything learned sits between .1 and .9 and above the no-model floor on at least 2 tasks") == "passes")
# K2 is in success units.
rows = []; row(rows, p=0.5, cells=dict(oooo=0.52), flip=0.9)
ok("a credit inside planner noise fails K2", status(rows, "K2 top credit is above planner noise") == "FAILS"); rows = []; row(rows, p=0.3, cells=dict(ollo=0.8, olll=0.8, oool=0.9, oooo=0.9), flip=0.3)
ok("a credit over twice its noise passes K2", status(rows, "K2 top credit is above planner noise") == "passes")
# K4: the published direction, not either gain.
rows = []; row(rows, task="rooms", p=0.2, off=dict(llnl=0.9, loll=0.2)); ok("graph cost gaining and true predictor not passes K4", status(rows, "K4 the published disagreement shows on rooms at the far offset") == "passes")
rows = []; row(rows, task="rooms", p=0.2, off=dict(llnl=0.2, loll=0.9)); ok("true predictor gaining alone fails K4", status(rows, "K4 the published disagreement shows on rooms at the far offset") == "FAILS")
rows = []; row(rows, task="rooms", p=0.2, off=dict(llnl=0.9, loll=0.9)); ok("both gaining equally fails K4", status(rows, "K4 the published disagreement shows on rooms at the far offset") == "FAILS")
rows = []; row(rows, task="rooms", p=0.2, off=dict(llnl=0.9, loll=0.2), conn=0.3); ok("a broken graph leaves K4 not judged", status(rows, "K4 the published disagreement shows on rooms at the far offset") == "not judged")
# K5 and K7 need fresh worlds and the floor.
rows = []; [row(rows, task="small", p=0.2, w=w) for w in range(3)]; row(rows, "lewmidm", "small", p=0.5); ok("a gain with no fresh worlds is not judged", status(rows, "K5 inverse dynamics, a published fix, moves the small object") == "not judged")
rows = []; [row(rows, task="small", p=0.2 + 0.1 * (w == 2), w=w) for w in range(3)]; [row(rows, "lewmidm", "small", p=0.5 if w == 0 else 0.25, w=w) for w in range(3)]; ok("a world-0 gain that fresh worlds don't repeat fails K5", status(rows, "K5 inverse dynamics, a published fix, moves the small object") == "FAILS")
rows = []; [row(rows, task="small", p=0.2, w=w) for w in range(3)]; [row(rows, "lewmidm", "small", p=0.5, w=w) for w in range(3)]; ok("a repeated gain passes K5", status(rows, "K5 inverse dynamics, a published fix, moves the small object") == "passes")
# K6 reads every task it ran on.
rows = []; row(rows, task="push", p=0.4); row(rows, "broken", "push", p=0.05, rand=0.05); row(rows, task="wind", p=0.4); row(rows, "broken", "wind", p=0.35, rand=0.05); ok("broken earning on wind fails K6 even if push is fine", status(rows, "K6 the broken control earns nothing") == "FAILS")
rows = []; row(rows, task="push", p=0.4); row(rows, "broken", "push", p=0.05, rand=0.05); ok("broken matching random plans passes K6", status(rows, "K6 the broken control earns nothing") == "passes")


# Rule 2.
def called(top_cells, worlds=3, faults=True, **kw):
    rows, F = [], []
    for w in range(worlds): r = row(rows, task="push", p=0.3, w=w, cells=top_cells, tune=0.5, **kw)
    r = rows[0]
    if faults:
        for f, s in beat.FAULT_SLOT.items(): q = row(F, task="push", p=0.3, fault=f, cells=top_cells, tune=0.5); q["scores"].update({f"shap.{x}": r["scores"][f"shap.{x}"] + (0.3 if x == s else 0) for x in "EPCS"})
    return beat.bottleneck(rows, F, "push", "5m")
CC = {c: (0.9 if c[2] == "o" else 0.3) for c in At.CELLS}; got, why = called(CC); ok("a slot with a clear solo gain is called", got == "C", "; ".join(why)[:200])
JJ = {c: (0.9 if c[2] == "o" and c[3] == "o" else 0.3) for c in At.CELLS}; got, why = called(JJ); ok("two slots needed together are called as a pair", got in ("C+S", "S+C"), f"{got} {'; '.join(why)[:200]}")
TIE = {c: (0.9 if c[2] == "o" or c[3] == "o" else 0.3) for c in At.CELLS}; got, why = called(TIE); ok("two slots that each fix it alone and tie are not called", got is None, str(got))
got, why = called(CC, worlds=1); ok("one world can't call a bottleneck", got is None and any("worlds" in w for w in why)); got, why = called(CC, faults=False); ok("no fault rows can't call a bottleneck", got is None and any("faults" in w for w in why))
got, why = called(CC, idle=0.3); ok("no gain over holding still can't call a bottleneck", got is None and any("idle" in w for w in why))
got, why = called(CC, r2=0.2); ok("a weak probe can't call the cost", got is None and any("probe" in w for w in why))
rows, F = [], []; [row(rows, task="push", p=0.3, w=w, cells=CC, tune=0.5, lr=0.001) for w in range(3)]; row(rows, task="push", p=0.1, tune=9.0, lr=0.003, state_tune=0.5); got, why = beat.bottleneck(rows, F, "push", "5m"); ok("an under-tuned state model can't call a bottleneck", got is None and any("under-tuned" in w for w in why))
# Fault rows must match the healthy row.
rows, F = [], []; h = row(rows, task="push", p=0.3, cells=CC, tune=0.5); q = row(F, task="push", p=0.3, fault="blind", knob=3.0); ok("a fault row at another knob isn't used", not beat.named_faults(h, F, "5m", "push")); q2 = row(F, task="push", p=0.3, fault="blind"); q2["stand"] = True; ok("a stand-in fault row isn't used", not beat.named_faults(h, F, "5m", "push"))


# Rule 1.
def field(mp=0.8, edge=False, fresh=True):
    rows = []; G = [(l, k) for l in (0.0003, 0.001, 0.003) for k in (0.3, 1.0, 3.0)]
    for task, b, (lr, kn) in itertools.product(TASKS, ("steps", "compute"), G):
        for m in ("new",) + tuple(beat.BASELINES):
            pick = (lr, kn) == ((0.003, 1.0) if edge and m != "new" else (0.001, 1.0))
            for w in range(3 if pick and (fresh or m != "new") else 1): row(rows, m, task, p=(mp if m == "new" else 0.3) if pick else 0.1, lr=lr, knob=kn, budget=b, w=w, tune=0.0 if pick else 9.0)
    return rows


rows = field(); F0 = [r for r in rows if not r["v"].get("x")]
win, why = beat.judge(rows, "new", "llll", "5m"); ok("a clear winner wins", win, "; ".join(why)[:300])
win, why = beat.judge(field(edge=True), "new", "llll", "5m"); ok("a baseline on the edge of its grid blocks a win", not win and any("edge" in w for w in why))
win, why = beat.judge(field(fresh=False), "new", "llll", "5m"); ok("no fresh worlds of the design blocks a win", not win and any("fresh worlds" in w for w in why))
print(f"{len(fails)} failed"); sys.exit(1 if fails else 0)
