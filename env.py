"""The one test. python3 env.py --method lewm --task push --tier 7s --lr 0.001 [--knob 1] [--budget steps] [--world 0] [--fault blind] [--tag x] [--final] [--stand] [--max-seconds N]

One row trains a pixel model and a state model on one task, fits a probe and a goal policy on the frozen latents, and runs every cell on the same episodes. Every cell uses the same planner noise, so the comparison is paired.
Scores per cell are success, success at twice the threshold, and final distance. Nothing is combined. Items go to runs/*.npz.
Faults are the known answers: blind masks the object channel in training and planning (encoder), noact zeroes the action going into the predictor (predictor), flip gives the latent cost another episode's goal (cost), tiny cuts the search to 3 samples and 1 step (search).
--strength s grades a fault: the object is dimmed to 1 - s, the action is scaled by 1 - s, the goal is mixed s of the way to another episode's, and the search keeps 1 - s of its samples and steps. 1 is the full fault.
--shape changes the true cost's pull towards the point behind the disk. --stand replaces training with a perfect stand-in (latent is the true state, predictor is the true task), so the planner, cells and attribution can be tested in seconds.
--final uses hidden episodes and writes final_results.jsonl. Nothing should do this before a design is chosen.
"""
import argparse, json, os, pickle, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import world as W, model as Mo, plan as Pl, attrib as At, diag as Di

HERE = os.path.dirname(os.path.abspath(__file__)); FAULTS = ("blind", "noact", "flip", "tiny")
BASE = dict(sub=4, T=20)
TIERS = {
    "7s": dict(BASE, steps=25, batch=8, nt=24, T=8, w=4, pw=32, eps=2, n=12, iters=2, elite=3, mult=2, fit=20, nodes=40),
    "1m": dict(BASE, steps=300, batch=32, nt=200, w=12, pw=128, eps=12, n=48, iters=4, elite=6, mult=4, fit=300, nodes=400),
    "5m": dict(BASE, steps=1500, batch=64, nt=800, w=16, pw=192, eps=30, n=100, iters=6, elite=10, mult=8, fit=1000, nodes=1500),
    "1h": dict(BASE, steps=15000, batch=128, nt=4000, w=32, pw=512, eps=100, n=300, iters=10, iters_push=30, elite=30, mult=16, fit=3000, nodes=5000),  # 300 samples, 30 elites, 10 search steps and 30 on the push tasks are LeWM's values
}
FIX = ("llnl", "lllp"); REPEAT = "repeat"
replans = lambda off: int(np.ceil(2 * off / (W.H * W.SKIP))) + 1


def speed_ratio(method, tier):
    path = os.path.join(HERE, "speed.json"); S = json.load(open(path)) if os.path.exists(path) else {}; return S.get(tier, {}).get(method)


class Stand:
    """A perfect world model, for testing the test: latent is the true state, predictor is the true task. mask dims the object by strength."""

    def __init__(s, task, mask=None, strength=1.0): s.task, s.mask, s.kind, s.strength = task, mask, "stand", strength
    def _m(s, x):
        if s.mask != "blind": return x
        x = x.clone(); i = W.BLIND[s.task]; x[..., (0, 1) if i == 0 else (2, 3)] *= 1 - s.strength; return x
    def encode(s, x): return s._m(torch.as_tensor(np.asarray(x), dtype=torch.float32))
    def step(s, z, a): return s._m(torch.from_numpy(W.block(s.task, z.numpy().astype(np.float64), a.numpy().astype(np.float64)).astype(np.float32)))
    def decode(s, z): return z.numpy().astype(np.float64)
    def policy(s, z, zg): return torch.from_numpy(W.expert(s.task, z.numpy().astype(np.float64), zg.numpy().astype(np.float64)).astype(np.float32))


def enc(m, s):
    if m.kind == "stand": return m.encode(s)
    return torch.cat([m.encode(c.numpy() if m.kind == "pix" else c) for c in torch.from_numpy(np.asarray(s, dtype=np.float32)).split(Pl.CHUNK)])


def run_cell(task, E, name, c, models, ngc, fault, world, diags=None, st=1.0):
    """Runs one cell on episode dict E. Returns per-episode success, success at 2x threshold, final distance."""
    ce, cp, cc, cs = name; m = models[ce]; ok, ok2, dist = [], [], []; its = c["iters_push"] if "iters_push" in c and task in W.RBS else c["iters"]; n, iters, elite = (c["n"] * c["mult"], its, c["elite"] * c["mult"]) if cs == "o" else (c["n"], its, c["elite"])
    if cs == "l" and fault == "tiny": n, iters, elite = (3, 1, 1) if st >= 1 else (max(int(n * (1 - st)), 3), max(int(iters * (1 - st)), 1), max(int(elite * (1 - st)), 1))
    for off, (S, G) in E.items():
        M = len(S); s = S.copy(); zg0 = enc(m, G); zg = (1 - st) * zg0 + st * zg0.roll(1, 0) if fault == "flip" and cc == "l" else zg0; a1, a2 = np.zeros(M, bool), np.zeros(M, bool)
        for r in range(replans(off)):
            gen = torch.Generator().manual_seed(world * 100003 + off * 1009 + r + (777 if diags == REPEAT else 0)); z0 = enc(m, s)

            def cost(A):
                B, k = A.shape[:2]; f = A.flatten(0, 1); G2 = np.repeat(G, k, 0)
                if cp == "l":
                    z = z0.repeat_interleave(k, 0)
                    for t in range(W.H): z = m.step(z, f[:, t] * (1 - st if fault == "noact" else 1))
                    if cc == "o": return torch.from_numpy(W.true_cost(task, m.decode(z), G2)).float().view(B, k)
                else:
                    x = np.repeat(s, k, 0); pl = f.numpy().astype(np.float64)
                    for t in range(W.H): x = W.block(task, x, pl[:, t])
                    if cc == "o": return torch.from_numpy(W.true_cost(task, x, G2)).float().view(B, k)
                    z = enc(m, x)
                z = z.view(B, k, -1)
                return ngc[ce].cost(z, zg0) if cc == "n" else (z - zg[:, None]).square().sum(-1)
            extra = mean = None
            if cs == "o":
                e = torch.from_numpy(W.expert_plan(task, s, G).astype(np.float32)); j = max(n // 32, 2); extra = (e[:, None] + 0.1 * torch.randn(M, j, W.H, 2, generator=gen)).clamp(-1, 1); extra[:, 0] = e
            if cs == "p":
                z = z0; mean = []
                for _ in range(W.H): a = m.policy(z, zg0).clamp(-1, 1); mean.append(a); z = m.step(z, a)
                mean = torch.stack(mean, 1)
            plan, pop, c_pop = Pl.cem(cost, M, W.H, n, iters, elite, gen, mean=mean, extra=extra, best=cs == "o"); pl = plan.numpy().astype(np.float64); traj = W.rollout(task, s, pl)
            if isinstance(diags, dict): _diag(diags, off, task, m, s, G, pop, c_pop, plan, traj, gen, fault, st)
            s = traj[:, -1]; d = W.dist(task, s, G); a1 |= d < W.THR[task]; a2 |= d < 2 * W.THR[task]
        ok.append(a1); ok2.append(a2); dist.append(d)
    return np.concatenate(ok).astype(np.float32), np.concatenate(ok2).astype(np.float32), np.concatenate(dist).astype(np.float32)


def _diag(D, off, task, m, s, G, pop, c_pop, plan, traj, gen, fault, st):
    B, k = pop.shape[:2]; x = np.repeat(s, k, 0); f = pop.flatten(0, 1).numpy().astype(np.float64); G2 = np.repeat(G, k, 0)
    for t in range(W.H): x = W.block(task, x, f[:, t])
    tc = W.true_cost(task, x, G2).reshape(B, k); hit = (W.dist(task, x, G2) < W.THR[task]).reshape(B, k).mean(1); e = D.setdefault(off, {"zero": [], "rank": [], "pred": [], "centroid": [], "ir": [], "sr": []})
    e["zero"].append(hit == 0); e["rank"].append(Di.rank_corr(c_pop.numpy(), tc))
    z = enc(m, s); a0 = plan * (1 - st if fault == "noact" else 1)
    for t in range(W.H - 1): z = m.step(z, a0[:, t])
    real = enc(m, traj[:, -2]); e["pred"].append(((z - real).square().mean(-1) / real.var(0).mean().clamp_min(1e-6)).numpy())
    if m.kind != "stand":
        e["centroid"].append(Di.centroid_cos(m, real, z, a0[:, -1]).numpy())
        if m.kind == "pix": r, sr = Di.acpc(m, task, s, a0, gen); e["ir"].append(r.numpy()); e["sr"].append(sr.numpy())


def train(m, method, task, kind, X, U, frames, c, lr, knob, steps, world, ck, deadline):
    """Trains in place. Saves model, aux, optimizer and step at the deadline so the same command continues."""
    g = torch.Generator().manual_seed(world * 7 + (kind == "pix")); aux = Mo.Aux(method, task, kind, c); opt = torch.optim.Adam(list(m.parameters()) + list(aux.parameters()), lr); i0 = 0; e = torch.zeros(())
    if ck and os.path.exists(ck):
        d = torch.load(ck, weights_only=False); m.load_state_dict(d["m"]); aux.load_state_dict(d["a"]); opt.load_state_dict(d["o"]); i0 = d["i"]; g.set_state(d["g"]); torch.set_rng_state(d["r"])
    N, T = U.shape[:2]; X = torch.from_numpy(X); U = torch.from_numpy(U)
    for i in range(i0, steps):
        if deadline and time.time() > deadline and ck: torch.save(dict(m=m.state_dict(), a=aux.state_dict(), o=opt.state_dict(), i=i, g=g.get_state(), r=torch.get_rng_state()), ck); raise TimeoutError
        idx = torch.randint(0, N, (c["batch"],), generator=g); t0 = torch.randint(0, T - c["sub"] + 2, (c["batch"],), generator=g); ts = t0[:, None] + torch.arange(c["sub"]); a = U[idx[:, None], ts[:, :-1]]
        x = m.obs(frames[idx[:, None], ts].float() / 255) if kind == "pix" else m.obs(X[idx[:, None], ts]); l, e = Mo.loss(m, aux, x, a, i, steps, knob); opt.zero_grad(); l.backward(); opt.step()
    return float(e), sum(p.numel() for p in aux.parameters())


def attach(m, task, X, U, c, world):
    """Fits a probe from latent to state and a goal-conditioned policy on frozen latents, with no gradient into the model. Returns R2 of the probe on held-out frames."""
    g = torch.Generator().manual_seed(world + 5); n, T = U.shape[:2]; lat = enc(m, X.reshape(-1, X.shape[-1])).view(n, T + 1, -1).detach(); y = Mo._state(task, torch.from_numpy(X)); cut = max(int(0.8 * n), 1)
    probe = Mo.mlp(Mo.LATENT, 128, y.shape[-1]); po = torch.optim.Adam(probe.parameters(), 1e-3); pol = Mo.mlp(2 * Mo.LATENT, 128, 2); oo = torch.optim.Adam(pol.parameters(), 1e-3)
    for _ in range(c["fit"]):
        i = torch.randint(0, cut, (64,), generator=g); t = torch.randint(0, T, (64,), generator=g); j = (t + 1 + torch.randint(0, T, (64,), generator=g)).clamp_max(T)
        l = F.mse_loss(probe(lat[i, t]), y[i, t]); po.zero_grad(); l.backward(); po.step(); l = F.mse_loss(pol(torch.cat([lat[i, t], lat[i, j]], -1)), torch.from_numpy(U)[i, t]); oo.zero_grad(); l.backward(); oo.step()
    with torch.no_grad(): v = (y[cut:] if cut < n else y[:1]).flatten(0, 1); p = probe((lat[cut:] if cut < n else lat[:1]).flatten(0, 1)); r2 = float(1 - (v - p).square().sum() / (v - v.mean(0)).square().sum().clamp_min(1e-9))

    def decode(z):
        with torch.no_grad(): o = probe(z)
        return torch.atan2(o[..., :2], o[..., 2:4]).numpy().astype(np.float64) if task == "reach" else o.numpy().astype(np.float64)
    m.decode = decode; m.policy = lambda z, zg: pol(torch.cat([z, zg], -1)).detach(); return r2


def row(a):
    t00 = time.time(); c = TIERS[a.tier]; task = a.task; deadline = t00 + a.max_seconds if a.max_seconds else 0; torch.manual_seed(a.world); np.random.seed(a.world); name = f"{a.tier}_{task}_{a.method}_{a.budget}_lr{a.lr:g}_k{a.knob:g}_w{a.world}" + (f"_{a.fault}" if a.fault else "") + (f"_s{a.strength:g}" if a.strength != 1 else "") + (f"_{a.tag}" if a.tag else ""); runs = os.path.join(HERE, "runs"); os.makedirs(runs, exist_ok=True); part = os.path.join(runs, "part_" + name + ".pkl")
    steps = 0 if a.budget == "untrained" else c["steps"]; ratio = speed_ratio(a.method, a.tier)
    if a.budget == "compute":
        if not ratio: raise SystemExit("run speed.py first: --budget compute needs speed.json")
        steps = int(round(c["steps"] / ratio))
    t0 = time.time(); W.SHAPE = a.shape; mask = "blind" if a.fault == "blind" else None; nodes = {}; r2 = {}; mse = {}; auxw = 0; models = {}; ngc = {}
    if a.stand:
        models = {"l": Stand(task, mask, a.strength), "o": Stand(task)}; cap = tw = 0; ngc = {}
    else:
        X, U = W.collect(task, c["nt"], c["T"], a.world); frames = torch.from_numpy((np.stack([W.render(task, x) for x in X.astype(np.float64).transpose(1, 0, 2)], 1) * 255).astype(np.uint8)); cap = Mo.cap_weights(c, task)
        for e, kind in (("l", "pix"), ("o", "state")):
            m = Mo.build(a.method, task, kind, c, mask=mask if kind == "pix" else None); m.strength = a.strength; mse[e], auxw = train(m, a.method, task, kind, X, U, frames, c, a.lr, a.knob, steps, a.world, os.path.join(runs, f"ck_{name}_{kind}.pt") if a.max_seconds else None, deadline); m.requires_grad_(False)
            r2[e] = attach(m, task, X, U, c, a.world); idx = torch.randperm(X.shape[0] * X.shape[1])[:c["nodes"]]
            with torch.no_grad(): nodes = enc(m, X.reshape(-1, X.shape[-1])[idx.numpy()]); ngc[e] = Pl.NGC(nodes)
            models[e] = m
        tw = models["l"].test_weights()
    t_train = time.time() - t0; split = "final" if a.final else "report"; Etune, Erep = W.episodes(task, c["eps"], a.world, "tune"), W.episodes(task, c["eps"], a.world, split); done = pickle.load(open(part, "rb")) if a.max_seconds and os.path.exists(part) else {}; D = {}
    cells = At.CELLS + ([] if a.stand else list(FIX)); t0 = time.time(); cell_s = {}
    with torch.no_grad():
        for n in cells + [REPEAT, "tune"]:
            if n in done: continue
            if deadline and time.time() > deadline: pickle.dump(done, open(part, "wb")); raise TimeoutError
            if n[1:3] == "oo" and n[0] == "o" and "l" + n[1:] in done: done[n] = done["l" + n[1:]]; continue  # with the true predictor and true cost the encoder is unused, so the outcome is the same by construction
            tc = time.time(); done[n] = run_cell(task, Etune if n == "tune" else Erep, At.LL if n in (REPEAT, "tune") else n, c, models, ngc, a.fault, a.world, diags=REPEAT if n == REPEAT else D if n == At.LL else None, st=a.strength)
            cell_s[n] = round(time.time() - tc, 2)
            if n == At.LL: done["_diag"] = D
    D = done.pop("_diag"); t_plan = time.time() - t0; v = {n: done[n][0] for n in At.CELLS}; off = np.concatenate([[o] * len(Erep[o][0]) for o in Erep]); sc = {}; it = {"offset": off}
    for n in cells:
        s1, s2, d = done[n]; sc[f"success.{n}"], sc[f"success2x.{n}"], sc[f"dist.{n}"] = float(s1.mean()), float(s2.mean()), float(d.mean()); it[f"success.{n}"], it[f"success2x.{n}"], it[f"dist.{n}"] = s1, s2, d
        for o in Erep: sc[f"success.{n}.o{o}"] = float(s1[off == o].mean())
    fail = done[At.LL][0] < 0.5; sc["noise.flip"] = float((done[REPEAT][0] != done[At.LL][0]).mean()); sc["tune.dist"], sc["tune.success"] = float(done["tune"][2].mean()), float(done["tune"][0].mean())
    phi = At.shapley(v); top, zg = At.top_z(phi)
    for s in At.SLOTS: sc[f"shap.{s}"] = float(phi[s].mean()); it[f"shap.{s}"] = phi[s]
    sc.update({f"label.{k}": x for k, x in At.labels(v).items()}); sc.update({f"pair.{k}": float(x.mean()) for k, x in At.interactions(v).items()}); sc["shap.top_z"] = zg
    if not a.stand: sc.update({"pred.mse": mse["l"], "probe.r2": r2["l"], "probe.r2.state": r2["o"]})
    for k in ("zero", "rank", "pred", "centroid", "ir", "sr"):
        if D[next(iter(D))][k]:
            x = np.concatenate([np.mean(D[o][k], 0) if k != "zero" else np.all(D[o][k], 0) for o in Erep]).astype(np.float32); it[f"diag.{k}"] = x; sc[f"diag.{k}"] = float(x.mean()); sc[f"auc.{k}"] = Di.auc(x if k not in ("rank", "centroid", "sr") else -x, fail)
    if fail.any(): sc["diag.zero_when_failed"] = float(it["diag.zero"][fail].mean())
    if not a.stand: Xa = X.reshape(-1, X.shape[-1]); g = np.concatenate([Erep[o][1] for o in Erep]); w = 4 if task == "small" else g.shape[1]; it["cover"] = np.sqrt(((g[:, None, :w] - Xa[None, :, :w]) ** 2).sum(-1)).min(1).astype(np.float32); sc["auc.cover"] = Di.auc(it["cover"], fail)
    r = dict(method=a.method, task=task, tier=a.tier, lr=a.lr, knob=a.knob, world=a.world, budget=a.budget, fault=a.fault or "", strength=a.strength, tag=a.tag or "", shape=a.shape, final=bool(a.final), top=top, top_z=zg, steps=steps, test_weights=tw, aux_weights=auxw, cap=cap, n=len(off), H=W.H, search=dict(n=c["n"], iters=c["iters"], elite=c["elite"]), train_s=round(t_train, 1), plan_s=round(t_plan, 1), cell_s=cell_s, s_per_step=round(t_train / max(2 * steps, 1), 5), speed_ratio=ratio, scores={k: round(float(x), 5) for k, x in sc.items()})
    np.savez_compressed(os.path.join(runs, ("final_" if a.final else "") + name + ".npz"), **it); fn = "final_results.jsonl" if a.final else "results_tag.jsonl" if a.tag or a.fault or a.stand else "results.jsonl"
    with open(os.path.join(HERE, fn), "a") as f: f.write(json.dumps(r) + "\n")
    for k in [part] + [os.path.join(runs, f"ck_{name}_{x}.pt") for x in ("pix", "state")]:
        if os.path.exists(k): os.remove(k)
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--method", default="lewm"); p.add_argument("--task", default="push"); p.add_argument("--tier", default="7s"); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--knob", type=float, default=1.0); p.add_argument("--budget", default="steps", choices=("steps", "compute", "untrained")); p.add_argument("--world", type=int, default=0); p.add_argument("--fault", choices=FAULTS); p.add_argument("--strength", type=float, default=1.0); p.add_argument("--shape", type=float, default=0.1); p.add_argument("--tag"); p.add_argument("--final", action="store_true"); p.add_argument("--stand", action="store_true"); p.add_argument("--max-seconds", type=float, default=0); a = p.parse_args()
    os.nice(19); torch.set_num_threads(1)
    try: r = row(a)
    except TimeoutError: raise SystemExit(f"stopped at --max-seconds {a.max_seconds:g}: the same command continues")
    s = r["scores"]; print(f"{r['task']} {r['method']} {r['tier']} lr {r['lr']:g} fault {r['fault'] or '-'}: llll {s['success.llll']:.2f} oooo {s['success.oooo']:.2f} top {r['top']} z {r['top_z']:+.1f} flip {s['noise.flip']:.2f} train {r['train_s']}s plan {r['plan_s']}s weights {r['test_weights']}")
