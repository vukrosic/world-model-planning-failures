"""Published failure diagnostics, computed on the all-learned cell, so each can be scored against the same known answers.

pred    prediction error after the plan runs, in units of the latent's variance (the usual check).
zero    no sample in the last search population reaches the goal when run in the true task (IMWM, 2606.01626).
rank    rank correlation between the planner's cost and the true cost, over the last population (the question behind IMWM and NGC).
centroid  cosine between the predictor's centroid at the realized and the predicted latent (2609.33940's c_cen, with the whole predictor as one block, since this predictor has no layers to pick).
ir, sr  Invariance Radius and Separation Rate (ACPC, 2608.12939): how far rollouts move when pixels get noise, and how often different starts stay further apart than that. Only defined for pixel models. The noise is a stand-in for its visual perturbation.
"""
import numpy as np, torch


def rank_corr(a, b):
    """a, b (M, n) -> (M,)"""
    ra, rb = a.argsort(1).argsort(1).astype(np.float64), b.argsort(1).argsort(1).astype(np.float64); ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.maximum(np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)), 1e-9)


def centroid(m, z, a):
    with torch.enable_grad():
        i = torch.cat([z, a], -1).detach().requires_grad_(True); m.pred(i).sum().backward(); g = i.grad.detach()
    for p in m.pred.parameters(): p.grad = None
    return g


def centroid_cos(m, real, pred, a): return torch.cosine_similarity(centroid(m, real, a), centroid(m, pred, a), -1)


def acpc(m, task, s, plan, gen, sigma=0.1, reps=4):
    """Per episode. Rollouts from clean and noised pixels under the same actions."""
    import world as W
    im = torch.from_numpy(W.render(task, s)); M = len(s); roll = lambda z: [z := m.step(z, plan[:, t]) for t in range(plan.shape[1])][-1]; zc = roll(m.enc(m.obs(im))); r = 0
    for _ in range(reps): r = r + (roll(m.enc(m.obs((im + sigma * torch.randn(im.shape, generator=gen)).clamp(0, 1)))) - zc).norm(dim=-1) / reps
    d = torch.cdist(zc, zc); d.fill_diagonal_(-1.0); return r, ((d > r[:, None]).float().sum(1) / max(M - 1, 1))


def auc(score, bad):
    """Chance a failed episode scores higher than a good one."""
    bad = np.asarray(bad, bool); a, b = score[bad], score[~bad]
    if not len(a) or not len(b): return float("nan")
    return float(((a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum()) / (len(a) * len(b)))
