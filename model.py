"""World models: an encoder from pixels (or true state) to a latent, and a predictor from latent and action to the next latent.

The shared shape is a small convolutional encoder and a residual MLP predictor with one frame of history, because every task here is Markov in one frame. LeWM uses ViT-Tiny and a ViT-S predictor with three frames of history; that is too slow at 32 pixels on small tiers, and the README says so.

Baselines, every one training the same test-time modules:
lewm     LeWM (2603.19312): prediction MSE + lambda SIGReg, end to end. SIGReg is the authors' code from lucas-maes/le-wm, with lambda 0.1, 17 knots, 1024 projections.
pldm     PLDM as listed in LeWM's Eq. 8: prediction + variance + covariance + time similarity + time variance + time covariance + inverse dynamics. Weights are not in the papers I read, so the terms start at 1 and the knob scales them all; the tuning rows pick it, which gives PLDM more search than its paper describes, not less.
frozen   DINO-WM's idea (a frozen encoder, predictor trained on top), adapted: DINOv2 has no meaning at 32 pixels, so the encoder is a plain autoencoder trained for the first 30% of steps and frozen.
lewmidm  SLIM (2610.03137): LeWM plus an inverse-dynamics head used on (z, z') and on (stop-grad z, predicted z'), weights 0.09, 1, 1, width 256, three layers. The head is dropped at test time and does not count against the cap.
Controls that must never win: broken reads a blank image; wide has more weights than the cap allows.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import world as W

BASELINES = ("lewm", "pldm", "frozen", "lewmidm"); CONTROLS = ("broken", "wide"); METHODS = BASELINES + CONTROLS; REF = "lewm"; LATENT = 32


class SIGReg(nn.Module):
    def __init__(s, knots=17, num_proj=1024):
        super().__init__(); s.num_proj = num_proj; t = torch.linspace(0, 3, knots); dt = 3 / (knots - 1); w = torch.full((knots,), 2 * dt); w[[0, -1]] = dt; win = torch.exp(-t.square() / 2.0)
        s.register_buffer("t", t); s.register_buffer("phi", win); s.register_buffer("weights", w * win)

    def forward(s, proj):  # (T, B, D)
        A = torch.randn(proj.size(-1), s.num_proj, device=proj.device); A = A.div_(A.norm(p=2, dim=0)); x = (proj @ A).unsqueeze(-1) * s.t
        err = (x.cos().mean(-3) - s.phi).square() + x.sin().mean(-3).square(); return ((err @ s.weights) * proj.size(-2)).mean()


def mlp(i, h, o, n=2): return nn.Sequential(nn.Linear(i, h), nn.GELU(), *sum([[nn.Linear(h, h), nn.GELU()] for _ in range(n - 1)], []), nn.Linear(h, o))


class WM(nn.Module):
    """kind is pix (renders go in) or state (true state goes in, which is what the oracle encoder slot means)."""

    def __init__(s, task, kind, w, pw, mask=None):
        super().__init__(); s.task, s.kind, s.mask = task, kind, mask; sd = W.SDIM[task]; d = LATENT
        if kind == "pix": s.enc = nn.Sequential(nn.Conv2d(3, w, 4, 2, 1), nn.GELU(), nn.Conv2d(w, 2 * w, 4, 2, 1), nn.GELU(), nn.Conv2d(2 * w, 2 * w, 4, 2, 1), nn.GELU(), nn.Flatten(), nn.Linear(2 * w * 16, d))
        else: s.enc = mlp(_feat(task), 4 * w, d)
        s.pred = mlp(d + 2, pw, d)
        for p in [s.pred[-1].weight]: nn.init.normal_(p, std=0.01)
        nn.init.zeros_(s.pred[-1].bias)

    def obs(s, x):
        """x is what the task gives: states for this model's kind. Returns a tensor ready for enc."""
        if s.kind == "pix": im = torch.from_numpy(W.render(s.task, x)) if isinstance(x, np.ndarray) else x; return _mask(im, s.mask, s.task)
        t = torch.as_tensor(x, dtype=torch.float32); return _state(s.task, t) if t.shape[-1] == W.SDIM[s.task] else t

    def encode(s, x): return s.enc(s.obs(x))
    def step(s, z, a): return z + s.pred(torch.cat([z, a], -1))
    def test_weights(s): return sum(p.numel() for p in s.enc.parameters()) + sum(p.numel() for p in s.pred.parameters())


def _mask(im, mask, task, st=1.0):
    if mask == "blank": return im * 0
    if mask == "blind": im = im.clone(); im[..., W.BLIND[task], :, :] *= 1 - st
    return im
def _feat(task): return 4 if task == "reach" else W.SDIM[task]
def _state(task, t):  # angles enter through sin and cos, so the state model is not hurt by the wrap at pi
    return torch.cat([t[..., :2].sin(), t[..., :2].cos()], -1) if task == "reach" else t


def cap_weights(c, task): return WM(task, "pix", c["w"], c["pw"]).test_weights()
def build(method, task, kind, c, mask=None):
    w = c["w"] * 2 if method == "wide" else c["w"]; pw = c["pw"] * 2 if method == "wide" else c["pw"]; return WM(task, kind, w, pw, "blank" if method == "broken" and kind == "pix" else mask)


class Aux(nn.Module):
    """Training-only parts: the inverse-dynamics head and the autoencoder decoder. Neither is used at test time."""

    def __init__(s, method, task, kind, c):
        super().__init__(); s.sig = SIGReg(); s.method = method
        if method == "lewmidm": s.idm = nn.Sequential(nn.Linear(2 * LATENT, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2))
        if method == "pldm": s.idm = nn.Linear(2 * LATENT, 2)
        if method == "frozen":
            n = 3 * W.IMG * W.IMG if kind == "pix" else _feat(task); s.dec = mlp(LATENT, 4 * c["w"], n, 1); s.kind = kind


def _vc(z, eps=1e-4):
    """VICReg's variance hinge and covariance, over the batch at each time step."""
    z = z - z.mean(0, keepdim=True); v = F.relu(1 - (z.var(0) + eps).sqrt()).mean(); C = z.T @ z / max(len(z) - 1, 1); return v, (C - torch.diag(C.diag())).square().sum() / z.shape[1]


def loss(m, aux, x, a, step_i, steps, knob=1.0):
    """x (B, T + 1, ...) obs tensors, a (B, T, 2). Returns the loss to minimise and the prediction MSE."""
    B, T = a.shape[:2]; me = aux.method
    if me == "frozen":
        stop = int(0.3 * steps)
        if step_i < stop:
            f = x.flatten(0, 1); z = m.enc(f); return F.mse_loss(aux.dec(z), f.flatten(1)), torch.zeros(())
        with torch.no_grad(): z = m.enc(x.flatten(0, 1)).view(B, T + 1, -1)
        p = m.step(z[:, :-1], a); e = F.mse_loss(p, z[:, 1:]); return e, e.detach()
    z = m.enc(x.flatten(0, 1)).view(B, T + 1, -1); p = m.step(z[:, :-1], a); e = F.mse_loss(p, z[:, 1:])
    if me in ("lewm", "broken", "wide"): return e + 0.1 * knob * aux.sig(z.transpose(0, 1)), e.detach()
    if me == "lewmidm":
        h1 = aux.idm(torch.cat([z[:, :-1], z[:, 1:]], -1)); h2 = aux.idm(torch.cat([z[:, :-1].detach(), p], -1)); return e + 0.09 * knob * aux.sig(z.transpose(0, 1)) + F.mse_loss(h1, a) + F.mse_loss(h2, a), e.detach()
    if me == "pldm":
        vc = [_vc(z[:, t]) for t in range(T + 1)]; var = sum(v for v, _ in vc) / len(vc); cov = sum(c for _, c in vc) / len(vc); sim = (z[:, 1:] - z[:, :-1]).square().mean(); tv = [_vc(z[i]) for i in range(B)]; tvar = sum(v for v, _ in tv) / B; tcov = sum(c for _, c in tv) / B
        idm = F.mse_loss(aux.idm(torch.cat([z[:, :-1], z[:, 1:]], -1)), a); return e + knob * (var + cov + sim + tvar + tcov + idm), e.detach()
    raise KeyError(me)
