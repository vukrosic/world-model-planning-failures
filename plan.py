"""The search and the two published costs that change it.

cem follows LeWM: Gaussian samples around a mean that starts at 0 with variance 1, the top elites set the next mean and spread, and the final mean is the plan. The whole plan runs before the next search.
The oracle search keeps the best sample ever seen and starts with scripted expert plans in the population, so it asks: if a plan that reaches the goal exists, does the cost pick it?
NGC is 2610.25013's cost: a graph over training latents with edges between mutual 15-nearest neighbours, all-pairs shortest paths offline, and Eq. 2 online with 4 attachment nodes on each side.
"""
import numpy as np, torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

CHUNK = 4096


def cem(cost, M, H, n, iters, elite, gen, mean=None, extra=None, best=False):
    """cost takes A (M, n, H, 2) and returns (M, n). Returns the plan, the last population and its costs."""
    mu = torch.zeros(M, H, 2) if mean is None else mean; std = torch.ones(M, H, 2); bc, bp = torch.full((M,), float("inf")), torch.zeros(M, H, 2)
    for i in range(iters):
        A = (mu[:, None] + std[:, None] * torch.randn(M, n, H, 2, generator=gen)).clamp(-1, 1)
        if extra is not None and i == 0: A[:, :extra.shape[1]] = extra; A = A[:, torch.randperm(n, generator=gen)]  # shuffled, or a useless cost that ties every sample would pick the expert plan by index
        c = cost(A); idx = c.topk(elite, 1, largest=False).indices; E = A.gather(1, idx[..., None, None].expand(-1, -1, H, 2)); mu = E.mean(1); std = E.std(1) if elite > 1 else std
        cb, ib = c.min(1); take = cb < bc; bc = torch.where(take, cb, bc); bp[take] = A[take, ib[take]]
    return (bp if best else mu), A, c


class NGC:
    def __init__(s, nodes, k=15, attach=4):
        s.v = nodes; s.attach = attach; s.k = min(k, len(nodes) - 1); d = torch.cdist(nodes, nodes); d.fill_diagonal_(float("inf")); nn = d.topk(s.k, 1, largest=False).indices; n = len(nodes); a = torch.zeros(n, n, dtype=torch.bool); a[torch.arange(n)[:, None], nn] = True; m = (a & a.T).numpy()
        s.D = torch.from_numpy(shortest_path(csr_matrix(np.where(m, d.numpy(), 0)), directed=False)).float(); s.connected = float((s.D < float("inf")).float().mean()); s.D = torch.nan_to_num(s.D, posinf=1e4)

    def near(s, q):
        d = torch.cdist(q, s.v); t = d.topk(s.attach, 1, largest=False); return t.values, t.indices, d.topk(s.k, 1, largest=False).values[:, -1]

    def cost(s, q, g):
        """q (M, n, D), g (M, D) -> (M, n). Eq. 2 of the paper."""
        M, n = q.shape[:2]; out = []; gv, gi, gr = s.near(g)
        for i in range(M):
            qv, qi, qr = [], [], []
            for c in q[i].split(CHUNK): a, b, r = s.near(c); qv.append(a); qi.append(b); qr.append(r)
            qv, qi, qr = torch.cat(qv), torch.cat(qi), torch.cat(qr); path = s.D[qi[:, :, None], gi[i][None, None, :]] + qv[:, :, None] + gv[i][None, None, :]; far = path.flatten(1).min(1).values; l2 = (q[i] - g[i]).norm(dim=-1)
            out.append(torch.where(l2 <= torch.minimum(qr, gr[i].expand_as(qr)), l2, far))
        return torch.stack(out)
