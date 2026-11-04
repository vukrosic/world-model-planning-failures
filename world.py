"""Five small control tasks, batched in numpy, with scripted experts and pixel renders. No physics engine, so thousands of rollouts cost little.

reach  two-link arm reaches a pose (Reacher analog). Success is end-effector distance.
rooms  a dot crosses a door between two rooms (TwoRoom analog). Euclidean distance misleads here, because a point beside the wall can be far along the path.
push   a dot pushes a disk (Push-T analog, without the T shape).
small  push with a one-pixel disk and three still distractors of the same size (the small-object case in SLIM).
wind   push where the disk also drifts around the table's centre by itself (the world-effect case in DWM).

An action is 2 numbers held for SKIP environment steps, so a plan of H blocks has H x 2 numbers. LeWM packs SKIP different actions into a block, so its search space is larger than this one.
Channels: 0 the controlled body, 1 the object to move, 2 walls and distractors. Masking the channel in BLIND is the planted encoder fault.
"""
import numpy as np

TASKS = ("reach", "rooms", "push", "small", "wind"); SPEED = 0.04; SKIP = 5; H = 5; IMG = 32; OFFSETS = (25, 50, 75)
RA, RB, RS = 0.05, 0.07, 0.03; WALL, DOOR = 0.03, 0.14; LINK = 0.22; DRIFT = 0.004
SHAPE = 0.1  # the pull towards the point behind the disk in the true cost; --shape changes it, to test how much the cost slot depends on this choice
THR = {"reach": 0.05, "rooms": 0.06, "push": 0.06, "small": 0.05, "wind": 0.06}; SDIM = {"reach": 2, "rooms": 2, "push": 4, "small": 10, "wind": 4}; BLIND = {"reach": 0, "rooms": 0, "push": 1, "small": 1, "wind": 1}
RBS = {"push": RB, "small": RS, "wind": RB}
_X, _Y = np.meshgrid((np.arange(IMG) + 0.5) / IMG, (np.arange(IMG) + 0.5) / IMG)


def _clip01(x, r): return np.clip(x, r, 1 - r)
def _unit(v): return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
def ee(s): return np.stack([0.5 + LINK * (np.cos(s[:, 0]) + np.cos(s[:, 0] + s[:, 1])), 0.5 + LINK * (np.sin(s[:, 0]) + np.sin(s[:, 0] + s[:, 1]))], 1)


def step(task, s, a):
    """One environment step for a batch. a is clipped to [-1, 1]."""
    a = np.clip(a, -1, 1); s = s.copy()
    if task == "reach": s[:, :2] = s[:, :2] + 2 * SPEED * a; return s
    if task == "rooms":
        x, y = s[:, 0], s[:, 1]; nx, ny = _clip01(x + SPEED * a[:, 0], RA), _clip01(y + SPEED * a[:, 1], RA); band = np.abs(x - 0.5) < WALL + RA
        ny = np.where(band, np.clip(ny, 0.5 - DOOR + RA, 0.5 + DOOR - RA), ny); door = np.abs(ny - 0.5) <= DOOR - RA
        nx = np.where(door, nx, np.where(x < 0.5, np.minimum(nx, 0.5 - WALL - RA), np.maximum(nx, 0.5 + WALL + RA))); s[:, 0], s[:, 1] = nx, ny; return s
    rb = RBS[task]; ag = _clip01(s[:, :2] + SPEED * a, RA); b = s[:, 2:4]; d = b - ag; n = np.linalg.norm(d, axis=1, keepdims=True); b = b + d / np.maximum(n, 1e-6) * np.maximum(RA + rb - n, 0)
    if task == "wind": c = _unit(b - 0.5); b = b + DRIFT * np.stack([-c[:, 1], c[:, 0]], 1)
    s[:, :2] = ag; s[:, 2:4] = _clip01(b, rb); return s


def block(task, s, a):
    for _ in range(SKIP): s = step(task, s, a)
    return s


def rollout(task, s, plan):
    """s (N, sdim), plan (N, H, 2) -> every block's state (N, H + 1, sdim)."""
    out = [s]
    for t in range(plan.shape[1]): out.append(block(task, out[-1], plan[:, t]))
    return np.stack(out, 1)


def init(task, rng, n):
    if task == "reach": return rng.uniform(-np.pi, np.pi, (n, 2))
    if task == "rooms":
        s = rng.uniform(RA, 1 - RA, (n, 2)); s[:, 0] = np.where(rng.random(n) < 0.5, rng.uniform(RA, 0.5 - WALL - RA, n), rng.uniform(0.5 + WALL + RA, 1 - RA, n)); return s
    rb = RBS[task]; ag = rng.uniform(RA, 1 - RA, (n, 2)); b = rng.uniform(rb + 0.12, 1 - rb - 0.12, (n, 2)); near = np.linalg.norm(ag - b, axis=1) < RA + rb + 0.02; ag[near, 0] = np.where(b[near, 0] < 0.5, b[near, 0] + 0.3, b[near, 0] - 0.3)
    return np.concatenate([ag, b] + ([rng.uniform(0.1, 0.9, (n, 6))] if task == "small" else []), 1)


def dist(task, s, g):
    """The task's own distance, which success is judged on."""
    if task == "reach": return np.linalg.norm(ee(s) - ee(g), axis=-1)
    if task == "rooms": return np.linalg.norm(s[:, :2] - g[:, :2], axis=-1)
    return np.linalg.norm(s[:, 2:4] - g[:, 2:4], axis=-1)


def true_cost(task, s, g):
    """The cost a perfect planner would use. rooms uses the path through the door; the push tasks add a small pull towards the point behind the disk, because disk distance alone is flat until contact. Both are choices, stated in the README."""
    if task == "rooms":
        door = np.array([0.5, 0.5]); same = (s[:, 0] < 0.5) == (g[:, 0] < 0.5); return np.where(same, np.linalg.norm(s[:, :2] - g[:, :2], axis=1), np.linalg.norm(s[:, :2] - door, axis=1) + np.linalg.norm(door - g[:, :2], axis=1))
    if task == "reach": return dist(task, s, g)
    rb = RBS[task]; u = _unit(s[:, 2:4] - g[:, 2:4]); return dist(task, s, g) + SHAPE * np.linalg.norm(s[:, :2] - (s[:, 2:4] + u * (RA + rb)), axis=1)


def expert(task, s, g):
    """A scripted action towards goal state g, held for one block."""
    k = SPEED * SKIP
    if task == "reach":
        d = (g[:, :2] - s[:, :2] + np.pi) % (2 * np.pi) - np.pi; return np.clip(d / (2 * k), -1, 1)
    if task == "rooms":
        door = np.array([0.5, 0.5]); cross = (s[:, 0] < 0.5) != (g[:, 0] < 0.5); t = np.where((cross & (np.linalg.norm(s - door, axis=1) > 0.05))[:, None], door, g); inband = np.abs(s[:, 0] - 0.5) < WALL + RA
        t = np.where((inband & ~cross)[:, None], np.stack([np.where(g[:, 0] < 0.5, 0.5 - WALL - RA - 0.03, 0.5 + WALL + RA + 0.03), np.full(len(s), 0.5)], 1), t); return np.clip((t - s) / k, -1, 1)
    rb = RBS[task]; ag, b, gb = s[:, :2], s[:, 2:4], g[:, 2:4]; u = _unit(b - gb); gap = np.linalg.norm(b - gb, axis=1); behind = b + u * (RA + rb + 0.005); rel = ag - b; ahead = (rel * u).sum(1) < 0.3 * (RA + rb); perp = np.stack([-u[:, 1], u[:, 0]], 1) * np.sign((rel * np.stack([-u[:, 1], u[:, 0]], 1)).sum(1) + 1e-6)[:, None]
    around = b + perp * (RA + rb + 0.05) + u * (0.6 * (RA + rb) + 0.01); t = np.where(ahead[:, None], around, behind); go = np.clip((t - ag) / k, -1, 1); push = -u * np.clip(gap / k + (0.2 if task == "wind" else 0.0), 0, 1)[:, None]; ready = np.linalg.norm(ag - behind, axis=1) < 0.04
    return np.where((ready & (gap > 0.01))[:, None], push, np.where(ready[:, None], 0 * go, go))


def expert_plan(task, s, g, h=H):
    """Plan (N, h, 2) from running the expert in the true task."""
    out = []
    for _ in range(h): a = expert(task, s, g); out.append(a); s = block(task, s, a)
    return np.stack(out, 1)


def target(task, s, rng):
    """A far state for the expert to head to while it makes goals and data."""
    t = init(task, rng, len(s)); t[:, 4:] = s[:, 4:]
    if task == "rooms": flip = rng.random(len(s)) < 0.8; t[flip, 0] = np.where(s[flip, 0] < 0.5, rng.uniform(0.5 + WALL + RA, 1 - RA, flip.sum()), rng.uniform(RA, 0.5 - WALL - RA, flip.sum()))
    return t


def episodes(task, n, seed, split):
    """Each split draws from its own stream, so tune, report and hidden episodes never share a start. Goals are states a noisy expert reached at the given offset, and episodes where the goal already holds are dropped, so no episode succeeds by doing nothing."""
    assert split in ("tune", "report", "final"); rng = np.random.default_rng([seed, {"tune": 11, "report": 22, "final": 33}[split], TASKS.index(task)]); E = {}
    for off in OFFSETS:
        S, G = [], []; have = 0
        for _ in range(12):
            m = max(4 * n, 64); s0 = init(task, rng, m); s = s0.copy(); t = target(task, s, rng)
            for _ in range(off // SKIP): s = block(task, s, expert(task, s, t) + rng.normal(0, 0.3, (m, 2)))
            keep = dist(task, s0, s) > 2 * THR[task]; S.append(s0[keep]); G.append(s[keep]); have += keep.sum()
            if have >= n: break
        S, G = np.concatenate(S)[:n], np.concatenate(G)[:n]; E[off] = (S, G)
    return E


def collect(task, n, T, seed):
    """Half noisy expert, half smooth random actions. Returns states (n, T + 1, sdim) and actions (n, T, 2)."""
    rng = np.random.default_rng([seed, 44, TASKS.index(task)]); s = init(task, rng, n); t = target(task, s, rng); noisy = rng.random(n) < 0.5; A = rng.uniform(-1, 1, (n, 2)); X, U = [s], []
    for i in range(T):
        A = np.clip(0.7 * A + 0.3 * rng.uniform(-1, 1, (n, 2)) * 2, -1, 1)
        if i % 8 == 0 and i: t = target(task, s, rng)
        a = np.where(noisy[:, None], expert(task, s, t) + rng.normal(0, 0.3, (n, 2)), A); a = np.clip(a, -1, 1); s = block(task, s, a); X.append(s); U.append(a)
    return np.stack(X, 1).astype(np.float32), np.stack(U, 1).astype(np.float32)


def _disk(c, y, x, r):
    d = np.sqrt((_X[None] - x[:, None, None]) ** 2 + (_Y[None] - y[:, None, None]) ** 2); c[...] = np.maximum(c, np.clip((r - d) * IMG + 0.5, 0, 1))


def render(task, s):
    """(N, sdim) -> (N, 3, 32, 32) floats in [0, 1]."""
    n = len(s); im = np.zeros((n, 3, IMG, IMG), np.float32)
    if task == "reach":
        e1 = np.stack([0.5 + LINK * np.cos(s[:, 0]), 0.5 + LINK * np.sin(s[:, 0])], 1); e2 = ee(s); im[:, 2] = np.clip((0.04 - np.sqrt((_X - 0.5) ** 2 + (_Y - 0.5) ** 2)) * IMG + 0.5, 0, 1)
        for f in (0.33, 0.66, 1.0): p = 0.5 + f * (e1 - 0.5); _disk(im[:, 0], p[:, 1], p[:, 0], 0.03)
        for f in (0.33, 0.66, 1.0): p = e1 + f * (e2 - e1); _disk(im[:, 0], p[:, 1], p[:, 0], 0.03)
        return im
    if task == "rooms":
        im[:, 2] = ((np.abs(_X - 0.5) < WALL) & (np.abs(_Y - 0.5) > DOOR)).astype(np.float32); _disk(im[:, 0], s[:, 1], s[:, 0], RA); return im
    _disk(im[:, 0], s[:, 1], s[:, 0], RA); _disk(im[:, 1], s[:, 3], s[:, 2], RBS[task])
    if task == "small":
        for i in (4, 6, 8): _disk(im[:, 2], s[:, i + 1], s[:, i], RS)
    return im
