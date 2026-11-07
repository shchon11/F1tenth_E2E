"""The redesigned reward (11-07, agreed with the user step by step): lap time, with dense help that cannot change it.

* Lap bonus: +1 each time the car crosses the finish line forwards (-1 backwards). Discounted by the trainer's gamma, an
  earlier lap is worth more: the return is the lap time objective.
* Shaping, exact (Ng et al. 1999): gamma Phi(s') - Phi(s), Phi = -(the shortest distance left to the finish line) / (its
  value one lap out). Over a lap it sums to the bonus it lends, so it cannot change the best driving -- "lend the points
  on the way, pay them back at the line" (the user: "그건 괜찮네"). The distance is through the free space between the
  walls (`GoalField`), not along any line (the user: "레이싱라인을 정의할 수가 없잖아"): the centreline's projection
  jumped near the island tips (out/goal_distance_iccas25.png).
* Contacts: the simulator's own response (the speed into the obstacle is lost, `Sim.ideal_rows` keeps it lost until the
  next plan) and a time penalty on every contact, (CONTACT_S + CONTACT_S_PER_MPS x the impact's normal speed) seconds,
  in laps of T_REF seconds. Every contact is bad, a graze only less bad (the user, 11-07). A car contact is charged only
  if a LiDAR beam hit a car within SEEN_S before it (the user: "시야에서 아예 안보이는 상태였으면 그건 주면 안되는겨").
* No stall price, no other shaping: standing still earns no laps."""
import hashlib
import math
import os
from typing import Dict, Optional

import numpy as np
import torch

T_REF = 7.5               # [s] a lap, to put seconds in laps
CONTACT_S = 0.5           # [s] every contact
CONTACT_S_PER_MPS = 1.0   # [s per m/s of impact normal speed]
SEEN_S = 1.0              # [s] a car contact counts if a beam hit a car this recently
CACHE = os.path.expanduser("~/.cache/f1sim/goal_field")


def _field_one(edt: np.ndarray, res: float, origin, start_xy, start_yaw) -> np.ndarray:
    """(H, W) shortest distance [m] to the finish line through the free cells, not crossing the line backwards; NaN in
    the walls. The line: across the track through `start_xy`, square to `start_yaw`."""
    from scipy import ndimage, sparse
    from scipy.sparse.csgraph import dijkstra
    H, W = edt.shape
    free = edt > res                                          # a cell clear of the wall
    ii, jj = np.mgrid[0:H, 0:W]
    X = origin[0] + jj * res; Y = origin[1] + ii * res        # cell centres as `Track.sample_edt` rounds them
    i0 = int(round((start_xy[1] - origin[1]) / res)); j0 = int(round((start_xy[0] - origin[0]) / res))
    lab, _ = ndimage.label(free)
    if lab[i0, j0] == 0:
        raise ValueError("the start point is not in the free space")
    free &= lab == lab[i0, j0]
    t = np.array([math.cos(start_yaw), math.sin(start_yaw)])
    A = (X - start_xy[0]) * t[0] + (Y - start_xy[1]) * t[1]
    band = free & (np.abs(A) < res)
    bl, _ = ndimage.label(band, structure=np.ones((3, 3)))
    barrier = bl == bl[i0, j0]
    src = free & ~barrier & ndimage.binary_dilation(barrier, iterations=2) & (A < 0)
    node = free & ~barrier
    idx = -np.ones((H, W), np.int64); idx[node] = np.arange(int(node.sum()))
    rows, cols, wts = [], [], []
    ia0, ja0 = np.nonzero(node)
    for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (2, -1), (1, -2)]:
        ib, jb = ia0 + di, ja0 + dj
        ok = (ib >= 0) & (ib < H) & (jb >= 0) & (jb < W)
        ia, ja, ib, jb = ia0[ok], ja0[ok], ib[ok], jb[ok]
        ok = node[ib, jb]
        if abs(di) + abs(dj) == 3:                            # a knight's move: the two cells it passes free too
            si, sj = int(np.sign(di)), int(np.sign(dj))
            ok &= node[np.clip(ia + si, 0, H - 1), np.clip(ja + sj, 0, W - 1)]
            ok &= node[np.clip(ia + (di if abs(di) == 1 else si), 0, H - 1), np.clip(ja + (dj if abs(dj) == 1 else sj), 0, W - 1)]
        elif abs(di) + abs(dj) == 2:
            ok &= node[ia + di, ja] & node[ia, ja + dj]
        ia, ja, ib, jb = ia[ok], ja[ok], ib[ok], jb[ok]
        w = res * math.hypot(di, dj)
        rows += [idx[ia, ja], idx[ib, jb]]; cols += [idx[ib, jb], idx[ia, ja]]; wts += [np.full(ia.size, w)] * 2
    n = int(node.sum())
    G = sparse.csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    d = dijkstra(G, directed=False, indices=idx[src], min_only=True)
    D = np.full((H, W), np.nan, np.float32); D[node] = d
    D[barrier] = 0.0                                          # on the line: just crossed / about to
    D[~np.isfinite(D)] = np.nan
    return D


class GoalField:
    """Every track's shortest distance to its finish line, laid out like `Track.edt` so `sample_edt` reads it."""

    def __init__(self, track, device):
        self.track = track
        T = int(track.t_res.numel())
        res = track.t_res.cpu().numpy(); org = track.t_origin.cpu().numpy()
        Hs = track.t_H.cpu().numpy().astype(int); Ws = track.t_W.cpu().numpy().astype(int); off = track.t_off.cpu().numpy()
        edt = track.edt.float().cpu().numpy()
        xy0, yaw0 = track.pose_at_s(torch.zeros(T, device=track.t_res.device), torch.arange(T, device=track.t_res.device))
        xy0 = xy0.cpu().numpy(); yaw0 = yaw0.cpu().numpy()
        flat = np.full(edt.shape, np.nan, np.float32)
        lap = np.zeros(T, np.float32)
        os.makedirs(CACHE, exist_ok=True)
        for k in range(T):
            g = edt[off[k]: off[k] + Hs[k] * Ws[k]].reshape(Hs[k], Ws[k])
            key = hashlib.sha1(g.tobytes() + np.array([res[k], *org[k], *xy0[k], yaw0[k]], np.float64).tobytes()).hexdigest()[:20]
            fn = os.path.join(CACHE, key + ".npy")
            if os.path.exists(fn):
                D = np.load(fn)
            else:
                D = _field_one(g, float(res[k]), org[k], xy0[k], float(yaw0[k]))
                np.save(fn, D)
            flat[off[k]: off[k] + Hs[k] * Ws[k]] = D.reshape(-1)
            lap[k] = float(np.nanmax(D))
        self.field = torch.from_numpy(flat).to(device)
        self.lap = torch.from_numpy(lap).to(device)            # (T,) [m] the distance one lap out

    def remaining(self, xy: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """(...,) [m] the distance left to the finish line; NaN in a wall / off the grid"""
        d, inside = self.track.sample_edt(xy, tid, field=self.field, with_inside=True)
        return torch.where(inside, d, torch.full_like(d, float("nan")))


class PlanReward:
    """Per learner row: call `begin(ids)` after resets, `control_step()` after every control step, and `decision()` once
    a decision's control steps are done; `decision` returns the reward (B_l,) and its parts."""

    def __init__(self, env, gamma: float, device):
        self.env, self.gamma = env, float(gamma)
        self.lid = env.learner_ids
        self.goal = GoalField(env.sim.track, device)
        Bl = int(self.lid.numel())
        self.phi = torch.zeros(Bl, device=device)              # Phi at the last decision
        self.d_prev = torch.zeros(Bl, device=device)
        self.cost = torch.zeros(Bl, device=device)             # [laps] contacts since the last decision
        self.seen_t = torch.full((Bl,), 1e9, device=device)    # [s] since a beam last hit a car
        self.n_contacts = torch.zeros(Bl, 3, device=device)
        self.begin(torch.arange(Bl, device=device))

    def _d(self) -> torch.Tensor:
        st = self.env.sim.state[self.lid]
        return self.goal.remaining(st[:, :2], self.env.sim.tid[self.lid])

    def begin(self, rows: torch.Tensor) -> None:
        """rows (indices into the learner rows) just (re)started: Phi from where they stand, nothing owed"""
        if rows.numel() == 0:
            return
        d = self._d()[rows]
        L = self.goal.lap[self.env.sim.tid[self.lid][rows]]
        d = torch.where(torch.isfinite(d), d, L)
        self.d_prev[rows] = d
        self.phi[rows] = -d / L
        self.cost[rows] = 0.0
        self.seen_t[rows] = 1e9

    def control_step(self, seen_car: Optional[torch.Tensor] = None) -> None:
        """after an `env.step`: charge the contacts it began; `seen_car` (B_l,) bool = a beam hit a car this step"""
        env = self.env
        dt = float(env.sim.control_dt)
        self.seen_t = self.seen_t + dt
        if seen_car is not None:
            self.seen_t = torch.where(seen_car, torch.zeros_like(self.seen_t), self.seen_t)
        r = env.last_result
        if r is None or r.contact_onsets is None:
            return
        on = r.contact_onsets[self.lid].float()                # (B_l, 3) wall, prop, car
        imp = r.contact_impacts[self.lid].float().clamp_min(0.0)
        charge = on.clone()
        charge[:, 2] = charge[:, 2] * (self.seen_t <= SEEN_S).float()
        self.cost = self.cost + ((CONTACT_S + CONTACT_S_PER_MPS * imp) * charge).sum(1) / T_REF
        self.n_contacts = self.n_contacts + on

    def decision(self, restarted: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """the reward of the decision just driven. `restarted` (B_l,) bool: rows reset during it -- no shaping across an
        episode boundary (their Phi starts again from where they stand), their contact cost still charged"""
        d = self._d()
        L = self.goal.lap[self.env.sim.tid[self.lid]]
        d = torch.where(torch.isfinite(d), d, self.d_prev)      # inside a wall for a moment: hold the last reading
        jump = d - self.d_prev
        lap = (jump > 0.5 * L).float() - (jump < -0.5 * L).float()   # crossed the line forwards / backwards
        phi = -d / L
        shaping = self.gamma * phi - self.phi
        if restarted is not None:
            shaping = torch.where(restarted, torch.zeros_like(shaping), shaping)
            lap = torch.where(restarted, torch.zeros_like(lap), lap)
        out = {"lap": lap, "shaping": shaping, "contact": -self.cost, "reward": lap + shaping - self.cost}
        self.phi, self.d_prev = phi, d
        self.cost = torch.zeros_like(self.cost)
        if restarted is not None and bool(restarted.any()):
            self.begin(torch.nonzero(restarted)[:, 0])
        return out
