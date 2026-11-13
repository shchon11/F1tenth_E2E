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
* The whole plan counts (the user agreed 11-07: the car drives only a plan's first 0.1 s, so without these the other
  nine points would touch nothing and come out anyhow again):
  - Phi is read at the PLAN's points: the mean distance left over its 10 points. The plan is part of the state (the
    policy sees it), so this is still exact shaping: every point is pushed to be as far along as the car can reach,
    and the lap objective is unchanged.
  - A plan that runs into a wall or a box within its second (any footprint corner of its poses inside one) costs
    PLAN_HIT_SHARE of the contact it would be, at the plan's speed there: drawn every 0.1 s, a second of such plans
    costs about one contact.
* No stall price, no other shaping: standing still earns no laps. (11-13, the user: "저번에 분명히 서 있는거에 보상 매겨서
  후진하도록 유도했는데" -- the 11-06 stall price had gone with this redesign; policies stood nose-on at a box for 20 s, even
  after it was taken away: `stall_s` puts it back, off by default.)"""
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
PLAN_HIT_SHARE = 0.1      # a plan that hits something: this share of the contact it plans
CACHE = os.path.expanduser("~/.cache/f1sim/goal_field")
CLEAR = 0.5               # [m] the distance field's route keeps this far from the walls (`_field_one`)


def _grid_graph(node: np.ndarray, res: float):
    """8 + 8 (knight) neighbour graph over the `node` cells: (csr matrix, (H, W) node index, -1 off the graph)"""
    from scipy import sparse
    H, W = node.shape
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
    return sparse.csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)), idx


def _field_one(edt: np.ndarray, res: float, origin, start_xy, start_yaw, clear: float = CLEAR) -> np.ndarray:
    """(H, W) shortest distance [m] to the finish line, not crossing the line backwards; NaN in the walls. The line:
    across the track through `start_xy`, square to `start_yaw`.
    The route keeps `clear` from every wall, so it follows the course: one cell of clearance let it cut through the
    gaps between an island and a partition wall (0.2 m on the original track, up to 0.7 m on its variants, 11-11), and
    the distance left then GREW along 5-40 % of a lap driven forwards. A free cell nearer a wall than `clear` reads the
    nearest (through the free space) cell that is not."""
    from scipy import ndimage
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
    wide = free & (edt > max(clear, res))
    lab, _ = ndimage.label(wide)
    if lab[i0, j0] == 0:
        raise ValueError(f"the start point is nearer a wall than {clear} m")
    wide &= lab == lab[i0, j0]
    t = np.array([math.cos(start_yaw), math.sin(start_yaw)])
    A = (X - start_xy[0]) * t[0] + (Y - start_xy[1]) * t[1]
    band = free & (np.abs(A) < res)
    bl, _ = ndimage.label(band, structure=np.ones((3, 3)))
    barrier = bl == bl[i0, j0]
    src = wide & ~barrier & ndimage.binary_dilation(barrier, iterations=2) & (A < 0)
    node = wide & ~barrier
    G, idx = _grid_graph(node, res)
    d = dijkstra(G, directed=False, indices=idx[src], min_only=True)
    D = np.full((H, W), np.nan, np.float32); D[node] = d
    D[~np.isfinite(D)] = np.nan
    # the free cells near the walls: the value of the nearest cell that has one, through the free space (not over the line)
    node2 = free & ~barrier
    known = node2 & np.isfinite(D)
    if bool((node2 & ~known).any()):
        G2, idx2 = _grid_graph(node2, res)
        edge = known & ndimage.binary_dilation(node2 & ~known, structure=np.ones((5, 5)))
        _, _, near = dijkstra(G2, directed=False, indices=idx2[edge], min_only=True, return_predecessors=True)
        flat_i = np.flatnonzero(node2.reshape(-1))            # graph node -> cell
        fill = node2 & ~known
        got = near[idx2[fill]]
        vals = np.full(got.shape, np.nan, np.float32)
        ok = got >= 0
        vals[ok] = D.reshape(-1)[flat_i[got[ok]]]
        D[fill] = vals
    D[barrier] = 0.0                                          # on the line: just crossed / about to
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
            key = hashlib.sha1(g.tobytes() + np.array([res[k], *org[k], *xy0[k], yaw0[k], CLEAR], np.float64).tobytes()).hexdigest()[:20]
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
    """Per learner row: `begin(rows)` after resets, `control_step()` after every control step, and once a decision's
    control steps are done `decision(pts)` with the plan just driven (its 10 points in the world); `plan_cost(...)` prices
    a plan when it is drawn. `decision` returns the reward (B_l,) and its parts."""

    def __init__(self, env, gamma: float, device):
        self.env, self.gamma = env, float(gamma)
        self.lid = env.learner_ids
        self.goal = GoalField(env.sim.track, device)
        Bl = int(self.lid.numel())
        self.phi = torch.zeros(Bl, device=device)              # Phi of the state at the last decision
        self.d_prev = torch.zeros(Bl, device=device)           # the CAR's distance left there (laps are the car's)
        self.cost = torch.zeros(Bl, device=device)             # [laps] contacts since the last decision
        self.seen_t = torch.full((Bl,), 1e9, device=device)    # [s] since a beam last hit a car
        self.n_contacts = torch.zeros(Bl, 3, device=device)
        self.corners = env.sim.corners.float()                 # (4, 2) footprint about the CoG
        # the plan-hit cost's share of a contact and how far into the plan it looks (poses on the control grid; None = all)
        self.plan_hit_share, self.plan_hit_n = PLAN_HIT_SHARE, None
        # (11-13) a factor on every contact's time penalty (and so on the plan-hit cost, a share of it); 1 = the 11-07 prices
        self.contact_scale = 1.0
        # (11-13) the stall price of 11-06 ("--stall-mode progress"): each second the car's progress over the last
        # `stall_window_s` falls short of `stall_vmin` [m/s] costs `stall_s` seconds x the shortfall's share; 0 = off.
        # Progress = the drop of the walls-only distance left, so standing, rocking and going the wrong way all pay, and a
        # quick back-out pays only while it lasts. The window starts again at every restart.
        self.stall_s, self.stall_vmin, self.stall_window_s = 0.0, 1.0, 2.0
        self._st_buf = None
        self.stall_time = torch.zeros(Bl, device=device)        # [s] the shortfall's share, summed (for the log)
        self.begin(torch.arange(Bl, device=device))

    def _d_car(self) -> torch.Tensor:
        st = self.env.sim.state[self.lid]
        return self.goal.remaining(st[:, :2], self.env.sim.tid[self.lid])

    def _phi_pts(self, pts: torch.Tensor, d_car: torch.Tensor) -> torch.Tensor:
        """-(mean distance left over the plan's points) / lap. A point in a wall reads the one before it (the car's for
        the first), so the potential stays a function of the plan. A point past the line reads a lap more than the
        car's side: the potential jumps once per point per lap, and the sum over a lap still telescopes."""
        tid = self.env.sim.tid[self.lid]
        L = self.goal.lap[tid]
        d = self.goal.remaining(pts, tid[:, None].expand(-1, pts.shape[1]))
        prev = d_car
        cols = []
        for k in range(d.shape[1]):
            cur = torch.where(torch.isfinite(d[:, k]), d[:, k], prev)
            cols.append(cur); prev = cur
        return -torch.stack(cols, 1).mean(1) / L

    def begin(self, rows: torch.Tensor) -> None:
        """rows (indices into the learner rows) just (re)started: no plan yet, Phi from where the car stands"""
        if rows.numel() == 0:
            return
        d = self._d_car()[rows]
        L = self.goal.lap[self.env.sim.tid[self.lid][rows]]
        d = torch.where(torch.isfinite(d), d, L)
        self.d_prev[rows] = d
        self.phi[rows] = -d / L
        self.cost[rows] = 0.0
        self.seen_t[rows] = 1e9
        if self._st_buf is not None:
            self._st_buf[:, rows] = 0.0; self._st_sum[rows] = 0.0; self._st_n[rows] = 0
            self._st_d[rows] = d

    def control_step(self, seen_car: Optional[torch.Tensor] = None, restarted: Optional[torch.Tensor] = None) -> None:
        """after an `env.step`: charge the contacts it began; `seen_car` (B_l,) bool = a beam hit a car this step;
        `restarted` (B_l,) bool = the row reset in this step (the stall window starts again)"""
        env = self.env
        dt = float(env.sim.control_dt)
        if self.stall_s > 0:
            self._stall(dt, restarted)
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
        self.cost = self.cost + self.contact_scale * ((CONTACT_S + CONTACT_S_PER_MPS * imp) * charge).sum(1) / T_REF
        self.n_contacts = self.n_contacts + on

    def _stall(self, dt: float, restarted: Optional[torch.Tensor]) -> None:
        """the stall price of one control step (see `__init__`)"""
        d = self._d_car()
        L = self.goal.lap[self.env.sim.tid[self.lid]]
        if self._st_buf is None:
            W = max(1, int(round(self.stall_window_s / dt)))
            self._st_buf = torch.zeros(W, d.shape[0], device=d.device)
            self._st_sum = torch.zeros(d.shape[0], device=d.device)
            self._st_n = torch.zeros(d.shape[0], dtype=torch.long, device=d.device)
            self._st_ptr = 0
            self._st_d = torch.where(torch.isfinite(d), d, L)
        d = torch.where(torch.isfinite(d), d, self._st_d)          # inside a wall for a moment: hold the last reading
        step = self._st_d - d                                       # [m] forward this control step
        step = step + L * ((step < -0.5 * L).float() - (step > 0.5 * L).float())   # across the finish line
        fresh = step.abs() > 1.0                                    # a reset: no car covers 1 m in a control step
        if restarted is not None:
            fresh = fresh | restarted
        step = torch.where(fresh, torch.zeros_like(step), step)
        W = self._st_buf.shape[0]
        self._st_sum = self._st_sum + step - self._st_buf[self._st_ptr]
        self._st_buf[self._st_ptr] = step
        self._st_ptr = (self._st_ptr + 1) % W
        self._st_n = (self._st_n + 1).clamp(max=W)
        if bool(fresh.any()):
            self._st_buf[:, fresh] = 0.0
            self._st_sum = torch.where(fresh, torch.zeros_like(self._st_sum), self._st_sum)
            self._st_n = torch.where(fresh, torch.zeros_like(self._st_n), self._st_n)
        rate = self._st_sum / (self._st_n.clamp(min=1).float() * dt)
        short = ((self.stall_vmin - rate) / max(self.stall_vmin, 1e-6)).clamp(0.0, 1.0) * (self._st_n > 0).float()
        self.cost = self.cost + self.stall_s * dt * short / T_REF
        self.stall_time = self.stall_time + dt * short
        self._st_d = d

    def plan_cost(self, x: torch.Tensor, y: torch.Tensor, psi: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        """A plan just drawn, its poses in the world (B_l, n) each and speeds (B_l, n): the first pose whose footprint
        is in a wall or a box, and its price [laps] (0 for a clear plan). The pose now (index 0) is the car, not the plan."""
        tid = self.env.sim.tid[self.lid]
        c, s = torch.cos(psi)[..., None], torch.sin(psi)[..., None]
        cx = x[..., None] + self.corners[:, 0] * c - self.corners[:, 1] * s           # (B_l, n, 4)
        cy = y[..., None] + self.corners[:, 0] * s + self.corners[:, 1] * c
        pts = torch.stack([cx, cy], -1)
        tt = tid[:, None, None].expand_as(cx)
        hit = (self.env.sim.track.sample_edt(pts, tt) <= 0.0).any(-1)                  # (B_l, n)
        proc = getattr(self.env, "procedural", None)
        if proc is not None:
            eid = self.env.sim.eid[self.lid][:, None, None].expand_as(cx).reshape(-1)
            cl = proc.clearance(pts.reshape(-1, 2), eid).reshape(cx.shape)
            hit = hit | (cl <= 0.0).any(-1)
        hit[:, 0] = False
        if self.plan_hit_n is not None:                                               # only the plan's first poses
            hit[:, self.plan_hit_n + 1:] = False
        any_ = hit.any(1)
        first = torch.where(any_, hit.float().argmax(1), torch.zeros_like(any_, dtype=torch.long))
        v_hit = v.gather(1, first[:, None])[:, 0].abs()
        price = torch.where(any_, self.contact_scale * self.plan_hit_share * (CONTACT_S + CONTACT_S_PER_MPS * v_hit) / T_REF,
                            torch.zeros_like(v_hit))
        return {"hit": any_, "first": first, "price": price}

    def decision(self, pts: torch.Tensor, plan_price: Optional[torch.Tensor] = None,
                 restarted: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """The reward of the decision just driven. `pts` (B_l, 10, 2): the plan driven, its points in the world (it is
        the state's plan now). `plan_price` (B_l,) [laps] from `plan_cost` when it was drawn. `restarted` (B_l,) bool:
        rows reset during it -- no shaping across an episode boundary (their Phi starts again from where they stand),
        their costs still charged."""
        d = self._d_car()
        tid = self.env.sim.tid[self.lid]
        L = self.goal.lap[tid]
        d = torch.where(torch.isfinite(d), d, self.d_prev)      # inside a wall for a moment: hold the last reading
        jump = d - self.d_prev
        lap = (jump > 0.5 * L).float() - (jump < -0.5 * L).float()   # the car crossed the line forwards / backwards
        phi = self._phi_pts(pts, d)
        shaping = self.gamma * phi - self.phi
        if restarted is not None:
            shaping = torch.where(restarted, torch.zeros_like(shaping), shaping)
            lap = torch.where(restarted, torch.zeros_like(lap), lap)
        pp = plan_price if plan_price is not None else torch.zeros_like(lap)
        out = {"lap": lap, "shaping": shaping, "contact": -self.cost, "plan_hit": -pp,
               "reward": lap + shaping - self.cost - pp}
        self.phi, self.d_prev = phi, d
        self.cost = torch.zeros_like(self.cost)
        if restarted is not None and bool(restarted.any()):
            self.begin(torch.nonzero(restarted)[:, 0])
        return out
