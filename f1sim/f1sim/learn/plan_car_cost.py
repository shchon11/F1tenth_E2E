"""The plan's overlap with the other car's REAL future (11-24, plan_rl --plan-car-cost; the race phase, agreed with the user).

Why: race32c rear-ends slow cars (5-6 at-fault contacts a session against a 0.6x copy). On its u1000 exam4 record 15 % of
the plans drawn with a 0.6x car 0.5-4 m ahead pass within 0.25 m of where that car really is at the same moment (5 % at
0.8x): the policy treats cars as moving, but expects slow ones to clear the way faster than they do. The box plan cost
prices a plan through a box from the box's true place; this is the same hindsight price for a car, from the simulator's
true future of the other car -- never from the prediction head.

* Time-aligned: plan point k (time t_k after the decision, the plan's own stamps: 0.1 k s) is compared with the other car
  where it really was t_k after the decision (the decision ``kofs[k]`` later).
* Footprints at true size: each car covered by 3 discs along its length, centres at 0 and +-L/3, radius
  r = sqrt((W/2)^2 + (L/6)^2) (0.183 m for 0.58 x 0.31). An overlap is a disc pair closer than 2 r - ``margin``: the
  margin is the disc overlap allowed, so a tight side-by-side pass (centres 0.35 m apart, 4 cm of daylight between the
  bodies) is not one -- sideways the threshold is then the body width (0.315 vs 0.31 m), nose to tail the cover first
  overlaps at 0.70 m between centres (the bodies touch at 0.58). Ours: the plan's pose (its heading channel); theirs:
  their true pose (true yaw).
* Knowable only: the other car within --pred-range and hit by a LiDAR beam within the last 1 s at the decision (the
  prediction head's 'there' rule) -- the caller's ``know``. Only up to a restart of either car (our episode / scene,
  the other car put elsewhere) or the rollout's end: the points from there on are not compared.
* At fault only: the env's rule (`GymEnv.car_contact_charge`: responsible when the other car is ahead along the track or
  less than a car length, 0.6 m, behind), read where the env's latch reads it -- at the contact's start: at the plan
  point just BEFORE the first overlapping one (our plan pose there against the other car's true pose then; the car's own
  pose for the first point). With 0.1 s points the first overlapping point can be 0.1 s x the closing speed into the
  other car; read there, a car closing from behind would come out 'alongside' and be charged to us.
* Price: the first overlapping point only (like the box plan cost's first hit), its weight w_k (near points weigh
  more: 1.0 at 0.1 s down to ``w_end`` at 1.0 s, linear) x the contact the env would charge for it, CONTACT_S +
  CONTACT_S_PER_MPS x the relative speed there [s]. The trainer turns that into laps with the contact's own scale
  (x contact_scale x --plan-car-cost / T_REF), so --plan-car-cost is a share of a contact, as --plan-hit-share is.

Written once for both array kinds (``xp`` = torch or numpy): the trainer runs it on the GPU, the exam table
(`tools/exam_race_table.py`) on numpy without importing torch."""
import math

CONTACT_S = 0.5             # [s] every contact      (plan_reward's; repeated here so the exam table needs no torch)
CONTACT_S_PER_MPS = 1.0     # [s per m/s of impact]
FAULT_GAP = -0.6            # [m] the env's rule: responsible unless the other car is more than this far behind


def disc_radius(length: float, width: float) -> float:
    """the radius of the 3 discs that cover a length x width car (centres at 0 and +-length/3)"""
    return math.sqrt((0.5 * width) ** 2 + (length / 6.0) ** 2)


def weights(n: int = 10, w_end: float = 0.3):
    """the points' weights: 1.0 at the first, linear down to `w_end` at the last"""
    return [1.0 - (1.0 - w_end) * k / max(1, n - 1) for k in range(n)]


def _discs(xp, xy, psi, length):
    """(..., 2), (...) -> (..., 3, 2): the disc centres along the car's length"""
    c, s = xp.cos(psi)[..., None], xp.sin(psi)[..., None]
    z = psi * 0.0
    off = xp.stack([z - length / 3.0, z, z + length / 3.0], -1)          # (..., 3)
    return xp.stack([xy[..., 0:1] + off * c, xy[..., 1:2] + off * s], -1)


def overlap(xp, xy_a, psi_a, xy_b, psi_b, length: float, width: float, margin: float = 0.05):
    """(...) bool: the two cars' disc covers overlap by more than `margin` (any disc pair closer than 2 r - margin)"""
    da, db = _discs(xp, xy_a, psi_a, length), _discs(xp, xy_b, psi_b, length)       # (..., 3, 2)
    d = da[..., :, None, :] - db[..., None, :, :]                                     # (..., 3, 3, 2)
    d2 = (d * d).sum(-1)
    thr = max(2.0 * disc_radius(length, width) - margin, 0.0)
    return (d2 < thr * thr).reshape(d2.shape[:-2] + (9,)).any(-1)


def _wrap(xp, d, L):
    return xp.remainder(d + 0.5 * L, L) - 0.5 * L


def along_gap(xp, me_xy, me_psi, me_rem, op_xy, op_rem, L):
    """(...) [m] the other car ahead of us along the track (+) or behind (-): from the walls-only distance to go (the
    reward's field, no centre line) when both are on it, else along our heading"""
    g = _wrap(xp, me_rem - op_rem, L)
    d = op_xy - me_xy
    ego = d[..., 0] * xp.cos(me_psi) + d[..., 1] * xp.sin(me_psi)
    return xp.where(xp.isfinite(g), g, ego)


def _shift(f, at0):
    """(T, B, K) -> (T, B, K): column 0 = `at0` (T, B), column j = f's column j-1 (the value one point earlier)"""
    out = f * 0.0
    out[:, :, 0] = at0
    out[:, :, 1:] = f[:, :, :-1]
    return out


def plan_car_cost(xp, me_xy, me_psi, me_v, me_rem, op_xy, op_psi, op_v, op_rem, op_ok, brk, know, lap, kofs, w,
                  length: float, width: float, margin: float = 0.05, fault_gap: float = FAULT_GAP):
    """The cost of every decision of a rollout (T decisions, B cars, K plan points).

    me_xy (T, B, K+1, 2), me_psi / me_rem (T, B, K+1), me_v (T, B, K+1, 2): index 0 our car at the decision (pose, distance
    to go), 1..K the plan's points (pose, world velocity, distance to go). op_xy (T+1, B, 2), op_psi / op_rem (T+1, B), op_v
    (T+1, B, 2), op_ok (T+1, B) bool: the other car at each decision's start and after the last (index T), present or
    not. brk (T, B) bool: either car restarted / was put elsewhere during decision t. know (T, B) bool: the other car
    knowable at decision t. lap (T, B): the lap's length [m]. kofs: K ints, the decisions after t at which each plan point
    falls (1..K for a 0.1 s plan). w: K weights.

    Returns a dict of (T, B): ``price`` [s of contact, weighted] (0 where not charged), ``charged`` / ``over`` bool (a
    charged overlap / any overlap among the points compared), ``first`` the first overlapping point (1..K, 0 none),
    ``n_ok`` the points compared, ``gap0`` the other car along the track at the decision (+ ahead)."""
    T, K = me_psi.shape[0], len(kofs)
    zf = me_psi[:, :, 1:] * 0.0                                            # (T, B, K) zeros of the right kind / device
    fx, fy, fpsi, frem, fvx, fvy = (zf + 0.0 for _ in range(6))           # the other car at t + kofs[j]
    ok = zf > 1.0                                                          # (T, B, K) the point is compared
    b_ = brk * 1
    cb = xp.cumsum(b_, 0)
    before = cb - b_                                                       # breaks before decision t
    for j, k in enumerate(kofs):
        k = int(k)
        n = T - k + 1                                                      # t + k <= T: inside the rollout
        if n <= 0 or k < 1:
            continue
        sl = slice(k, k + n)
        fx[:n, :, j] = op_xy[sl][..., 0]; fy[:n, :, j] = op_xy[sl][..., 1]; fpsi[:n, :, j] = op_psi[sl]
        frem[:n, :, j] = op_rem[sl]; fvx[:n, :, j] = op_v[sl][..., 0]; fvy[:n, :, j] = op_v[sl][..., 1]
        ok[:n, :, j] = ((cb[k - 1:k - 1 + n] - before[:n]) == 0) & op_ok[sl]
    ok = (xp.cumsum((~ok) * 1, -1) == 0) & know[..., None]               # a prefix: none after the first not compared
    ov = overlap(xp, me_xy[:, :, 1:], me_psi[:, :, 1:], xp.stack([fx, fy], -1), fpsi, length, width, margin) & ok
    over = ov.any(-1)
    idx = xp.cumsum(zf + 1.0, -1) - 1.0                                    # 0 .. K-1
    first = xp.argmax(ov * (float(K) - idx), -1)                           # the first overlapping point (0-based)
    hot = (idx == first[..., None] * 1.0) & over[..., None]                # (T, B, K) one-hot there
    hotf = hot * 1.0
    # the env's fault rule just before it: our pose one point earlier (the car itself before the first point) against the
    # other car then (at the decision itself before the first point)
    pxy = xp.stack([_shift(fx, op_xy[:T][..., 0]), _shift(fy, op_xy[:T][..., 1])], -1)
    prem = _shift(frem, op_rem[:T])
    gap_pre = along_gap(xp, me_xy[:, :, :K], me_psi[:, :, :K], me_rem[:, :, :K], pxy, prem, lap[..., None])
    gap = xp.where(hot, gap_pre, zf).sum(-1)
    charged = over & (gap > fault_gap)
    rvx = me_v[:, :, 1:, 0] - fvx; rvy = me_v[:, :, 1:, 1] - fvy
    vrel = (xp.sqrt(rvx * rvx + rvy * rvy) * hotf).sum(-1)
    wk = zf + 0.0
    for j in range(K):
        wk[:, :, j] = float(w[j])
    price = xp.where(charged, (wk * hotf).sum(-1) * (CONTACT_S + CONTACT_S_PER_MPS * vrel), zf[..., 0])
    gap0 = along_gap(xp, me_xy[:, :, 0], me_psi[:, :, 0], me_rem[:, :, 0], op_xy[:T], op_rem[:T], lap)
    return {"price": price, "charged": charged, "over": over, "first": xp.where(over, first + 1, first * 0),
            "n_ok": (ok * 1).sum(-1), "gap0": gap0}
