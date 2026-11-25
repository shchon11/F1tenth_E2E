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


def _later(xp, zf, fields, op_ok, brk, kofs):
    """the other car's `fields` ((T+1, B) each: at each decision's start and after the last) k = kofs[j] decisions after each
    decision t -> ((T, B, K) each, ok (T, B, K)): ok where t + k is inside the rollout, no break (`brk` (T, B): either car
    restarted / put elsewhere during a decision) came in between, and the car is there then (`op_ok` (T+1, B)). `zf` (T, B, K)
    zeros of the right kind / device"""
    T = zf.shape[0]
    outs = [zf + 0.0 for _ in fields]
    ok = zf > 1.0
    b_ = brk * 1
    cb = xp.cumsum(b_, 0)
    before = cb - b_                                                       # breaks before decision t
    for j, k in enumerate(kofs):
        k = int(k)
        n = T - k + 1                                                      # t + k <= T: inside the rollout
        if n <= 0 or k < 1:
            continue
        sl = slice(k, k + n)
        for o_, f_ in zip(outs, fields):
            o_[:n, :, j] = f_[sl]
        ok[:n, :, j] = ((cb[k - 1:k - 1 + n] - before[:n]) == 0) & op_ok[sl]
    return outs, ok


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
    (fx, fy, fpsi, frem, fvx, fvy), ok = _later(xp, zf, [op_xy[..., 0], op_xy[..., 1], op_psi, op_rem, op_v[..., 0],
                                                         op_v[..., 1]], op_ok, brk, kofs)   # the other car at t + kofs[j]
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


# ------------------------------------------------------------------------------------------------------------------------
# (11-25, plan_rl --pred-clear; agreed with the user) the policy's own read of "does my plan run into what is coming", as an
# input. The user watched race training: the policy plans a wide detour around where the other car IS, the car moves, the
# next decision detours around the new spot -- an endless chase. Its prediction head's path was there but unread (a 0.5 m
# shift of it moved the plan 0.5 cm, race33b u875). So the overlap test the plan-car cost does in hindsight is done at every
# decision on the PREDICTION and given to the trunk as 10 numbers -- the same geometry as above.
#
# THE INTERFACE (11-25, the user: "정적 객체가 여러개라도 모듈이 달라지는게 아니잖아" -- the mechanism must not depend on how many
# moving things there are, as the scan does not on the number of boxes):
#   clearance(our footprint's poses at t_1..t_K, a SET of obstacle elements at t_1..t_K) -> (B, K) the smallest free distance
#   at each t_k. An element is a disc: its centre at each t_k (B, M, K, 2), its radius (a float or broadcastable to (B, M)),
#   and a confidence ((B, M) or (B, M, K), in [0, 1]); an element counts at t_k where its confidence >= w_min. M is free:
#     * today: the head's one predicted car = 3 discs (`car_cover`, the same cover as the cost above), confidence = its
#       P(there) -- a car slot not there / an empty slot has confidence 0;
#     * several car slots (a head predicting more cars, each with its own P(there)): 3 discs per slot, side by side in M;
#     * a predicted future scan (points per beam at 0.5 s / 1.0 s, interpolated to each t_k): one element of radius 0 per
#       beam point (a beam that hit nothing: confidence 0).
#   Neither changes the result's shape (B, K) nor this contract, so the trunk input stays 10 numbers whatever feeds it.
# Clearance of one pair = |centre_ours - centre_elem| - r_ours - r_elem + margin; negative = an overlap by more than the margin
# (for two car covers: exactly `overlap`'s test). No element counting at t_k -> `far`; the result is clipped to [lo, hi].

def car_cover(xp, xy, psi, length: float, width: float, conf=None):
    """C cars as clearance elements: xy (B, C, K, 2), psi (B, C, K), conf (B, C) or (B, C, K) (None: 1) -> (centres
    (B, 3C, K, 2), radius (float), confidence (B, 3C) / (B, 3C, K)): each car's 3 discs (element 3c + j = car c's disc j)"""
    B, C, K = psi.shape[0], psi.shape[1], psi.shape[2]
    d = xp.moveaxis(_discs(xp, xy, psi, length), 3, 2)                     # (B, C, 3, K, 2)
    ob_xy = d.reshape(B, 3 * C, K, 2)
    if conf is None:
        conf = psi[:, :, 0] * 0.0 + 1.0
    w = xp.stack([conf, conf, conf], 2)                                    # (B, C, 3[, K])
    return ob_xy, disc_radius(length, width), w.reshape((B, 3 * C) + tuple(w.shape[3:]))


def clearance(xp, me_xy, me_psi, ob_xy, ob_r, ob_w, length: float, width: float, margin: float = 0.05, far: float = 3.0,
              lo: float = -1.0, hi: float = 3.0, w_min: float = 0.5):
    """(B, K) [m]: at each t_k the smallest clearance between our car's 3-disc cover (centre me_xy (B, K, 2), heading me_psi
    (B, K); the cover of a length x width car) and the elements that count then (see THE INTERFACE above: ob_xy (B, M, K, 2),
    ob_r float or broadcastable to (B, M), ob_w (B, M) / (B, M, K) or None = all count). `far` where none counts; clipped to
    [lo, hi]"""
    me = _discs(xp, me_xy, me_psi, length)                                 # (B, K, 3, 2)
    d = me[:, None, :, :, :] - ob_xy[:, :, :, None, :]                     # (B, M, K, 3, 2)
    dist = xp.amin(xp.sqrt((d * d).sum(-1)), -1)                           # (B, M, K): the nearest of our discs
    r_ob = ob_r[..., None] if (hasattr(ob_r, "shape") and len(ob_r.shape) > 0) else ob_r
    c = dist - disc_radius(length, width) - r_ob + margin
    if ob_w is not None:
        cnt = ob_w[..., None] >= w_min if len(ob_w.shape) == 2 else ob_w >= w_min
        c = xp.where(cnt, c, c * 0.0 + float("inf"))
    m = xp.amin(c, 1) if c.shape[1] > 0 else me_psi * 0.0 + float("inf")  # (B, K)
    m = xp.where(xp.isfinite(m), m, me_psi * 0.0 + far)                    # (not m * 0: inf * 0 is NaN)
    return xp.clip(m, lo, hi)


AHEAD_LON = (0.5, 4.0)       # [m] "a car ahead": its centre this far ahead of ours along our heading at the decision
AHEAD_LAT = 1.0              # [m] and less than this to the side


def moving_read(xp, me_xy, me_psi, op_xy, op_psi, op_ok, brk, kofs, length: float, width: float, margin: float = 0.05):
    """(11-25) does a plan treat the car ahead as moving? Per decision (T, B) of a rollout / record:
      * ``ahead``: the other car there, its centre 0.5-4 m ahead of ours along our heading and |lateral| < 1 m (our frame at
        the decision), and every plan point comparable with where it really was then (no break, inside the rollout);
      * ``now``: a plan point's footprint overlaps the other car's where it is AT the decision (as if it stood still);
      * ``fut``: a plan point's footprint overlaps the other car's where it really is at that point's time (`plan_car_cost`'s
        time-aligned test, no knowable / fault rule);
      * ``want`` = ahead & now & ~fut: the plan goes through the car's current spot and stays clear of its real future --
        what a plan behind a moving car should do (a plan that avoids ``now`` treats it as a box).
    me_xy (T, B, K+1, 2), me_psi (T, B, K+1): index 0 our car at the decision, 1..K the plan's points (pose: the plan's
    heading channel, or point to point in a record); op_xy (T+1, B, 2), op_psi / op_ok (T+1, B), brk (T, B) as above."""
    T, K = me_psi.shape[0], len(kofs)
    zf = me_psi[:, :, 1:] * 0.0
    (fx, fy, fpsi), ok = _later(xp, zf, [op_xy[..., 0], op_xy[..., 1], op_psi], op_ok, brk, kofs)
    full = (ok * 1).sum(-1) == K
    pxy, ppsi = me_xy[:, :, 1:], me_psi[:, :, 1:]
    fut = (overlap(xp, pxy, ppsi, xp.stack([fx, fy], -1), fpsi, length, width, margin) & ok).any(-1)
    nx = zf + op_xy[:T, :, None, 0]; ny = zf + op_xy[:T, :, None, 1]; npsi = zf + op_psi[:T, :, None]
    now = overlap(xp, pxy, ppsi, xp.stack([nx, ny], -1), npsi, length, width, margin).any(-1)
    dx = op_xy[:T, :, 0] - me_xy[:, :, 0, 0]; dy = op_xy[:T, :, 1] - me_xy[:, :, 0, 1]
    c, s = xp.cos(me_psi[:, :, 0]), xp.sin(me_psi[:, :, 0])
    lon, lat = dx * c + dy * s, -dx * s + dy * c
    ahead = op_ok[:T] & (lon >= AHEAD_LON[0]) & (lon <= AHEAD_LON[1]) & (xp.abs(lat) < AHEAD_LAT) & full
    return {"ahead": ahead, "now": now, "fut": fut, "want": ahead & now & ~fut}
