"""ForzaETH's MAP controller (Model- and Acceleration-based Pursuit): the planner baselines' path
follower (2026-10-19).

The user, on the planner baselines following their path by pure pursuit: "waypoint 추종하는거 pure
pursuit이여? 에바야 레이싱에 무슨.. 그 MAP 제어기라고 그 가벼운거 있거든. 그걸로 해봐. 근데 그거 깃헙에 있는
코드가 논문 설명과 좀 다를거여. L1 distance 쪽인가? 암튼 유의해." -- then "제어기 튜닝도 잘 해놔".

Read side by side:

* the paper: Becker, Imholz, Schwarzenbach, Ghignone, Baumann, Magno, "Model- and Acceleration-based
  Pursuit Controller for High-Performance Autonomous Racing", ICRA 2023 (arXiv 2209.04346v4);
* its code: `ETH-PBL/MAP-Controller` @ 5cce331 (`map_controller/src/MAP_controller.py`,
  `steering_lookup/src/LUT_Generation/simulate_model.py`, `.../lookup_steer_angle.py`);
* what ForzaETH races with: `ForzaETH/race_stack` @ 6a2900a (`controller/controller/map.py`,
  `controller_manager.py`, `system_identification/steering_lookup/.../lookup_steer_angle.py`,
  `stack_master/config/*/l1_params.yaml`). Clones in /data/ext_src (read-only, never staged).

Where they differ -- the L1 distance, as the user remembered, and more:

1. **The L1 distance.** Paper III-D: ``L_d = m + q * v_ctrl``, ``v_ctrl`` "the speed commanded to the
   car". ETH-PBL: ``q_map + target_speed * m_map`` (the commanded speed; the letters swap roles).
   race_stack: ``q_l1 + speed_now * m_l1`` -- the MEASURED speed -- clipped to
   ``[max(t_clip_min, sqrt(2) |d|), t_clip_max]``, the floor growing with the car's offset ``d`` from
   the GLOBAL line (none of this clipping is in the paper). `MAPParams.l1_speed`, `l1_floor`.
2. **The L1 point.** Paper (L1 guidance, Park et al. 2004): the reference point on the path at
   distance L1 from the vehicle. Both codes: the waypoint ``int(L1 / 0.1 + 0.5)`` indices after the
   one nearest the car -- L1 of ARC along the path, 0.1 m spacing assumed -- while ``a_lat`` still
   divides by the nominal L1. `MAPParams.l1_point`.
3. **The speed in a_lat and in the lookup.** Paper eq. 2: the tangential speed ``v_t``. Both codes:
   the planned speed at the waypoint nearest the car (race_stack: at the position propagated
   ``speed_lookahead_for_steer`` s ahead, scaled down with the lateral error; the measured speed when
   trailing). `MAPParams.steer_speed`.
4. **The lookup.** Paper: "interpolating between the closest elements in the table". Code: the
   NEAREST velocity column, then ``np.interp`` between the two accelerations closest to the request in
   that column, the pair in whichever order it came -- and ``np.interp`` interpolates only an
   increasing pair: between a decreasing one it returns the SECOND-closest's steering.
   `MAPParams.lut_interp`.
5. **race_stack only:** the steering x ``acc_scaler_for_steer`` / ``dec_scaler_for_steer`` while the
   mean of the last ten IMU longitudinal accelerations is >= 1 / <= -1 m/s^2; x ``1 - downscale *
   ramp(v; start_scale_speed, end_scale_speed)``; x ``clip(1 + v/10, 1, 1.25)``, a fixed hand gain
   (`MAPParams.v_steer_gain_max`); a 0.4 rad per-step change limit; the speed command read
   ``speed_lookahead`` s ahead and scaled down with lateral error x curvature; a PID trailing
   controller in the TRAILING state.

The default here is race_stack, what wins races (NUC2's parameters); every difference is a switch
the fair tuning is free to set. The lookup table is built the way ETH-PBL builds theirs -- constant
steering at constant commanded speed until the yaw rate settles, ``a_lat = v * r`` -- on this
simulator's own model through its own actuators (`build_lut`), which is the identified car the paper
assumes; one table per grip level of the reference teacher's profiles, as ForzaETH identifies one per
surface (`NUC2_hangar`, `NUC6_glc`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Sequence

import torch

from . import dynamics as dyn
from .actuators import servo_target, vesc_accel


# ------------------------------------------------------------------------------------- parameters
@dataclass
class MAPParams:
    """`race_stack/stack_master/config/NUC2/l1_params.yaml` (a real car's race setting) plus the
    paper-vs-code switches. Units: m, s, m/s, rad."""
    t_clip_min: float = 0.8
    t_clip_max: float = 5.0
    m_l1: float = 0.583            # slope on speed (race_stack's letters: L1 = q_l1 + v * m_l1)
    q_l1: float = -0.167           # intercept
    speed_lookahead: float = 0.25  # [s] the speed command is read this far ahead along the velocity
    lat_err_coeff: float = 1.0     # [0, 1] speed reduction with lateral error x curvature
    acc_scaler_for_steer: float = 1.2
    dec_scaler_for_steer: float = 0.9
    start_scale_speed: float = 7.0
    end_scale_speed: float = 8.0
    downscale_factor: float = 0.2
    speed_lookahead_for_steer: float = 0.0
    v_steer_gain_max: float = 1.25  # race_stack's hard-coded steer *= clip(1 + v/10, 1, 1.25)
    steer_rate_limit: float = 0.4   # [rad per control step], race_stack's `threshold`
    # trailing (race_stack's PID, in the planner's TRAILING state)
    prioritize_dyn: bool = True
    trailing_gap: float = 2.0
    trailing_p_gain: float = 1.0
    trailing_i_gain: float = 0.0
    trailing_d_gain: float = 0.2
    blind_trailing_speed: float = 1.5
    loop_rate: float = 40.0
    # paper-vs-code switches (module docstring, items 1-4)
    l1_speed: str = "measured"      # "measured" (race_stack) | "commanded" (paper III-D, ETH-PBL)
    l1_floor: bool = True           # race_stack's max(t_clip_min, sqrt(2)|d|) lower bound
    l1_point: str = "arc"           # "arc" (both codes) | "euclid" (the paper's L1 guidance)
    steer_speed: str = "planned"    # "planned" (both codes) | "measured" (paper eq. 2's v_t)
    lut_interp: str = "code"        # "code" (nearest v column, np.interp pair) | "linear" (bilinear)
    #: [s] not in MAP: the pose is first carried forward this long at the measured speed and yaw rate,
    #: the actuation delay the reference teacher's pure pursuit compensates (cmd_delay + servo_tau / 2,
    #: 0.055 s on this car). 0 is race_stack.
    delay_comp: float = 0.0
    #: [1/m] not in MAP either: the speed command x clip(1 - lat_slow * |error to the path|, 0.3, 1), the
    #: off-line slowdown of the pure pursuit MAP replaces (its 0.6). race_stack's own reduction is gated by
    #: the curvature ahead (2 mean|kappa| / 0.8 - 2 > 0, i.e. hairpins of radius < 1.25 m) and so is
    #: almost never active on these tracks. 0 is race_stack.
    lat_slow: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "MAPParams":
        names = {f.name for f in fields(cls)}
        bad = set(d or {}) - names
        if bad:
            raise ValueError(f"unknown MAP parameter(s) {sorted(bad)}")
        return cls(**dict(d or {}))

    def __post_init__(self):
        for name, ok in (("l1_speed", ("measured", "commanded")), ("l1_point", ("arc", "euclid")),
                         ("steer_speed", ("planned", "measured")), ("lut_interp", ("code", "linear"))):
            if getattr(self, name) not in ok:
                raise ValueError(f"MAPParams.{name} must be one of {ok}, not {getattr(self, name)!r}")


#: `race_stack/stack_master/config/SIM/l1_params.yaml`, the simulator setting, the other published
#: starting point (the fair tuning enqueues both).
SIM_PARAMS = dict(t_clip_min=1.0, t_clip_max=5.0, m_l1=0.3, q_l1=0.15, speed_lookahead=0.0, lat_err_coeff=1.0,
                  acc_scaler_for_steer=1.0, dec_scaler_for_steer=1.0, start_scale_speed=7.0, end_scale_speed=8.0,
                  downscale_factor=0.2, speed_lookahead_for_steer=0.0, trailing_gap=1.5, trailing_p_gain=0.5,
                  trailing_i_gain=0.001, trailing_d_gain=0.2, blind_trailing_speed=1.5)


# ------------------------------------------------------------------------------------ lookup table
#: ETH-PBL's grid, stretched to this car: 30 fine steering values to 0.1 rad and 30 coarse ones to the
#: steering stop (theirs 0.4), speeds 0.5 m/s up in 0.1 m/s steps to past the speed cap (theirs 7).
LUT_FINE_END = 0.1
LUT_N_STEER = 30
LUT_V = (0.5, 10.0, 96)
LUT_SETTLE_S = 3.0             # ETH-PBL simulates 2 s at 10 ms; this car has servo and motor lags
LUT_CONVERGED = 0.05           # [rad/s] ETH-PBL: |r(T) - r(T - 9 dt)| above this is "no steady state"
LUT_VERSION = "map-lut-v1"


def lut_grids(s_max: float, device="cpu"):
    fine = torch.linspace(0.0, LUT_FINE_END, LUT_N_STEER, dtype=torch.float64)
    coarse = torch.linspace(LUT_FINE_END, float(s_max), LUT_N_STEER, dtype=torch.float64)
    steers = torch.cat([fine, coarse]).to(device=device, dtype=torch.float32)
    vels = torch.linspace(LUT_V[0], LUT_V[1], LUT_V[2], dtype=torch.float64).to(device=device, dtype=torch.float32)
    return steers, vels


@torch.no_grad()
def build_lut(P_row: Dict[str, torch.Tensor], grip_mu: Sequence[float], *, wheel: bool, dt: float,
              steers: torch.Tensor, vels: torch.Tensor, settle_s: float = LUT_SETTLE_S,
              chunk: int = 200_000) -> torch.Tensor:
    """(G, S, V) steady-state lateral acceleration [m/s^2] for each friction in `grip_mu`, steering
    COMMAND in `steers` and commanded speed in `vels`; NaN where the yaw rate did not settle, and from
    there to every higher speed at that steering (ETH-PBL's `generate_lookup`).

    Driven through this simulator's own substep pieces -- `servo_target`, `vesc_accel`,
    `dynamics.step_dynamics` -- with `P_row`, one car's nominal parameter dict, so the table maps what
    the controller COMMANDS to what the car does, calibration and tyres included; the command delay,
    the IMU and the body roll do not move a steady state and are left out. ``a_lat = |v| r`` with the
    speed the car actually holds."""
    dev = steers.device
    G, S, V = len(grip_mu), steers.numel(), vels.numel()
    g_i, s_i, v_i = torch.meshgrid(torch.arange(G, device=dev), torch.arange(S, device=dev),
                                   torch.arange(V, device=dev), indexing="ij")
    g_i, s_i, v_i = g_i.reshape(-1), s_i.reshape(-1), v_i.reshape(-1)
    out = torch.full((G * S * V,), float("nan"), device=dev)
    mu = torch.tensor([float(m) for m in grip_mu], device=dev)
    n_sub = int(round(settle_s / dt))
    back = max(1, int(round(0.09 / dt)))                        # ETH-PBL: 9 steps of 10 ms
    for lo in range(0, g_i.numel(), chunk):
        sl = slice(lo, min(lo + chunk, g_i.numel()))
        n = sl.stop - sl.start
        P = {k: (v.reshape(-1)[:1].to(dev).expand(n).clone() if torch.is_tensor(v) and v.numel() >= 1 else v)
             for k, v in P_row.items()}
        P["mu"] = mu[g_i[sl]]
        steer_cmd, v_cmd = steers[s_i[sl]], vels[v_i[sl]]
        state = torch.zeros(n, dyn.STATE_DIM, device=dev)
        state[:, dyn.IVX] = v_cmd
        state[:, dyn.ISTEER] = servo_target(steer_cmd, P)
        state[:, dyn.IOMEGA] = v_cmd / P["r_w"]
        ax = torch.zeros(n, device=dev)
        r_hist = None
        for k in range(n_sub):
            steer_tgt = servo_target(steer_cmd, P)
            v_fb = state[:, dyn.IOMEGA] * P["r_w"] if wheel else state[:, dyn.IVX]
            a_cmd = vesc_accel(v_cmd, v_fb, P)
            state, ax, _ay = dyn.step_dynamics(state, steer_tgt, a_cmd, ax, P, P["servo_tau"], dt, wheel=wheel)
            if k == n_sub - 1 - back:
                r_hist = state[:, dyn.IR].clone()
        r = state[:, dyn.IR]
        speed = torch.sqrt(state[:, dyn.IVX] ** 2 + state[:, dyn.IVY] ** 2)
        ok = ((r - r_hist).abs() <= LUT_CONVERGED) & torch.isfinite(r) & (state[:, dyn.IVX] > 0.1)
        out[sl] = torch.where(ok, speed * r, torch.full_like(r, float("nan")))
    table = out.view(G, S, V)
    # ETH-PBL: once a steering angle fails to settle at some speed, every higher speed at that angle
    # is left empty (`self.lookup_table[steer_idx+1, vel_idx+1:] = None; break`).
    bad = torch.cumsum(torch.isnan(table).to(torch.int32), dim=2) > 0
    return torch.where(bad, torch.full_like(table, float("nan")), table)


class SteerLUT:
    """The steering lookup: (G, S, V) steady-state lateral accelerations, queried per car."""

    def __init__(self, steers: torch.Tensor, vels: torch.Tensor, table: torch.Tensor):
        self.steers, self.vels, self.table = steers, vels, table
        # ETH-PBL / race_stack truncate a column at its first NaN (`find_closest_neighbors`)
        self.valid = torch.cumprod((~torch.isnan(table)).to(torch.int32), dim=1).bool()       # (G, S, V)

    def to(self, device) -> "SteerLUT":
        return SteerLUT(self.steers.to(device), self.vels.to(device), self.table.to(device))

    def lookup(self, accel: torch.Tensor, vel: torch.Tensor, g: torch.Tensor, interp: str = "code") -> torch.Tensor:
        """Steering command for lateral acceleration `accel` at speed `vel` on grip table `g`, (R,) each."""
        sign = torch.where(accel > 0.0, torch.ones_like(accel), -torch.ones_like(accel))
        a = accel.abs()
        if interp == "linear":
            return sign * self._bilinear(a, vel, g)
        cv = (self.vels[None, :] - vel[:, None]).abs().argmin(1)                           # nearest column
        return sign * self._column(a, g, cv)

    def _column(self, a: torch.Tensor, g: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        """`lookup_steer_angle` on one column per row, np.interp's two-point semantics included."""
        S = self.steers.numel()
        col = self.table[g, :, cv]                                                          # (R, S)
        ok = self.valid[g, :, cv]
        dist = torch.where(ok, (col - a[:, None]).abs(), torch.full_like(col, float("inf")))
        c = dist.argmin(1)
        last = (ok.sum(1) - 1).clamp_min(0)
        R = a.shape[0]
        ar = torch.arange(R, device=a.device)
        lo = (c - 1).clamp_min(0)
        hi = torch.minimum(c + 1, last)
        # `find_nearest(array[[c-1, c+1]], value)`: the first of the two on a tie
        s = torch.where(dist[ar, hi] < dist[ar, lo], hi, lo)
        edge = (c == 0) | (c == last)                      # the code returns the closest twice
        s = torch.where(edge, c, s)
        xp0, xp1 = col[ar, c], col[ar, s]
        fp0, fp1 = self.steers[c], self.steers[s]
        # np.interp(a, [xp0, xp1], [fp0, fp1]) to the letter (numpy's `binary_search_with_guess`): right
        # of xp1 -> fp1 FIRST, then left of xp0 -> fp0, then a == xp1 -> fp1, a == xp0 -> fp0, else the
        # line. A decreasing pair (the second-closest below the closest) therefore returns fp1 for any
        # request between them -- the second-closest's steering.
        den = xp1 - xp0
        lerp = fp0 + (fp1 - fp0) * (a - xp0) / torch.where(den.abs() > 1e-12, den, torch.ones_like(den))
        out = torch.where(a > xp1, fp1, torch.where(a < xp0, fp0, torch.where(a >= xp1, fp1,
                          torch.where(a == xp0, fp0, lerp))))
        return torch.where(edge, fp0, out)

    def _bilinear(self, a: torch.Tensor, vel: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """The paper's "interpolating between the closest elements": linear in speed between the two
        bracketing columns, and in each column linear in acceleration along its rising part (a request
        past the column's largest settled acceleration gets that steering)."""
        V = self.vels.numel()
        v = vel.clamp(float(self.vels[0]), float(self.vels[-1]))
        hi = torch.searchsorted(self.vels, v.contiguous()).clamp(1, V - 1)
        lo = hi - 1
        w = ((v - self.vels[lo]) / (self.vels[hi] - self.vels[lo])).clamp(0.0, 1.0)
        return (1 - w) * self._rising(a, g, lo) + w * self._rising(a, g, hi)

    def _rising(self, a: torch.Tensor, g: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
        col = self.table[g, :, cv]
        ok = self.valid[g, :, cv]
        # the rising part: up to the largest settled acceleration
        masked = torch.where(ok, col, torch.full_like(col, -float("inf")))
        top = masked.argmax(1)
        S = self.steers.numel()
        k = torch.arange(S, device=a.device)[None, :]
        rising = ok & (k <= top[:, None])
        mono = torch.where(rising, torch.cummax(torch.where(rising, col, torch.full_like(col, -float("inf"))), 1).values,
                           torch.full_like(col, float("inf")))
        j = torch.searchsorted(mono.contiguous(), a[:, None].contiguous())[:, 0].clamp(1, S - 1)
        j = torch.minimum(j, top.clamp_min(1))
        ar = torch.arange(a.shape[0], device=a.device)
        x0, x1 = mono[ar, j - 1], mono[ar, j]
        y0, y1 = self.steers[j - 1], self.steers[j]
        t = ((a - x0) / torch.where((x1 - x0).abs() > 1e-9, x1 - x0, torch.ones_like(x0))).clamp(0.0, 1.0)
        return y0 + t * (y1 - y0)


def lut_for(P_row: Dict[str, torch.Tensor], grip_mu: Sequence[float], *, wheel: bool, dt: float, s_max: float,
            device, cache_dir: Optional[str] = None) -> SteerLUT:
    """`build_lut` behind an on-disk cache keyed by the parameters that shape it."""
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "f1sim", "map_lut")
    key = {k: [round(float(x), 9) for x in v.reshape(-1)[:1].tolist()] if torch.is_tensor(v) else v
           for k, v in sorted(P_row.items()) if torch.is_tensor(v) or isinstance(v, (int, float))}
    blob = json.dumps({"P": key, "mu": [round(float(m), 9) for m in grip_mu], "wheel": bool(wheel), "dt": float(dt),
                       "s_max": round(float(s_max), 9), "grid": [LUT_FINE_END, LUT_N_STEER, list(LUT_V)],
                       "settle": LUT_SETTLE_S, "v": LUT_VERSION}, sort_keys=True)
    h = hashlib.md5(blob.encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"lut_{h}.pt")
    steers, vels = lut_grids(s_max, device=device)
    if os.path.exists(path):
        table = torch.load(path, map_location=device, weights_only=True)
    else:
        # built on a GPU whenever there is one (80k cars x 3000 substeps), whatever device it is for
        build_dev = device if torch.device(device).type == "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")
        table = build_lut(P_row, grip_mu, wheel=wheel, dt=dt, steers=steers.to(build_dev), vels=vels.to(build_dev)).to(device)
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        torch.save(table.cpu(), tmp)
        os.replace(tmp, path)
    return SteerLUT(steers, vels, table.to(device))


# --------------------------------------------------------------------------------------- controller
class MAPController:
    """race_stack's `MAP_Controller.main_loop`, batched over R cars (`MAPParams` switches aside).

    Per call it is handed, per car: its pose and measured speed, its local path -- waypoints
    (R, n, 2) with planned speeds and curvatures at a fixed spacing, the waypoint array the state
    machine publishes -- its signed offset from the GLOBAL line (`position_in_map_frenet[1]`), and, in
    the TRAILING state, the opponent race_stack's `obstacle_cb` picks. It keeps what race_stack keeps
    between calls: the last steering command (the change limit), the trailing integral and the last
    ten IMU longitudinal accelerations (`acc_now`)."""

    ACC_BUFFER = 10

    def __init__(self, lut: SteerLUT, params: MAPParams, rows: int, device, s_max: float):
        self.lut, self.p, self.s_max = lut, params, float(s_max)
        self.curr_steer = torch.zeros(rows, device=device)
        self.i_gap = torch.zeros(rows, device=device)
        self.acc_now = torch.zeros(rows, self.ACC_BUFFER, device=device)
        #: `curvature_waypoints`: race_stack updates it in `calc_L1_point`, which `main_loop` runs AFTER
        #: `calc_speed_command`, so the speed command reads the previous step's value
        self.curv_wp = torch.zeros(rows, device=device)
        self.last = {}

    def reset(self, mask: torch.Tensor) -> None:
        m = mask.to(self.curr_steer.device)
        self.curr_steer = torch.where(m, torch.zeros_like(self.curr_steer), self.curr_steer)
        self.i_gap = torch.where(m, torch.zeros_like(self.i_gap), self.i_gap)
        self.acc_now = torch.where(m[:, None], torch.zeros_like(self.acc_now), self.acc_now)
        self.curv_wp = torch.where(m, torch.zeros_like(self.curv_wp), self.curv_wp)

    def push_imu(self, ax: torch.Tensor) -> None:
        """New IMU longitudinal accelerations (R, k), oldest first: race_stack's rolling `acc_now`."""
        for i in range(ax.shape[1]):
            self.acc_now = torch.cat([ax[:, i:i + 1].to(self.acc_now.dtype), self.acc_now[:, :-1]], 1)

    @staticmethod
    def _nearest(points: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        return ((path - points[:, None, :]) ** 2).sum(-1).argmin(1)

    def __call__(self, pos: torch.Tensor, yaw: torch.Tensor, v_now: torch.Tensor, path_xy: torch.Tensor,
                 path_v: torch.Tensor, path_kappa: torch.Tensor, spacing: torch.Tensor, d_global: torch.Tensor,
                 grip: torch.Tensor, trailing: Optional[torch.Tensor] = None, opp_gap: Optional[torch.Tensor] = None,
                 opp_vs: Optional[torch.Tensor] = None, opp_visible: Optional[torch.Tensor] = None,
                 ego_vs: Optional[torch.Tensor] = None, yaw_rate: Optional[torch.Tensor] = None,
                 delay: Optional[torch.Tensor] = None):
        """-> (steer command [rad], speed command [m/s]), (R,) each. `trailing` (R,) bool marks the
        cars whose planner is TRAILING an opponent that exists (race_stack: ``state == "TRAILING" and
        opponent is not None``); `opp_gap` / `opp_vs` / `opp_visible` describe it, `ego_vs` is the car's
        own Frenet speed."""
        p = self.p
        R, n = path_v.shape
        dev = pos.device
        ar = torch.arange(R, device=dev)
        trailing = torch.zeros(R, dtype=torch.bool, device=dev) if trailing is None else trailing
        if delay is not None or p.delay_comp > 0:
            # `delay` (R,): each car's own calibrated actuation delay, in place of the tuned constant
            t_ = delay if delay is not None else p.delay_comp
            r_ = torch.zeros_like(yaw) if yaw_rate is None else yaw_rate
            mid = yaw + 0.5 * r_ * t_
            pos = pos + (v_now * t_)[:, None] * torch.stack([torch.cos(mid), torch.sin(mid)], 1)
            yaw = yaw + r_ * t_
        vvec = torch.stack([torch.cos(yaw) * v_now, torch.sin(yaw) * v_now], 1)

        # calc_lateral_error_norm: |frenet d| to the GLOBAL line, normalised over 0..0.5 m to 0..0.5
        lateral_error = d_global.abs()
        lat_e_norm = 0.5 * lateral_error.clamp(0.0, 0.5) / 0.5

        idx_near = self._nearest(pos, path_xy)
        k_ = torch.arange(n, device=dev)[None, :]
        after = k_ >= idx_near[:, None]

        def speed_adjust_lat_err(speed, e_norm, curv_wp):
            curv = (2 * (curv_wp / 0.8) - 2).clamp(0.0, 1.0)
            return speed * (1 - p.lat_err_coeff + p.lat_err_coeff * torch.exp(-(2 * e_norm) * curv))

        # ---------------------------------------------------------------- longitudinal (calc_speed_command)
        idx_la = self._nearest(pos + vvec * p.speed_lookahead, path_xy)
        global_speed = path_v[ar, idx_la]
        if opp_gap is not None:
            gap_error = p.trailing_gap - opp_gap
            v_diff = (v_now if ego_vs is None else ego_vs) - opp_vs
            i_new = (self.i_gap + gap_error / p.loop_rate).clamp(-10.0, 10.0)
            trail = (opp_vs - gap_error * p.trailing_p_gain - i_new * p.trailing_i_gain
                     - v_diff * p.trailing_d_gain).clamp(min=0.0)
            trail = torch.minimum(trail, global_speed)
            if opp_visible is not None:
                blind = (~opp_visible) & (opp_gap > p.trailing_gap)
                trail = torch.where(blind, torch.maximum(torch.full_like(trail, p.blind_trailing_speed), trail), trail)
            speed_cmd = torch.where(trailing, trail, global_speed)
            self.i_gap = torch.where(trailing, i_new, torch.zeros_like(i_new))
        else:
            speed_cmd = global_speed
            self.i_gap = torch.zeros_like(self.i_gap)
        speed_cmd = speed_adjust_lat_err(speed_cmd, lat_e_norm, self.curv_wp)
        if p.lat_slow > 0:
            # the error to the path being followed: the distance to its nearest waypoint
            e_path = (path_xy[ar, idx_near] - pos).norm(dim=1)
            speed_cmd = speed_cmd * (1.0 - p.lat_slow * e_path).clamp(0.3, 1.0)

        # ---------------------------------------------------------------- lateral (calc_L1_point)
        # `curvature_waypoints`: mean |kappa| from the nearest waypoint to the end of the local array
        # (updated only while more than two waypoints are left ahead)
        n_after = after.sum(1)
        fresh = (path_kappa.abs() * after).sum(1) / n_after.clamp_min(1)
        self.curv_wp = torch.where(n_after > 2, fresh, self.curv_wp)
        v_l1 = v_now if p.l1_speed == "measured" else path_v[ar, idx_near]
        L1 = p.q_l1 + v_l1 * p.m_l1
        lower = torch.clamp(math.sqrt(2.0) * lateral_error, min=p.t_clip_min) if p.l1_floor \
            else torch.full_like(L1, p.t_clip_min)
        L1 = torch.minimum(torch.maximum(L1, lower), torch.full_like(L1, p.t_clip_max))    # np.clip(x, lo, hi)
        if p.l1_point == "arc":
            j = torch.minimum(idx_near + (L1 / spacing + 0.5).floor().long(), torch.full_like(idx_near, n - 1))
            l1_pt = path_xy[ar, j]
        else:
            # the first path point past the nearest one at distance >= L1 from the car, the crossing
            # interpolated on the segment before it (Park et al.'s reference point)
            dist = (path_xy - pos[:, None, :]).norm(dim=-1)
            beyond = (dist >= L1[:, None]) & after
            first = torch.where(beyond.any(1), beyond.float().argmax(1), torch.full_like(idx_near, n - 1))
            prev = torch.maximum(first - 1, idx_near)
            d0, d1 = dist[ar, prev], dist[ar, first]
            t = ((L1 - d0) / torch.where((d1 - d0).abs() > 1e-9, d1 - d0, torch.ones_like(d0))).clamp(0.0, 1.0)
            l1_pt = path_xy[ar, prev] + t[:, None] * (path_xy[ar, first] - path_xy[ar, prev])

        # ---------------------------------------------------------------- calc_steering_angle
        if p.steer_speed == "measured":
            v_lu_raw = v_now
        else:
            idx_st = self._nearest(pos + vvec * p.speed_lookahead_for_steer, path_xy)
            v_lu_raw = path_v[ar, idx_st]
        v_lu_raw = torch.where(trailing, v_now, v_lu_raw)
        v_lu = speed_adjust_lat_err(v_lu_raw, lat_e_norm, self.curv_wp)
        vec = l1_pt - pos
        norm = vec.norm(dim=1)
        cross = -torch.sin(yaw) * vec[:, 0] + torch.cos(yaw) * vec[:, 1]
        eta = torch.where(norm > 0, torch.asin((cross / norm.clamp_min(1e-12)).clamp(-1.0, 1.0)), torch.zeros_like(norm))
        sin_eta = torch.sin(eta)
        lat_acc = torch.where((L1 == 0) | (sin_eta == 0), torch.zeros_like(L1), 2 * v_lu ** 2 / L1.clamp_min(1e-9) * sin_eta)
        steer = self.lut.lookup(lat_acc, v_lu, grip, p.lut_interp)
        acc_mean = self.acc_now.mean(1)
        steer = torch.where(acc_mean >= 1, steer * p.acc_scaler_for_steer,
                            torch.where(acc_mean <= -1, steer * p.dec_scaler_for_steer, steer))
        span = max(0.1, p.end_scale_speed - p.start_scale_speed)
        steer = steer * (1 - ((v_lu - p.start_scale_speed) / span).clamp(0.0, 1.0) * p.downscale_factor)
        steer = steer * (1 + v_now / 10).clamp(1.0, max(1.0, p.v_steer_gain_max))
        steer = torch.minimum(torch.maximum(steer, self.curr_steer - p.steer_rate_limit), self.curr_steer + p.steer_rate_limit)
        self.curr_steer = steer
        self.last = {"l1_point": l1_pt, "l1": L1, "idx_near": idx_near, "lat_acc": lat_acc, "path_xy": path_xy,
                     "path_v": path_v}
        return steer.clamp(-self.s_max, self.s_max), speed_cmd.clamp_min(0.0)


def describe(params: MAPParams) -> dict:
    return {"controller": "MAP (ForzaETH race_stack controller/map.py)", **asdict(params)}


# ------------------------------------------------------------------------------ our policy's plan
#: Our policy's plan follower when a run or the console asks for MAP without numbers of its own: the
#: paper's formulation (ETH-PBL's map_params.yaml numbers; L1 on the commanded speed, the point at
#: distance L1, v_t in a_lat, the lookup interpolated, none of race_stack's extras) with each car's
#: calibrated actuation delay. Zero-shot on our ICCAS pick (solo protocol, 128 cars): 7.92 s 128/128
#: against the iLQR's 8.40 s.
POLICY_MAP = dict(t_clip_min=0.3, t_clip_max=5.0, m_l1=0.3, q_l1=0.15, speed_lookahead=0.0, lat_err_coeff=0.0,
                  acc_scaler_for_steer=1.0, dec_scaler_for_steer=1.0, start_scale_speed=7.0, end_scale_speed=8.0,
                  downscale_factor=0.0, speed_lookahead_for_steer=0.0, v_steer_gain_max=1.0, l1_speed="commanded",
                  l1_floor=False, l1_point="euclid", steer_speed="measured", lut_interp="linear", delay_comp=-1.0)


#: The reference teacher's grip levels (`teacher.RacelineTeacher.grip_levels`): the steering tables are
#: built at these multiples of the nominal friction and a car reads the one at or below its own.
GRIP_LEVELS = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2)


def nominal_lut(sim, device=None) -> SteerLUT:
    """The steering tables of `sim`'s nominal car at every grip level (cached; what the planner
    baselines' MAP reads too, `learn.common.build_map_tracker`)."""
    nominal = sim.params.nominal
    P_row = {k: torch.tensor([float(v)]) for k, v in nominal.items()}
    mu_nom = float(nominal["mu"])
    return lut_for(P_row, [mu_nom * g for g in GRIP_LEVELS], wheel=bool(sim.wheel_model), dt=float(sim.dt),
                   s_max=float(nominal["s_max"]), device=device if device is not None else sim.state.device)


def grip_index(P, nominal: dict) -> torch.Tensor:
    """(B,) the steering table a car reads: the grip level at or below its front grip."""
    lv = torch.tensor(GRIP_LEVELS, device=P["mu"].device, dtype=P["mu"].dtype)
    g = (P["mu"] * P["mu_f_scale"]) / (float(nominal["mu"]) * float(nominal["mu_f_scale"]))
    return (torch.searchsorted(lv, g.contiguous(), right=True) - 1).clamp(0, len(GRIP_LEVELS) - 1)


#: The plan's waypoint array: 0.1 m apart (race_stack's spacing), as long as the longest plan plus the
#: furthest a speed lookahead reaches, the last stretch straight on at the plan's end speed.
PLAN_WP_SPACING = 0.1
PLAN_WP_EXTRA_M = 4.0


class MAPPlanTracker:
    """Our policy's plan followed by MAP (2026-10-19, the user: "우리도 그냥 map 쓰자").

    Wraps the env's `mpc.PlanTracker` and replaces the commands of `map_rows` (the policy's cars): the
    plan's own path -- `mpc.path_points` of its curvature knots over its length -- every 0.1 m, with
    its speed profile, is the waypoint array, and the car stands at its start heading along it (the
    plan is in the frame of the pose it was made from). The other rows -- the scripted and league
    cars -- keep the iLQR they were built with, so the traffic a policy races is the traffic it raced.
    Everything the env and the viewers read off a tracker (`last_ref`, `last_pred`, `u_seq`, the
    hooks, `spec`) is the wrapped tracker's: for the MAP rows `last_ref` is the reference of the same
    plan. The actuation delay compensated is each car's calibrated one (`delay`), the tracker's.
    """

    def __init__(self, inner, lut: SteerLUT, params: MAPParams, nominal: dict):
        self._inner = inner
        self.map = MAPController(lut, params, inner.B, inner.device, inner.s_max)
        self.nominal = nominal
        self.map_rows = torch.zeros(inner.B, dtype=torch.bool, device=inner.device)
        #: set by the env each step: its parameter dict (the grip level) and new IMU samples
        self.P = None
        self.last_map = None

    def __getattr__(self, name):                       # spec, last_ref, u_seq, hooks, ... are the inner's
        return getattr(self.__dict__["_inner"], name)

    def __setattr__(self, name, value):
        if name in ("_inner", "map", "nominal", "map_rows", "P", "last_map"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def reset(self, ids: torch.Tensor):
        self._inner.reset(ids)
        m = torch.zeros(self._inner.B, dtype=torch.bool, device=self._inner.device)
        m[ids] = True
        self.map.reset(m)

    def push_imu(self, ax: torch.Tensor) -> None:
        self.map.push_imu(ax)

    def waypoints(self, action: torch.Tensor, v_meas: torch.Tensor, speed_cap: torch.Tensor):
        """(path (B, n, 2), speed (B, n), curvature (B, n)) of each plan, body frame, 0.1 m apart."""
        from . import mpc
        sp = self._inner.spec
        if sp.speed_mode == "linear":
            k, Lp, v0, v1 = mpc.decode(action, v_meas, self._inner.v_max, speed_cap, sp)
            prof = None
        elif sp.speed_mode != "wp4":
            k, Lp, prof = mpc.decode_profile(action, v_meas, self._inner.v_max, speed_cap, sp)
        else:
            raise ValueError("MAP follows the linear and profile plan formats; 'wp4' is not wired")
        x, y, psi, s, kap = mpc.path_points(k, Lp, 49, return_kappa=True)
        n = int(math.ceil((float(sp.len_max) + PLAN_WP_EXTRA_M) / PLAN_WP_SPACING)) + 1
        sg = (torch.arange(n, device=k.device, dtype=k.dtype) * PLAN_WP_SPACING)[None].expand(k.shape[0], n)
        S = s[:, -1:].clamp_min(1e-3)
        idx = torch.searchsorted(s.contiguous(), sg.clamp(max=S).contiguous()).clamp(1, s.shape[1] - 1)
        s_lo, s_hi = s.gather(1, idx - 1), s.gather(1, idx)
        w = ((sg.clamp(max=S) - s_lo) / (s_hi - s_lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        g_ = lambda a: a.gather(1, idx - 1) * (1 - w) + a.gather(1, idx) * w
        px, py, pp, pk = g_(x), g_(y), g_(psi), g_(kap)
        over = (sg - S).clamp_min(0.0)                                  # past the end: straight on
        pe = psi[:, -1:]
        px = px + over * torch.cos(pe)
        py = py + over * torch.sin(pe)
        pk = torch.where(sg > S, torch.zeros_like(pk), pk)
        frac = (sg / S).clamp(0.0, 1.0)
        if prof is None:
            pv = v0[:, None] + (v1 - v0)[:, None] * frac
        else:
            pos = frac * (prof.shape[1] - 1)
            i0 = pos.floor().clamp(max=prof.shape[1] - 2).long()
            ww = pos - i0.to(prof.dtype)
            pv = prof.gather(1, i0) * (1 - ww) + prof.gather(1, i0 + 1) * ww
        return torch.stack([px, py], -1), pv, pk

    @torch.no_grad()
    def __call__(self, action: torch.Tensor, v_meas: torch.Tensor, speed_cap: torch.Tensor,
                 yaw_rate: Optional[torch.Tensor] = None, delay=None) -> torch.Tensor:
        out = self._inner(action, v_meas, speed_cap, yaw_rate, delay)     # every row; the traffic keeps it
        if not bool(self.map_rows.any()) or self.P is None:
            return out
        B = action.shape[0]
        dev = action.device
        path, pv, pk = self.waypoints(action, v_meas, speed_cap)
        zeros = torch.zeros(B, device=dev, dtype=action.dtype)
        dly = None
        if delay is not None:
            dly = delay if torch.is_tensor(delay) else torch.full((B,), float(delay), device=dev)
        steer, speed = self.map(torch.zeros(B, 2, device=dev, dtype=action.dtype), zeros, v_meas.abs(), path, pv, pk,
                                torch.full((B,), PLAN_WP_SPACING, device=dev), zeros, grip_index(self.P, self.nominal),
                                yaw_rate=yaw_rate, delay=dly if self.map.p.delay_comp < 0 else None)
        speed = torch.minimum(speed, speed_cap)
        sp = self._inner.spec
        # stopped, asking to stay stopped, and touching something: back out, as the tracker does
        stalled = (v_meas.abs() < sp.reverse_v_gate) & (speed < sp.reverse_cmd_gate) & self._inner.contact
        speed = torch.where(stalled, torch.full_like(speed, -sp.v_reverse), speed)
        mine = torch.stack([steer, speed], 1)
        self.last_map = self.map.last
        return torch.where(self.map_rows[:, None], mine, out)
