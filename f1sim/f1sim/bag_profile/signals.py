"""Topic roles, unit checks and the common 100 Hz signal grid a vehicle profile is computed from.

Everything here is decided from the bag itself and reported, never assumed:
  * which topic plays which role (`detect_roles`), and what is missing and what that costs;
  * the servo map servo = g * steer + o (from the mux output `/ackermann_cmd` against `/commands/servo/position`),
    and the window the VESC clamps the servo to (`/commands/servo/position` against
    `/sensors/servo_position_command`, the position the VESC actually applied);
  * the ERPM gain (commanded ERPM / commanded speed) and its cross-check against `/odom` (vesc_to_odom);
  * the gyro's unit (rad/s or deg/s) by regressing it on an independent yaw rate (localisation heading rate, else the
    kinematic rate of the applied steering) -- the "GYRO-BAD" bags are deg/s, not damaged;
  * the accelerometer's unit (g or m/s^2) from gravity, and its axis signs from v*r and dv/dt;
  * biases from standstill.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import bagio as B
from . import thresholds as TH

HZ = 100.0
G = 9.80665
L_WHEELBASE = 0.3302

#: role -> (preferred topic names in order, accepted message type)
ROLES = {
    "servo_cmd": (["/commands/servo/position"], "std_msgs/msg/Float64"),
    "servo_applied": (["/sensors/servo_position_command"], "std_msgs/msg/Float64"),
    "erpm_cmd": (["/commands/motor/speed"], "std_msgs/msg/Float64"),
    "ackermann": (["/ackermann_cmd", "/drive", "/teleop"], "ackermann_msgs/msg/AckermannDriveStamped"),
    "vesc": (["/sensors/core"], "vesc_msgs/msg/VescStateStamped"),
    "imu": (["/sensors/imu/raw", "/imu", "/imu/data"], "sensor_msgs/msg/Imu"),
    "wheel_odom": (["/odom"], "nav_msgs/msg/Odometry"),
    "pose": (["/pf/pose/odom", "/car_state/odom", "/pf/viz/inferred_pose", "/car_state/pose"], None),
    "scan": (["/scan"], "sensor_msgs/msg/LaserScan"),
    "map": (["/map"], "nav_msgs/msg/OccupancyGrid"),
    "tf_static": (["/tf_static"], "tf2_msgs/msg/TFMessage"),
}

#: what a missing role costs -- printed in the report
ROLE_COST = {
    "servo_cmd": "servo map and clamp window cannot be fitted; steering comes from the Ackermann command",
    "servo_applied": "the VESC servo clamp is invisible: replay uses the pre-clamp command (wrong where it clamps)",
    "erpm_cmd": "ERPM gain from /odom only; speed command from the Ackermann topic",
    "ackermann": "servo map cannot be fitted (no steering angle in rad); a stored map is needed",
    "vesc": "no motor current: accel / brake limits and current-limit windows cannot be found; wheel speed from /odom",
    "imu": "no gyro / accelerometer: yaw response, delay, lateral grip and every tyre value are unmeasurable",
    "wheel_odom": "no ERPM cross-check",
    "pose": "no position reference: position errors and the gyro-unit check against heading are unavailable",
    "scan": "no LiDAR statistics, no map from scans",
    "map": "no recorded map: bag_map has to build one from scan + odometry",
    "tf_static": "LiDAR mount position unknown",
}


def detect_roles(bag: B.Bag) -> Tuple[Dict[str, str], Dict[str, str]]:
    roles, missing = {}, {}
    for role, (names, typ) in ROLES.items():
        hit = next((n for n in names if bag.has(n) and (typ is None or bag.topics[n].type == typ)), None)
        if hit is None and typ is not None:
            cands = bag.of_type(typ)
            hit = cands[0] if cands else None
        if hit is None:
            missing[role] = ROLE_COST[role]
        else:
            roles[role] = hit
    return roles, missing


def hold(ts, xs, t):
    i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    return xs[i]


def smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    y = np.convolve(x, k, "same")
    # edge-correct so the ends are means of what exists, not pulled to zero
    c = np.convolve(np.ones_like(x), k, "same")
    return y / c


def deriv(x, n=5, hz=HZ):
    d = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) > 2 * n:
        d[n:-n] = (x[2 * n:] - x[:-2 * n]) * hz / (2 * n)
    return d


def robust_line(x, y, thr=None, iters=3):
    """y = g x + o, least squares with residual trimming (`thr` absolute, else 3 MAD)."""
    keep = np.isfinite(x) & np.isfinite(y)
    g = o = np.nan
    for _ in range(iters):
        if keep.sum() < 5:
            break
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        g, o = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        res = y - (g * x + o)
        t = thr if thr is not None else 3 * 1.4826 * np.median(np.abs(res[keep])) + 1e-9
        keep = np.isfinite(res) & (np.abs(res) < t)
    return float(g), float(o), keep


def pair_after(t_src, t_dst, tol=0.01):
    """index into dst of the first message at/after each src message, within tol seconds; -1 if none."""
    j = np.searchsorted(t_dst, t_src - 1e-6)
    j = np.clip(j, 0, len(t_dst) - 1)
    ok = np.abs(t_dst[j] - t_src) < tol
    return np.where(ok, j, -1)


@dataclass
class Calib:
    """Static conversions read from the bag (all with the evidence they came from)."""
    servo_g: float = np.nan
    servo_o: float = np.nan
    servo_fit_n: int = 0
    servo_fit_inlier: float = np.nan
    servo_fit_resid: float = np.nan
    servo_src: str = ""
    clamp_lo_servo: float = np.nan          # applied servo position range
    clamp_hi_servo: float = np.nan
    clamp_frac: float = np.nan              # fraction of commands the VESC changed
    clamp_seen_lo: bool = False             # the low edge was actually hit by larger commands
    clamp_seen_hi: bool = False
    clamp_n_lo: int = 0                     # commands beyond each edge (an edge needs CLAMP_EDGE_MIN_CMDS of them)
    clamp_n_hi: int = 0
    erpm_gain: float = np.nan
    erpm_gain_odom: float = np.nan
    erpm_pairs: int = 0
    erpm_quantum_mps: float = np.nan
    gyro_scale: float = 1.0                 # multiply raw gz by this to get rad/s
    gyro_unit: str = "rad/s"
    gyro_evidence: dict = field(default_factory=dict)
    accel_scale: float = G
    accel_unit: str = "g"
    gyro_by_magnitude: bool = False         # unit guessed from the size of the signal only
    imu_axes: dict = field(default_factory=dict)
    bias: dict = field(default_factory=dict)
    pose_ok: bool = False
    pose_corr: float = np.nan
    notes: List[str] = field(default_factory=list)
    #: values this bag could not give and that were taken from a car profile / default instead (shown in the summary)
    substitutions: List[str] = field(default_factory=list)

    def steer_window_rad(self) -> Tuple[float, float]:
        a = (self.clamp_lo_servo - self.servo_o) / self.servo_g
        b = (self.clamp_hi_servo - self.servo_o) / self.servo_g
        return float(min(a, b)), float(max(a, b))


def load_raw(bag: B.Bag, roles: Dict[str, str]) -> Dict[str, np.ndarray]:
    names = [roles[r] for r in ("servo_cmd", "servo_applied", "erpm_cmd", "ackermann", "vesc", "imu", "wheel_odom",
                                "pose") if r in roles]
    names += [n for n in ("/drive", "/teleop") if bag.has(n)]
    raw = bag.read(sorted(set(names)))
    return {r: raw[roles[r]] for r in roles if roles[r] in raw} | {n: raw[n] for n in ("/drive", "/teleop") if n in raw}


def fit_servo_map(Z, calib: Calib, fallback=None):
    """servo = g * steer + o. Pairs each Ackermann command with the servo command the converter sent within 10 ms."""
    if "ackermann" not in Z or "servo_cmd" not in Z:
        if fallback:
            calib.servo_g, calib.servo_o = fallback
            calib.servo_src = "fallback (no Ackermann/servo pair in bag)"
            calib.substitutions.append("서보 변환식(이득, 오프셋): 이 bag에서 못 재서 프로필 값 사용 (Ackermann 명령과 서보 명령의 쌍이 없음)")
        return
    a, s = Z["ackermann"], Z["servo_cmd"]
    j = pair_after(a[:, 0], s[:, 0], TH.SERVO_PAIR_TOL_S)
    ok = j >= 0
    x, y = a[ok, 2], s[j[ok], 1]
    if ok.sum() < TH.SERVO_FIT_MIN_PAIRS or np.std(x) < TH.SERVO_FIT_MIN_STEER_STD:
        if fallback:
            calib.servo_g, calib.servo_o = fallback
            calib.servo_src = "fallback (steering spread too small to fit)"
            calib.substitutions.append(
                f"서보 변환식(이득, 오프셋): 이 bag에서 못 재서 프로필 값 사용 (짝 {int(ok.sum())}개/{TH.SERVO_FIT_MIN_PAIRS}, "
                f"조향 표준편차 {float(np.std(x)) if len(x) else 0.0:.3f}/{TH.SERVO_FIT_MIN_STEER_STD} rad)")
        calib.notes.append("servo map: too little steering spread in the bag to fit it")
        return
    g, o, keep = robust_line(x, y, thr=TH.SERVO_FIT_INLIER_TOL)
    calib.servo_g, calib.servo_o = g, o
    calib.servo_fit_n = int(ok.sum())
    calib.servo_fit_inlier = float(keep.mean())
    calib.servo_fit_resid = float(np.std(y[keep] - (g * x[keep] + o))) if keep.any() else np.nan
    calib.servo_src = "fit: " + ("/ackermann_cmd" if True else "") + " vs /commands/servo/position"


def fit_clamp(Z, calib: Calib):
    if "servo_cmd" not in Z or "servo_applied" not in Z:
        calib.notes.append("servo clamp: /sensors/servo_position_command missing -- window unknown")
        return
    c, a = Z["servo_cmd"], Z["servo_applied"]
    j = pair_after(c[:, 0], a[:, 0], TH.CLAMP_PAIR_TOL_S)
    ok = j >= 0
    cmd, app = c[ok, 1], a[j[ok], 1]
    lo, hi = float(np.min(a[:, 1])), float(np.max(a[:, 1]))
    calib.clamp_lo_servo, calib.clamp_hi_servo = lo, hi
    diff = np.abs(cmd - app) > 1e-4
    calib.clamp_frac = float(diff.mean()) if len(diff) else np.nan
    calib.clamp_n_lo = int(np.sum(cmd < lo - TH.CLAMP_EDGE_MARGIN))
    calib.clamp_n_hi = int(np.sum(cmd > hi + TH.CLAMP_EDGE_MARGIN))
    calib.clamp_seen_lo = bool(calib.clamp_n_lo >= TH.CLAMP_EDGE_MIN_CMDS)
    calib.clamp_seen_hi = bool(calib.clamp_n_hi >= TH.CLAMP_EDGE_MIN_CMDS)


def fit_erpm(Z, calib: Calib):
    if "ackermann" in Z and "erpm_cmd" in Z:
        a, e = Z["ackermann"], Z["erpm_cmd"]
        j = pair_after(a[:, 0], e[:, 0], TH.SERVO_PAIR_TOL_S)
        ok = j >= 0
        xs, ys = a[ok, 3], e[j[ok], 1]
        m = np.abs(xs) > TH.ERPM_MIN_SPEED_CMD
        calib.erpm_pairs = int(m.sum())
        if m.sum() > TH.ERPM_MIN_PAIRS:
            calib.erpm_gain = float(np.median(ys[m] / xs[m]))
    if "vesc" in Z and "wheel_odom" in Z:
        v, o = Z["vesc"], Z["wheel_odom"]
        vo = np.interp(v[:, 0], o[:, 0], o[:, 5])
        m = np.abs(vo) > TH.ERPM_ODOM_MIN_V
        if m.sum() > TH.ERPM_ODOM_MIN_SAMPLES:
            calib.erpm_gain_odom = float(np.median(v[m, 2] / vo[m]))
    if "wheel_odom" in Z:
        dv = np.abs(np.diff(Z["wheel_odom"][:, 5]))
        dv = dv[(dv > 1e-7)]
        if len(dv) > 50:
            calib.erpm_quantum_mps = float(np.percentile(dv, 0.5))
    if not np.isfinite(calib.erpm_gain) and np.isfinite(calib.erpm_gain_odom):
        calib.erpm_gain = calib.erpm_gain_odom
        calib.notes.append("ERPM gain taken from /sensors/core vs /odom (no commanded speed pair)")
        calib.substitutions.append(f"ERPM 환산: 지령-회전수 쌍 {calib.erpm_pairs}개(필요 {TH.ERPM_MIN_PAIRS}개 초과)라 /odom 대조값 사용")


def _standstill(v, n=int(TH.STILL_SECONDS * HZ)):
    still = np.abs(v) < TH.STILL_V
    return np.convolve(still.astype(float), np.ones(n), "same") >= n - 0.5


def build_grid(Z, calib: Calib, steer_window: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
    """The 100 Hz grid: steer (applied, rad), steer_cmd (pre-clamp), speed_cmd, v_wheel, i_motor, gyro, accel, pose."""
    im = Z["imu"]
    # the grid spans what IMU and the wheel-speed source cover. Command topics do NOT limit it: a standstill with no
    # command traffic still has an IMU and wheel speed (bias, LiDAR still noise); where no command exists the
    # command-based masks (`cmd_valid`, `cmd_age`) say so.
    t_beg = im[0, 0]
    t_end = im[-1, 0]
    if "vesc" in Z:
        t_beg = max(t_beg, Z["vesc"][0, 0])
        t_end = min(t_end, Z["vesc"][-1, 0])
    t = np.arange(t_beg, t_end, 1 / HZ)
    S: Dict[str, np.ndarray] = {"t": t}
    g, o = calib.servo_g, calib.servo_o
    if "servo_cmd" in Z:
        sc = Z["servo_cmd"]
        S["steer_cmd"] = (hold(sc[:, 0], sc[:, 1], t) - o) / g
        i = np.clip(np.searchsorted(sc[:, 0], t, side="right") - 1, 0, len(sc) - 1)
        S["cmd_age"] = np.where(t < sc[0, 0], np.inf, t - sc[i, 0])
        S["cmd_valid"] = (t >= sc[0, 0]) & (t <= sc[-1, 0] + 0.5)
    else:
        a = Z["ackermann"]
        S["steer_cmd"] = hold(a[:, 0], a[:, 2], t)
        i = np.clip(np.searchsorted(a[:, 0], t, side="right") - 1, 0, len(a) - 1)
        S["cmd_age"] = np.where(t < a[0, 0], np.inf, t - a[i, 0])
        S["cmd_valid"] = (t >= a[0, 0]) & (t <= a[-1, 0] + 0.5)
        calib.substitutions.append("조향: /commands/servo/position 이 없어 Ackermann 명령의 조향각 사용 (서보 변환식 없이)")
    if "servo_applied" in Z:
        sp = Z["servo_applied"]
        S["steer"] = (hold(sp[:, 0], sp[:, 1], t) - o) / g
    else:
        calib.substitutions.append("적용된 조향: /sensors/servo_position_command 가 없어 잘리기 전 명령 사용 (창 근처에서 틀림)")
        S["steer"] = S["steer_cmd"].copy()
        if steer_window is not None:
            S["steer"] = np.clip(S["steer"], *steer_window)
    if "erpm_cmd" in Z and np.isfinite(calib.erpm_gain):
        S["speed_cmd"] = hold(Z["erpm_cmd"][:, 0], Z["erpm_cmd"][:, 1], t) / calib.erpm_gain
    elif "ackermann" in Z:
        S["speed_cmd"] = hold(Z["ackermann"][:, 0], Z["ackermann"][:, 3], t)
    if "vesc" in Z and np.isfinite(calib.erpm_gain):
        c = Z["vesc"]
        S["v_wheel"] = np.interp(t, c[:, 0], c[:, 2]) / calib.erpm_gain
        S["i_motor"] = np.interp(t, c[:, 0], c[:, 3])
        S["duty"] = np.interp(t, c[:, 0], c[:, 4])
    elif "wheel_odom" in Z:
        o_ = Z["wheel_odom"]
        S["v_wheel"] = np.interp(t, o_[:, 0], o_[:, 5])
        S["i_motor"] = np.full_like(t, np.nan)
    S["gx_raw"] = np.interp(t, im[:, 0], im[:, 2])
    S["gy_raw"] = np.interp(t, im[:, 0], im[:, 3])
    S["gz_raw"] = np.interp(t, im[:, 0], im[:, 4])
    for j, k in enumerate(("ax_raw", "ay_raw", "az_raw")):
        S[k] = np.interp(t, im[:, 0], im[:, 5 + j])
    if "pose" in Z:
        p = Z["pose"]
        S["px"] = np.interp(t, p[:, 0], p[:, 2])
        S["py"] = np.interp(t, p[:, 0], p[:, 3])
        S["pyaw"] = np.interp(t, p[:, 0], np.unwrap(p[:, 4]))
    S["still"] = _standstill(S["v_wheel"])
    return S


def decide_units(S, calib: Calib):
    """Accelerometer unit from gravity; gyro unit and IMU axis signs from independent references."""
    still = S["still"]
    amag = np.sqrt(S["ax_raw"] ** 2 + S["ay_raw"] ** 2 + S["az_raw"] ** 2)
    ref = np.median(amag[still]) if still.sum() > TH.STILL_SECONDS * HZ else np.median(amag)
    if 0.5 < ref < 3.0:
        calib.accel_scale, calib.accel_unit = G, "g"
    elif 5.0 < ref < 15.0:
        calib.accel_scale, calib.accel_unit = 1.0, "m/s^2"
    else:
        calib.accel_scale, calib.accel_unit = G, "unknown (|a| median %.2f), assumed g" % ref
        calib.notes.append("accelerometer unit could not be told from gravity")
        calib.substitutions.append(f"가속도 단위: 중력으로 못 가려 g 로 가정 (|a| 중앙값 {ref:.2f})")
    v = S["v_wheel"]
    mv = (v > TH.UNIT_MIN_V) & ~still
    gz = smooth(S["gz_raw"], 10)
    ev = {}
    # (1) localisation heading rate
    cands = []
    if "pyaw" in S:
        pr = deriv(smooth(S["pyaw"], 5), 5)
        ok = mv & np.isfinite(pr)
        if ok.sum() > TH.UNIT_MIN_SAMPLES:
            k = float(np.sum(gz[ok] * pr[ok]) / np.sum(pr[ok] ** 2))
            cc = float(np.corrcoef(gz[ok], pr[ok])[0, 1])
            ev["vs_pose_heading_rate"] = dict(slope=k, corr=cc, n=int(ok.sum()))
            if abs(cc) > 0.6:
                cands.append(("pose", k, cc))
    # (2) kinematic yaw rate of the applied steering (an upper bound on |r|: understeer only lowers it)
    rk = v * np.tan(S["steer"]) / L_WHEELBASE
    ok = mv & (np.abs(rk) > 0.2)
    if ok.sum() > TH.UNIT_MIN_SAMPLES:
        k = float(np.sum(gz[ok] * rk[ok]) / np.sum(rk[ok] ** 2))
        cc = float(np.corrcoef(gz[ok], rk[ok])[0, 1])
        ev["vs_kinematic_rate"] = dict(slope=k, corr=cc, n=int(ok.sum()))
        if abs(cc) > 0.5:
            cands.append(("kinematic", k, cc))
    ev["p99_abs_raw"] = float(np.percentile(np.abs(S["gz_raw"][mv]), 99)) if mv.sum() else float("nan")
    calib.gyro_evidence = ev
    unit = None
    for src, k, cc in cands:
        # rad/s: slope 1 (pose) or 0.3-1.2 (kinematic, understeer lowers it); deg/s: 57x those
        lo, hi = (0.6, 1.6) if src == "pose" else (0.25, 1.6)
        if lo <= abs(k) <= hi:
            unit = unit or "rad/s"
        elif lo * 57.3 * 0.6 <= abs(k) <= hi * 57.3 * 1.4:
            unit = unit or "deg/s"
    if unit is None:
        p99 = ev["p99_abs_raw"]
        unit = "deg/s" if p99 > TH.GYRO_DEG_P99 else "rad/s"
        calib.notes.append(f"gyro unit decided from magnitude only (|gz| p99 {p99:.1f}) -- no reference agreed")
        calib.gyro_by_magnitude = True
        calib.substitutions.append(f"자이로 단위: 기준(위치 추정·조향 운동학)이 맞지 않아 크기만 보고 {unit} 로 추정 (|gz| p99 {p99:.1f})")
    calib.gyro_unit = unit
    calib.gyro_scale = float(np.pi / 180) if unit == "deg/s" else 1.0
    # sign: a left (positive) steering command turns the car left (positive yaw rate) -- physics, not convention
    kin = ev.get("vs_kinematic_rate")
    if kin is not None and abs(kin["corr"]) > 0.5 and kin["n"] > 500 and kin["slope"] < 0:
        calib.gyro_scale = -calib.gyro_scale
        calib.notes.append("gyro z is inverted against the steering (kinematic slope %.2f): sign flipped" % kin["slope"])
    # pose quality: heading rate vs gyro (rad/s) correlation
    if "vs_pose_heading_rate" in ev:
        calib.pose_corr = ev["vs_pose_heading_rate"]["corr"]
        calib.pose_ok = bool(ev["vs_pose_heading_rate"]["corr"] > 0.7)


def finish_grid(S, calib: Calib):
    """Apply units and biases; adds r, ax, ay (body, SI), plus the signs used."""
    s = calib.gyro_scale
    for k in ("gx", "gy", "gz"):
        S[k] = S[k + "_raw"] * s
    for k in ("ax", "ay", "az"):
        S[k] = S[k + "_raw"] * calib.accel_scale
    still = S["still"]
    calib.bias = {}
    if still.sum() > TH.STILL_SECONDS * HZ:
        for k in ("gx", "gy", "gz", "ax", "ay"):
            b = float(np.median(S[k][still]))
            calib.bias[k] = b
            S[k] = S[k] - b
    else:
        calib.notes.append("no standstill of >= 1 s: gyro / accelerometer biases not removed")
    # axis signs: ay against v*r, ax against dv/dt (the VESC IMU orientation is a mounting choice)
    v = S["v_wheel"]
    mv = (v > TH.UNIT_MIN_V) & ~still
    r_s = smooth(S["gz"], 10)
    vr = v * r_s
    ay_s = smooth(S["ay"], 10)
    ax_s = smooth(S["ax"], 10)
    dv = deriv(smooth(v, 10), 5)
    ax_sign = ay_sign = 1.0
    if mv.sum() > TH.UNIT_MIN_SAMPLES:
        c_y = np.corrcoef(ay_s[mv], vr[mv])[0, 1]
        c_x = np.corrcoef(ax_s[mv & np.isfinite(dv)], dv[mv & np.isfinite(dv)])[0, 1]
        cx_y = np.corrcoef(ax_s[mv], vr[mv])[0, 1]
        calib.imu_axes = dict(corr_ay_vr=float(c_y), corr_ax_dvdt=float(c_x), corr_ax_vr=float(cx_y))
        if abs(cx_y) > abs(c_y) + 0.2:
            calib.notes.append("accelerometer x correlates with v*r better than y: axes may be swapped")
        # only flip on clear evidence: a straight-line bag has no lateral signal to judge the sign by
        ay_sign = -1.0 if c_y < -0.3 else 1.0
        ax_sign = -1.0 if c_x < -0.3 else 1.0
    calib.imu_axes.update(ax_sign=ax_sign, ay_sign=ay_sign)
    S["ax"] = S["ax"] * ax_sign
    S["ay"] = S["ay"] * ay_sign
    S["r"] = smooth(S["gz"], 10)
    S["ay_s"] = smooth(S["ay"], 10)
    S["ax_s"] = smooth(S["ax"], 10)
    return S


def load(bag_path: str, fallback_servo=None) -> Tuple[B.Bag, Dict[str, str], Dict[str, str], Calib, Dict]:
    """bag -> (bag, roles, missing, calib, 100 Hz grid). The one call everything else builds on."""
    bag = B.open_bag(bag_path)
    roles, missing = detect_roles(bag)
    Z = load_raw(bag, roles)
    calib = Calib()
    if "imu" not in Z:
        raise RuntimeError(f"{bag.name}: no IMU topic -- nothing dynamic can be measured")
    fit_servo_map(Z, calib, fallback=fallback_servo)
    if not np.isfinite(calib.servo_g):
        # a bag with no steering to fit the map from (a standstill recording) still has IMU, LiDAR and map to give;
        # the steering-derived values come out 'not identified' instead of the whole run failing
        calib.servo_g, calib.servo_o = 1.0, 0.0
        calib.servo_src = ("not identified (no steering spread; pass --car for the stored map)" if "servo_cmd" in Z
                           else "identity (steering from Ackermann topic)")
        calib.notes.append("servo map not identified in this bag: steering values below are not usable")
        calib.substitutions.append("서보 변환식(이득, 오프셋): 이 bag에서 못 재서 항등(1, 0) 사용 -- 조향 값은 쓸 수 없음 (--car 로 프로필 값을 줄 수 있음)")
    fit_clamp(Z, calib)
    fit_erpm(Z, calib)
    S = build_grid(Z, calib)
    decide_units(S, calib)
    finish_grid(S, calib)
    S["_raw"] = Z
    return bag, roles, missing, calib, S
