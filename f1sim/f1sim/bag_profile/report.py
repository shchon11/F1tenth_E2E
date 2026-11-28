"""Orchestration of one bag -> profile, and the writers (YAML records, Korean summary, figures)."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml

from . import (bagio as B, estimators as EST, fit as F, ident as I, measure as MS, models as M, segments as SEG,
               setupfile as SU, signals as SG, thresholds as TH, windows as WN)

#: the lateral structure adopted by the 11-28 re-identification (docs/pm/sysid-2026-11-28/REPORT.md): single track,
#: sin(C atan(B a)) tyres with C fixed at 1.3 (the bags cannot tell C; 1.0-2.2 fit equally), servo lag + delay.
ADOPTED = dict(Cf=1.3, Cr=1.3)
#: what a single race bag pins (REPORT.md section 4); the rest are held at the car profile's value
TYRE_FREE = ["k", "b", "tau_d", "tau_s", "sf", "sr"]
#: starting values when no car profile is given: the competition-car TRAIN fit of (b') (REPORT.md section 3), with the
#: two peak frictions at the plausible end of what the data allow (front: lower bound 1.4; rear: 1.35, inside 1.1-1.7)
TYRE_START = dict(k=1.124, b=-0.0244, tau_d=0.043, tau_s=0.037, sf=2.97, mf=1.4, sr=6.14, mr=1.35)


SOURCE_KO = {"measured": "실측", "estimated": "추정", "spec": "설정값(규격)", "assumed": "가정", "manual": "실측(수동)"}
STATUS_KO = {"measured": "측정됨", "lower_bound": "하한만", "not_identified": "식별 안 됨", "not_applied": "시뮬에 없음"}


def E(value, err=None, unit="", source="measured", status="measured", how="", note="", apply=None, origin=None, **extra):
    """one profile entry. `origin="manual"`: typed in by hand (setup.yaml hand_measured), shown as 실측(수동)."""
    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else round(float(v), 6)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    e = dict(value=clean(value), unit=unit, source=source, status=status)
    e["source_label"] = SOURCE_KO["manual"] if origin == "manual" else SOURCE_KO.get(source, source)
    if origin:
        e["origin"] = origin
    if err is not None:
        e["err"] = clean(err)
    if how:
        e["how"] = how
    if note:
        e["note"] = note
    if apply is not None:
        e["apply"] = bool(apply)
    for k, v in extra.items():
        e[k] = clean(v)
    return e


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def place_code(m: Optional[dict]) -> Dict:
    if not m:
        return dict(code="nomap")
    w_m, h_m = m["w"] * m["res"], m["h"] * m["res"]
    a, b = sorted((w_m, h_m))
    md5 = hashlib.md5(m["data"].tobytes()).hexdigest()[:10]
    return dict(code=f"map{int(round(a)):02d}x{int(round(b)):02d}", size_m=[round(w_m, 2), round(h_m, 2)],
                res=m["res"], grid_md5=md5)


# ------------------------------------------------------------------------------------------------------------------
def run(bag_path: str, out: str, car: Optional[str] = None, name: Optional[str] = None, do_map: bool = True,
        do_lidar: bool = True, do_tyre: bool = True, log: Callable = print, setup: Optional[str] = None) -> Dict:
    t_all = time.time()
    os.makedirs(out, exist_ok=True)
    figs = os.path.join(out, "figs")
    os.makedirs(figs, exist_ok=True)
    setup_path = SU.find(bag_path, setup)
    setup_d = SU.load(setup_path) if setup_path else None
    carprof = None
    if car:
        from ..profiles.loader import load_profile
        carprof = load_profile(car)
    fallback = None
    if carprof and "car" in carprof and "servo_map_gain" in carprof["car"]:
        fallback = (carprof["car"]["servo_map_gain"]["value"], carprof["car"]["servo_map_offset"]["value"])
    bag, roles, missing, calib, S = SG.load(bag_path, fallback_servo=fallback)
    name = name or bag.name[:15]
    log(f"[bag] {bag.name}: {bag.duration:.0f} s, 토픽 {len(bag.topics)}개, 역할 {len(roles)}/{len(SG.ROLES)}")
    for r, why in missing.items():
        log(f"  [없음] {r}: {why}")
    # ---- test segments marked during recording (FIELD_PROTOCOL.md: /sysid/segment std_msgs/String) ----
    ints, seg_events, seg_warn = {}, [], []
    seg_topic = next((n for n in ("/sysid/segment",) if bag.has(n)), None)
    if seg_topic:
        seg_events = sorted((t_, B.dec_string(b)) for _, t_, b in bag.iter_raw([seg_topic]))
        ints, seg_warn = SEG.parse(seg_events, bag.duration)
    methods: Dict[str, str] = {}

    def sel_for(tests, label):
        """boolean grid mask of the marked segments of `tests`, or None (= the tool finds stretches itself)"""
        m = SEG.mask(S["t"], ints, tests) if ints else None
        if m is not None:
            methods[label] = f"표시 토픽 ({'+'.join(tests)}: {SEG.describe(ints, tests)})"
        else:
            methods[label] = "자동 인식 (bag 전체에서 규칙으로 찾음)" + (f"; 표시에 {'/'.join(tests)} 없음" if ints else "")
        return m
    # servo window ends: only what the T1 segments show, when they are marked
    m_t1 = sel_for(("T1",), "서보 창 양 끝")
    if m_t1 is not None:
        Z = S["_raw"]
        tt = lambda a: np.zeros(len(a), bool) if len(a) == 0 else np.any([(a[:, 0] >= x0) & (a[:, 0] <= x1) for x0, x1 in ints["T1"]], 0)
        if "servo_cmd" in Z and "servo_applied" in Z:
            Zr = {"servo_cmd": Z["servo_cmd"][tt(Z["servo_cmd"])], "servo_applied": Z["servo_applied"][tt(Z["servo_applied"])]}
            if len(Zr["servo_cmd"]) and len(Zr["servo_applied"]):
                SG.fit_clamp(Zr, calib)
    R: Dict = dict(bag=bag.path, name=name, duration_s=bag.duration, roles=roles, missing=missing,
                   counts={n: bag.topics[n].count for n in roles.values()}, calib=dict(calib.__dict__))
    R["bag_files"] = len(bag.dbs)
    if len(bag.dbs) > 1:
        R["note_split"] = f"쪼개진 bag: .db3 {len(bag.dbs)}개를 전부 읽음"
    R["segments"] = [dict(t=round(t_, 2), label=lab) for t_, lab in seg_events]
    R["segment_intervals"] = {k: [[round(a, 2), round(b, 2)] for a, b in v] for k, v in ints.items()}
    R["segment_warnings"] = seg_warn
    R["segment_mode"] = "표시 토픽 /sysid/segment" if ints else "자동 인식 (표시 토픽 없음)"
    if seg_events:
        log(f"[구간 표시] {len(seg_events)}개: " + ", ".join(f"{t_:.1f}s {lab}" for t_, lab in seg_events[:12]))
    for w_ in seg_warn:
        log(f"  [구간 표시 경고] {w_}")
    log(f"[구간] 방식: {R['segment_mode']}")
    # ---- direct measurements ----
    t = time.time()
    sel_t3, sel_t6 = sel_for(("T3",), "가속·제동 한계, 전류 한계"), sel_for(("T6",), "그립 하한, 미끄러진 사건")
    sel_t2, sel_t4, sel_t5 = sel_for(("T2",), "저속 정상 선회"), sel_for(("T4",), "코스트다운"), sel_for(("T5",), "조향 응답(처프)")
    methods["지연·트림·재생 맞춤·IMU 바이어스·LiDAR"] = "bag 전체 (구간 표시와 무관)"
    R["methods"] = methods
    R["delay_xcorr"] = MS.delay_xcorr(S)
    R["accel"] = MS.accel_brake(S, sel_t3)
    R["lat"] = MS.lateral_limit(S, sel=sel_t6)
    R["low_speed_turn"] = MS.low_speed_turn(S, sel_t2)
    R["coastdown"] = EST.coastdown(S, sel_t4)
    R["steer_response"] = EST.steer_response(S, sel_t5)
    R["standstill_s"] = float(S["still"].sum() / SG.HZ)
    log(f"[식별] 코스트다운 {R['coastdown']['status']}, 조향 응답 {R['steer_response']['status']} (실차 미검증)")
    W = WN.windows_from_grid(S, name, calib.pose_ok)
    R["windows"] = int(len(W["r"]))
    R["impact_excluded"] = int(W["n_impact_excluded"])
    log(f"[측정] 지연·가속·횡가속 | 재생 구간 {len(W['r'])}개 (충돌로 뺀 것 {W['n_impact_excluded']}) | {time.time() - t:.1f} s")
    # ---- replay fits: speed loop, (a), (b) ----
    if len(W["r"]) >= TH.REPLAY_MIN_WINDOWS:
        t = time.time()
        # delays on grids (the models see them in whole samples), the rest by least squares
        sp, Jv, _ = F.fit_grid(M.speed_rollout, W, dict(M.SPEED_DEFAULT), ["tau_m", "a_a", "a_b"], "tau_v",
                               np.arange(0.0, 0.205, 0.02), target="v", max_nfev=40)
        R["speed_fit"] = dict(p={k: sp[k] for k in M.SPEED_DEFAULT}, rmse=float(np.sqrt(Jv)))
        pa, Ja, lin_curve = F.fit_grid(M.lin_rollout, W, {**sp, **M.LIN_DEFAULT, "k": 1.05, "b": -0.02, "kus": 0.015},
                                       ["k", "b", "kus"], "tau", np.arange(0.0, 0.255, 0.01), max_nfev=40)
        R["lin_fit"] = dict(p={k: pa[k] for k in ("k", "tau", "b", "kus")}, rmse=float(np.sqrt(Ja)), tau_curve=lin_curve,
                            gain_5mps=float(pa["k"] * M.L / (M.L + pa["kus"] * 25.0)))
        lin_blocks = F.blocks(W)
        R["lin_fit"]["blocks"] = int(len(np.unique(lin_blocks)))
        log(f"[재생] 속도 루프 τ {sp['tau_m']:.2f} s, 지연 {sp['tau_v']:.2f} s | 선형 자전거 k {pa['k']:.2f} τ {pa['tau']:.3f} "
            f"b {pa['b']:+.3f} K {pa['kus']:.3f} (1 s 요레이트 RMS {np.sqrt(Ja):.3f}) | {time.time() - t:.1f} s")
        R["trim"] = MS.trim(S, pa["k"])
        if do_tyre:
            t = time.time()
            base = dict(M.ST_DEFAULT, **sp, **TYRE_START, **ADOPTED)
            if carprof and "model_b" in carprof:
                base.update({k: v["value"] for k, v in carprof["model_b"].items() if k in base})
            # several starts in the pure delay: the two car setups differ by ~60 ms and a local fit started at the
            # other car's value can trade the missing delay into the servo lag instead
            pb, Jb = None, None
            for td0 in (0.03, 0.06, 0.09, 0.12):
                q, Jq, _ = F.fit(M.st_rollout, W, dict(base, tau_d=td0), TYRE_FREE, max_nfev=80)
                if Jb is None or Jq < Jb:
                    pb, Jb = q, Jq
            R["tyre_fit"] = dict(p={k: pb[k] for k in TYRE_FREE + ["mf", "mr", "Cf", "Cr"]}, rmse=float(np.sqrt(Jb)))
            # grip: shared mu profile (lower bound expected)
            def st_shared(W_, q, idx=None, steer_key="steer", hor=200):
                q = dict(q); q["mf"] = q["mr"] = q["mu"]
                return M.st_rollout(W_, q, idx=idx, steer_key=steer_key, hor=hor)
            F.BOUNDS.setdefault("mu", (0.3, 3.0))
            grid = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2]
            prof = I.profile_verdict(st_shared, W, dict(pb, mu=max(pb["mf"], pb["mr"])), TYRE_FREE + ["mu"], "mu",
                                     grid, max_nfev=25)
            R["mu_profile"] = prof
            # stiffness profiles: are sf, sr pinned by this bag?
            R["sf_profile"] = I.profile_verdict(M.st_rollout, W, pb, TYRE_FREE, "sf",
                                                [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0], max_nfev=25)
            R["sr_profile"] = I.profile_verdict(M.st_rollout, W, pb, TYRE_FREE, "sr",
                                                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 30.0], max_nfev=25)
            pr_b = M.st_rollout(W, pb)
            pr_a = M.lin_rollout(W, pa)
            R["fit_errors"] = {"(a)": F.summarize(W, pr_a), "(b)": F.summarize(W, pr_b)}
            # the car profile's own prediction on this bag: does the profile describe this bag?
            if carprof and "model_b" in carprof:
                pc = dict(base)
                pr_c = M.st_rollout(W, pc)
                R["fit_errors"]["car_profile"] = F.summarize(W, pr_c)
            R["_pred"] = dict(a=pr_a, b=pr_b)
            log(f"[타이어] 단일트랙 k {pb['k']:.2f} b {pb['b']:+.3f} τd {pb['tau_d']:.3f} τs {pb['tau_s']:.3f} "
                f"앞 {pb['sf']:.2f} 뒤 {pb['sr']:.2f} /rad (1 s RMS {np.sqrt(Jb):.3f}); 마찰 {prof['verdict']} "
                f"{prof['interval']} | {time.time() - t:.1f} s")
    else:
        R["trim"] = MS.trim(S, 1.0)
        log("[재생] 재생 구간이 30개 미만이라 응답 맞춤을 건너뜀")
    # ---- static topics: tf, map ----
    tfm = bag.read_one(roles["tf_static"], B.dec_tf) if "tf_static" in roles else None
    R["tf_static"] = tfm[1] if tfm else []
    mp = bag.read_one(roles["map"], B.dec_map, which="last") if "map" in roles else None
    R["place"] = place_code(mp[1] if mp else None)
    if mp:
        R["place"]["n_map_msgs"] = bag.topics[roles["map"]].count
    # ---- LiDAR ----
    if do_lidar and "scan" in roles:
        t = time.time()
        sc = bag.read_scans(roles["scan"], every=2)
        if sc is not None:
            R["lidar"] = MS.lidar_stats(sc[0], sc[3], sc[2], S)
            R["_lidar_meta"] = sc[2]
            del sc
            log(f"[LiDAR] 유효 빔 {R['lidar']['valid_frac']:.3f}, 빔 {R['lidar']['beams']}개, "
                f"{R['lidar']['rate_hz']:.1f} Hz (2장에 1장 읽음) | {time.time() - t:.1f} s")
    # ---- map via bag_map ----
    if do_map:
        R["bag_map"] = run_bag_map(bag.path, out, name, log)
    # ---- car-profile comparison ----
    R["setup_check"] = setup_check(R, carprof)
    R["setup_file"] = setup_path
    if setup_d is not None:
        R["setup_compare"] = SU.compare(setup_d, R)
        R["setup_manual"] = SU.manual_entries(setup_d)
        R["setup_ignored"] = setup_d.get("_ignored", [])
        n_bad = sum(1 for r in R["setup_compare"] if r["ok"] is False)
        log(f"[setup.yaml] {setup_path}: {len(R['setup_compare'])}항목 대조, 불일치 {n_bad}개")
    else:
        R["setup_compare"], R["setup_manual"], R["setup_ignored"] = [], {}, []
        log("[setup.yaml] 없음 (bag 폴더 안이나 옆에 두거나 --setup 으로 지정): 설정 대조·수동 실측 입력 없음")
    R["seconds"] = round(time.time() - t_all, 1)
    write_outputs(R, out, S, W, carprof, log)
    R["out"] = out
    return R


def run_bag_map(bag_dir: str, out: str, name: str, log) -> Dict:
    mdir = os.path.join(out, "map")
    cmd = [sys.executable, "-m", "f1sim.bag_map", bag_dir, "--name", name, "--out", mdir]
    try:
        import rosbag2_py  # noqa: F401
    except Exception as e:
        log(f"[지도] rosbag2_py 를 불러올 수 없어 건너뜀 ({e.__class__.__name__}); bash 에서 "
            f"`cd ~ && source ~/f1tenth_ws/activate.sh && {' '.join(cmd)}`")
        return dict(skipped=True, why=f"rosbag2_py unavailable: {e}", cmd=" ".join(cmd))
    t = time.time()
    env = dict(os.environ)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = here + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    js = os.path.join(mdir, f"{name}_bag.json")
    if p.returncode != 0 or not os.path.exists(js):
        log(f"[지도] bag_map 실패 (코드 {p.returncode}): {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''}")
        return dict(skipped=True, why="bag_map failed", returncode=p.returncode, stderr=p.stderr[-2000:], cmd=" ".join(cmd))
    rec = json.load(open(js))
    sc = rec.get("bagmap_scale", {})
    log(f"[지도] 출처 {rec.get('source')}, 축척 배율 {sc.get('scale', float('nan')):.3f} -> {rec.get('yaml')} | {time.time() - t:.0f} s")
    return dict(skipped=False, source=rec.get("source"), yaml=rec.get("yaml"), scale=sc.get("scale"),
                scale_iqr=sc.get("scale_iqr"), scans_used=sc.get("scans_used"), json=js, seconds=round(time.time() - t, 1))


# ------------------------------------------------------------------------------------------------------------------
FINGERPRINT = [  # (label, getter, tolerance) -- values that tell two car setups apart
    ("servo map gain", lambda R: R["calib"]["servo_g"], TH.SETUP_TOL_SERVO_GAIN),
    ("servo map offset", lambda R: R["calib"]["servo_o"], TH.SETUP_TOL_SERVO_OFFSET),
    ("ERPM gain", lambda R: R["calib"]["erpm_gain"], TH.SETUP_TOL_ERPM),
    ("gyro unit", lambda R: R["calib"]["gyro_unit"], None),
    ("regen current min [A]", lambda R: R["accel"].get("current_min_A"), TH.FP_TOL_REGEN_A),
    ("total delay (a) [s]", lambda R: R.get("lin_fit", {}).get("p", {}).get("tau"), TH.FP_TOL_DELAY_S),
    ("LiDAR mount x [m, base_link]", lambda R: next((t[2] for t in R["tf_static"] if t[1] in ("laser", "lidar")), None), TH.SETUP_TOL_TF_M),
]


def fingerprint(R) -> Dict:
    out = {}
    for label, g, tol in FINGERPRINT:
        try:
            out[label] = g(R)
        except Exception:
            out[label] = None
    return out


def setup_check(R, carprof) -> List[Dict]:
    if not carprof or "fingerprint" not in carprof:
        return []
    fp = fingerprint(R)
    rows = []
    for label, g, tol in FINGERPRINT:
        ref = carprof["fingerprint"].get(label)
        v = fp.get(label)
        if ref is None or v is None:
            continue
        ok = (v == ref) if tol is None else (abs(float(v) - float(ref)) <= tol)
        rows.append(dict(item=label, bag=v, profile=ref, tol=tol, ok=bool(ok)))
    return rows


# ------------------------------------------------------------------------------------------------------------------
def car_record(R) -> Dict:
    c = R["calib"]
    lo, hi = SG.Calib(**{k: v for k, v in c.items() if k in SG.Calib.__dataclass_fields__}).steer_window_rad() \
        if np.isfinite(c["servo_g"]) and np.isfinite(c.get("clamp_lo_servo", np.nan)) else (np.nan, np.nan)
    rec: Dict = {"_meta": dict(kind="car_calibration", bag=R["bag"], made=datetime.datetime.now().isoformat(timespec="seconds"),
                               tool="python -m f1sim.bag_profile", place=R["place"].get("code"),
                               fingerprint=_jsonable(fingerprint(R)))}
    car: Dict = {}
    servo_fit = "fit" in c["servo_src"]
    sub_note = "" if servo_fit else "이 bag에서 못 재서 프로필 값(또는 항등) 사용"
    car["servo_map_gain"] = E(c["servo_g"], unit="servo/rad", source="measured" if servo_fit else "assumed",
                              status="measured" if servo_fit else "not_identified",
                              how=c["servo_src"], note=sub_note, inlier_frac=c["servo_fit_inlier"], resid=c["servo_fit_resid"])
    car["servo_map_offset"] = E(c["servo_o"], unit="servo", source="measured" if servo_fit else "assumed",
                                status="measured" if servo_fit else "not_identified", how=c["servo_src"], note=sub_note)
    both = c["clamp_seen_lo"] and c["clamp_seen_hi"]
    car["steer_window"] = E([lo, hi], unit="rad (command)", source="measured",
                            status="measured" if both else "lower_bound", apply=False,
                            how="range of /sensors/servo_position_command through the servo map; an edge is measured "
                                "when larger commands were cut to it",
                            note=("양쪽 끝 모두 잘림이 관측됨" if both else "한쪽 또는 양쪽 끝에서 잘림이 관측되지 않음: 창은 적어도 이만큼") +
                                 " -- 시뮬 s_max 는 좌우 대칭이라 그대로 넣을 수 없음",
                            clamped_frac=c["clamp_frac"], edge_hit=[c["clamp_seen_lo"], c["clamp_seen_hi"]])
    car["erpm_gain"] = E(c["erpm_gain"], unit="ERPM/(m/s)", source="measured", status="measured",
                         how="/commands/motor/speed / Ackermann speed (median)", cross_check_odom=c["erpm_gain_odom"])
    car["gyro_unit"] = E(c["gyro_unit"], source="assumed" if c.get("gyro_by_magnitude") else "measured",
                         status="not_identified" if c.get("gyro_by_magnitude") else "measured",
                         how=("크기만 보고 추정 (위치 추정·조향 운동학 기준이 맞지 않음)" if c.get("gyro_by_magnitude") else
                              "gyro z regressed on the localisation heading rate and on v tan(steer)/L"),
                         evidence=_jsonable(c["gyro_evidence"]))
    acc_assumed = str(c["accel_unit"]).startswith("unknown")
    car["accel_unit"] = E(c["accel_unit"], source="assumed" if acc_assumed else "measured",
                          status="not_identified" if acc_assumed else "measured", how="|a| at standstill")
    car["imu_axes"] = E(_jsonable(c["imu_axes"]), source="measured", status="measured",
                        how="sign of a_y against v*r and of a_x against dv/dt")
    car["imu_bias"] = E(_jsonable(c["bias"]), unit="rad/s, m/s^2", source="measured",
                        status="measured" if c["bias"] else "not_identified", how="median over standstill >= 1 s")
    dx = R["delay_xcorr"]
    car["delay_cmd_to_yaw_xcorr"] = E(dx.get("value"), err=dx.get("ci"), unit="s", source="measured",
                                      status="measured" if np.isfinite(dx.get("value", np.nan)) else "not_identified",
                                      how=dx.get("how", ""), note="지연+서보+타이어+요 관성을 합친 겉보기 지연")
    tr = R["trim"]
    car["straight_cmd"] = E(tr.get("value"), err=tr.get("ci"), unit="rad", source="measured",
                            status="measured" if np.isfinite(tr.get("value", np.nan)) else "not_identified", how=tr.get("how", ""))
    acc = R["accel"]
    for key, fk, ext in (("current_max", "current_max_fit", "current_max_A"), ("current_min", "current_min_fit", "current_min_A")):
        fit = acc.get(fk) or {}
        meas = fit.get("status") == "measured"
        car[key] = E(fit.get("value", acc.get(ext)), err=fit.get("err"), unit="A", source="measured",
                     status="measured" if meas else "lower_bound",
                     how=("전류가 한계에 붙어 평평하게 유지된 구간이 확인됨: " if meas else "하한만 (한계에 평평하게 붙은 구간이 확인되지 않음): ")
                         + fit.get("how", ""),
                     plateau_runs=fit.get("runs"), plateau_seconds=fit.get("seconds"),
                     criterion=(f"극값의 {100 * TH.CUR_PLATEAU_BAND:.0f} % 안, 연속 {TH.CUR_PLATEAU_MIN_S} s 이상인 구간이 "
                                f"{TH.CUR_PLATEAU_SEP_S} s 이상 떨어져 {TH.CUR_PLATEAU_MIN_RUNS}개 이상"))
    lf = R.get("lin_fit")
    if lf:
        car["lin_model"] = E(_jsonable(lf["p"]), unit="k [-], tau [s], b [rad], kus [s^2/m]", source="estimated",
                             status="measured", how="(a) r = v tan(k d(t-tau)+b)/(L+K v^2), 1 s yaw-rate fit",
                             rmse_1s=lf["rmse"], gain_5mps=lf["gain_5mps"],
                             note="k 와 K 는 서로 맞바뀜: 5 m/s 대표 이득(gain_5mps)과 tau, b 만 재현됨")
    sp = R.get("speed_fit")
    if sp:
        car["speed_loop"] = E(_jsonable(sp["p"]), unit="s, m/s^2", source="estimated", status="measured",
                              how="first-order loop on the delayed command, accel clipped [-a_b, a_a]", rmse_1s=sp["rmse"])
    rec["car"] = car
    # ---- Config-mappable entries (loadable by f1sim.profiles.loader) ----
    act, veh, lid, odo = {}, {}, {}, {}
    if np.isfinite(acc.get("a_brake", {}).get("value", np.nan)):
        ab = acc["a_brake"]
        veh["a_brake"] = E(ab["value"], err=ab.get("ci"), unit="m/s^2", source="measured", status="measured",
                           how=ab["how"], seconds=ab["seconds"])
    else:
        veh["a_brake"] = E(None, unit="m/s^2", source="measured", status="not_identified", apply=False,
                           how="no regen-limited braking in the bag")
    fitv = acc.get("a_max_vswitch_fit")
    if fitv:
        tab = acc["a_max"]["by_speed"]
        veh["a_max"] = E(fitv["a_max"], unit="m/s^2", source="estimated", status="measured" if len(tab) >= 3 else "lower_bound",
                         how="a_max * min(1, v_switch / v) fitted to the current-limited table", by_speed=_jsonable(tab))
        veh["v_switch"] = E(fitv["v_switch"], unit="m/s", source="estimated",
                            status="measured" if len(tab) >= 3 else "not_identified", apply=len(tab) >= 3, how="same fit")
    if lf:
        act["steer_bias"] = E(lf["p"]["b"], unit="rad", source="estimated", status="measured",
                              how="(a) trim; the simulator adds it after its own steer_gain")
    tyre = R.get("tyre_fit")
    if tyre:
        p = tyre["p"]
        sepr = R.get("steer_response", {}).get("status") == "separated"
        tau_st = "measured" if sepr else "not_identified"
        act["cmd_delay"] = E(p["tau_d"], unit="s", source="estimated", status=tau_st, apply=False,
                             how="(b) pure delay; pair with servo_tau", note="(b) 구조의 값; 시뮬 구조에는 차 프로필 값을 쓸 것"
                             + ("" if sepr else ". 합 tau_d + tau_s 만 정해짐: 분리 식별 안 됨 (조향 처프 T5 필요)"))
        act["servo_tau"] = E(p["tau_s"], unit="s", source="estimated", status=tau_st, apply=False,
                             how="(b) servo first-order lag", note="tau_d 와 합만 잘 정해짐" if not sepr else "")
        rec["model_b"] = {k: E(v, source="estimated", status="measured" if k in TYRE_FREE else "not_identified",
                               how="(b) single-track fit on this bag" if k in TYRE_FREE else "held at the start value")
                          for k, v in p.items()}
    if c.get("erpm_quantum_mps") and np.isfinite(c["erpm_quantum_mps"]):
        odo["erpm_quantum"] = E(c["erpm_quantum_mps"], unit="m/s", source="measured", status="measured",
                                how="p0.5 of nonzero |dv| on /odom (one ERPM step)")
    for t in R["tf_static"]:
        if t[1] in ("laser", "lidar", "laser_frame"):
            lid["mount_x"] = E(t[2] - M.LR, unit="m (ahead of CoG)", source="spec", status="not_identified",
                               how=f"/tf_static {t[0]}->{t[1]} x {t[2]:.3f} minus lr {M.LR} (base_link = rear axle)",
                               note="차에 설정된 값을 읽은 것이지 잰 것이 아님. 실측은 setup.yaml hand_measured.lidar_mount_m")
            lid["mount_z"] = E(t[4], unit="m", source="spec", status="not_identified", how="/tf_static z",
                               note="설정값을 읽은 것 (실측 아님)")
    li = R.get("lidar")
    if li:
        sig = [s for s in li.get("still_noise", []) if s["r"][1] <= 4]
        if sig:
            lid["noise_std"] = E(float(np.median([s["sigma_mm"] for s in sig])) / 1000, unit="m", source="measured",
                                 status="measured", how="1.4826 MAD per beam over standstill scans, 0-4 m")
    # ---- coast-down and steering response (new; NOT validated on the real car) ----
    cd = R.get("coastdown")
    if cd:
        cm = cd["status"] == "measured"
        ci0, ci2 = cd.get("c0_ci"), cd.get("c2_ci")
        car["coast_c0"] = E(cd["c0"] if cm else None, err=ci0 if cm else None, unit="m/s^2", source="estimated",
                            status="measured" if cm else "not_identified", how=cd["how"], note=EST.NOT_VALIDATED,
                            seconds=cd["seconds"], segments=cd["segments"], need=_jsonable(cd["need"]))
        car["coast_c2"] = E(cd["c2"] if cm else None, err=ci2 if cm else None, unit="1/m", source="estimated",
                            status="measured" if cm else "not_identified", how=cd["how"], note=EST.NOT_VALIDATED)
        veh["c_roll"] = E(cd["c0"] if cm else None, err=ci0 if cm else None, unit="m/s^2", source="estimated",
                          status="measured" if cm else "not_identified", apply=False, how="coast_c0 (코스트다운)",
                          note="실차 미검증 추정기: 확인 뒤 apply 로 바꿀 것")
        veh["c_drag"] = E(cd["c2"] if cm else None, err=ci2 if cm else None, unit="1/m", source="estimated",
                          status="measured" if cm else "not_identified", apply=False, how="coast_c2 (코스트다운)",
                          note="실차 미검증 추정기: 확인 뒤 apply 로 바꿀 것")
    sr = R.get("steer_response")
    if sr:
        st = sr["status"]
        sep, sm = st == "separated", st in ("separated", "sum_only")
        car["steer_tau_d"] = E(sr.get("tau_d") if sep else None, err=sr.get("tau_d_ci") if sep else None, unit="s", source="estimated",
                               status="measured" if sep else "not_identified", how=sr["how"], note=EST.NOT_VALIDATED,
                               separation=st)
        car["steer_tau_s"] = E(sr.get("tau_s") if sep else None, err=sr.get("tau_s_ci") if sep else None, unit="s", source="estimated",
                               status="measured" if sep else "not_identified", how=sr["how"], note=EST.NOT_VALIDATED,
                               separation=st)
        car["steer_tau_sum"] = E(sr.get("tau_sum") if sm else None, err=sr.get("tau_sum_ci") if sm else None, unit="s",
                                 source="estimated", status="measured" if sm else "not_identified", how=sr["how"],
                                 note=EST.NOT_VALIDATED + (": 순수 지연과 서보 지연은 분리 식별 안 됨" if st == "sum_only" else ""))
    # ---- values typed in by hand (setup.yaml hand_measured): source 실측(수동) ----
    for grp, d in (R.get("setup_manual") or {}).items():
        tgt = {"car": car, "vehicle": veh, "lidar": lid}[grp]
        for k, kw in d.items():
            tgt[k] = E(kw["value"], err=kw.get("err"), unit=kw.get("unit", ""), source="measured", status="measured",
                       origin="manual", how=kw.get("how", ""))
    rec["vehicle"], rec["actuator"], rec["lidar"], rec["odom"] = veh, act, lid, odo
    return rec


def place_record(R) -> Dict:
    lat = R["lat"]
    rec: Dict = {"_meta": dict(kind="place_profile", bag=R["bag"], made=datetime.datetime.now().isoformat(timespec="seconds"),
                               place=_jsonable(R["place"]))}
    place: Dict = {}
    place["ay_max_held"] = E(lat.get("value"), unit="m/s^2", source="measured", status="lower_bound", how=lat.get("how", ""),
                             at_t=lat.get("t"), at_v=lat.get("v_at"), top10=lat.get("top10"),
                             note="실차가 이 횡가속을 잃지 않고 버팀: 그립의 하한")
    place["mu_lower_from_ay"] = E(lat.get("mu_lower"), source="measured", status="lower_bound",
                                  how="ay_max_held / g (no load transfer, no aero)")
    place["slides_seen"] = E(lat.get("slides_t", []), unit="s", source="measured", status="measured",
                             how="body-slip change > 30 deg within 1 s at v > 2 m/s (from a_y / v - r)",
                             note="미끄러진 사건이 있어야 그립의 상한이 생김")
    mp = R.get("mu_profile")
    if mp:
        st = mp["verdict"]
        place["mu_tyre_fit"] = E(mp["interval"][0] if st == "lower_bound" else mp["best"], err=mp["interval"], source="estimated",
                                 status="lower_bound" if st in ("lower_bound", "not_identified") else "measured",
                                 how="profile of the shared peak friction of the (b) single-track model, block-paired test",
                                 verdict=st)
    tyre = R.get("tyre_fit")
    if tyre:
        for k, prof in (("sf", R.get("sf_profile")), ("sr", R.get("sr_profile"))):
            v = prof["verdict"] if prof else "not_identified"
            place[f"stiffness_{k}"] = E(tyre["p"][k], err=prof["interval"] if prof else None, unit="1/rad per unit load",
                                        source="estimated", status="measured" if v == "measured" else
                                        ("lower_bound" if v == "lower_bound" else "not_identified"),
                                        how="(b) per-axle slope of F/Fz, profile with block-paired test", verdict=v,
                                        note="타이어와 노면의 조합 + 조향 컴플라이언스(앞)")
    li = R.get("lidar")
    if li:
        place["lidar_dropout_by_range"] = E(_jsonable(li["dropout_by_range"]), source="measured", status="measured",
                                            how="short gaps (<= 5 beams) between agreeing returns, binned by their range",
                                            note="시뮬은 거리와 무관한 누락만 있음 (적용 안 함)", apply=False)
        place["lidar_valid_frac"] = E(li["valid_frac"], source="measured", status="measured", how="all beams, all scans read")
    bm = R.get("bag_map")
    if bm and not bm.get("skipped"):
        place["map"] = E(bm.get("yaml"), source="measured", status="measured", how=f"bag_map source={bm.get('source')}",
                         scale=bm.get("scale"), scale_iqr=bm.get("scale_iqr"),
                         note="축척 검사: 1 에서 3 % 넘게 벗어나면 bag /map 대신 스캔 지도")
    elif bm:
        place["map"] = E(None, source="measured", status="not_identified", how=bm.get("why", ""), cmd=bm.get("cmd"))
    rec["place"] = place
    return rec


# ------------------------------------------------------------------------------------------------------------------
TESTS = {
    "standstill": ("정지 10 s (bag 앞뒤)", "IMU 바이어스, 자이로 단위 확인, LiDAR 정지 잡음"),
    "low_speed_circle": ("저속(1~1.5 m/s) 정상 원선회, 조향 3~4단계, 좌우", "조향 이득과 언더스티어를 떼어 냄 (k·K 맞바뀜 해소)"),
    "steer_edges": ("정지 상태 좌우 최대 조향 명령 + 바퀴 각 측정", "서보 창 양 끝, 명령->바퀴 각 이득"),
    "grip_ramp": ("일정 반경 속도 램프, 미끄러질 때까지 (좌우)", "최대 마찰(상한), 한계에서 앞/뒤 어느 쪽이 먼저 빠지는지"),
    "launch_brake": ("직선 전개 가속 -> 최대 제동 x3~4", "a_max(v), v_switch, a_brake, 회생 전류 한계"),
    "coastdown": ("코스트다운 6 m/s -> 정지 x2", "구름·공기 저항 (c_roll, c_drag)"),
    "steer_chirp": ("조향 스텝·처프 (2~3개 속도, 0.5~4 Hz)", "순수 지연과 서보 지연을 따로, 요 동특성"),
    "scale": ("저울(바퀴별), 진자 또는 이중 줄 매달기", "질량, 무게중심 앞뒤 위치, 요 관성 (bag 으로는 불가)"),
}


def missing_list(R, S) -> Dict[str, List[Dict]]:
    """What this bag did not give, in two lists.

    fixable: this KIND of recording can measure it but the conditions fell short -- filming the test again fixes it;
             every entry says what was missing and by how much (numbers).
    not_from_driving: a driving recording never contains it (wheel angles, mass, inertia, centre of gravity, a
             measured LiDAR position, which axle lets go first) -- it comes in through setup.yaml `hand_measured:`.
    """
    fix: List[Dict] = []
    c = R["calib"]
    acc = R["accel"]
    lat = R["lat"]
    hm = (R.get("setup_manual") or {})

    def add(item, why, test):
        fix.append(dict(item=item, why=why, test=test))
    if not c["bias"]:
        add("IMU 바이어스", f"정지 구간 {R.get('standstill_s', 0.0):.1f} s (연속 {TH.STILL_SECONDS} s 이상 필요)", "standstill")
    li = R.get("lidar")
    if li is not None and not li.get("still_noise"):
        add("LiDAR 정지 잡음", f"거의 정지한 스캔 {li.get('still_seconds', 0.0):.1f} s (바퀴 속도 < {TH.LIDAR_STILL_V} m/s, "
                              f"자이로 < {TH.LIDAR_STILL_GYRO} rad/s 로 {TH.LIDAR_STILL_BLOCK_S} s 블록이 필요)", "standstill")
    if not lat.get("slides_t"):
        add("최대 마찰 (상한)", f"미끄러진 사건 0회 (1 s 안 차체 미끄럼 변화 > {TH.SLIDE_DEG:.0f}°, v > {TH.SLIDE_MIN_V} m/s 가 필요): "
                                f"하한 {lat.get('value', float('nan')):.1f} m/s² 만 있음", "grip_ramp")
    lt = R.get("low_speed_turn", {})
    if lt.get("seconds", 0.0) < TH.T2_MIN_SECONDS:
        add("조향 이득 k 와 언더스티어 K 의 분리", f"저속 정상 선회 {lt.get('seconds', 0.0):.1f} s ({TH.T2_MIN_SECONDS:.0f} s 필요)", "low_speed_circle")
    sides = []
    if not c["clamp_seen_lo"]:
        sides.append(f"낮은 쪽 끝 밖 명령 {c.get('clamp_n_lo', 0)}개")
    if not c["clamp_seen_hi"]:
        sides.append(f"높은 쪽 끝 밖 명령 {c.get('clamp_n_hi', 0)}개")
    if sides:
        add("서보 창 양 끝", ", ".join(sides) + f" (끝마다 {TH.CLAMP_EDGE_MIN_CMDS}개 필요)", "steer_edges")
    if "fit" not in str(c.get("servo_src", "")):
        add("서보 변환식", "이 bag에서 못 재서 프로필 값 사용: " + (c["substitutions"][0] if c.get("substitutions") else c["servo_src"]), "steer_edges")
    tab = acc.get("a_max", {}).get("by_speed", []) if isinstance(acc.get("a_max"), dict) else []
    if "i_motor" in S and np.isfinite(S["i_motor"]).any():
        if len(tab) < TH.ACCEL_MIN_BINS:
            sec = acc.get("a_max", {}).get("seconds", 0.0) if isinstance(acc.get("a_max"), dict) else 0.0
            add("구동 한계 a_max(v)", f"한계 전류 구간 {sec:.1f} s, 속도 구간 {len(tab)}칸 ({TH.ACCEL_MIN_BINS}칸 필요, 칸마다 {TH.ACCEL_BIN_MIN_SAMPLES}샘플)", "launch_brake")
        ab = acc.get("a_brake")
        if not (isinstance(ab, dict) and np.isfinite(ab.get("value", np.nan))):
            add("제동 한계 a_brake", f"회생 한계 전류 구간 0 s ({TH.ACCEL_MIN_RUN_S} s 이상 필요)", "launch_brake")
        for key, lab in (("current_max_fit", "구동 전류 한계"), ("current_min_fit", "회생 전류 한계")):
            f = acc.get(key) or {}
            if f.get("status") != "measured":
                add(lab, f"한계에 평평하게 붙은 구간 {f.get('runs', 0)}개 ({TH.CUR_PLATEAU_MIN_RUNS}개 필요): bag 안 극값 {f.get('extreme', float('nan')):.1f} A 는 하한만",
                    "launch_brake")
    cd = R.get("coastdown", {})
    if cd.get("status") != "measured":
        add("구름·공기 저항", "; ".join(cd.get("lacking", [])) or cd.get("how", ""), "coastdown")
    sr = R.get("steer_response", {})
    if sr.get("status") == "not_identified":
        add("조향 응답 (지연·서보 지연)", sr.get("how", ""), "steer_chirp")
    elif sr.get("status") == "sum_only":
        add("순수 지연 대 서보 지연의 분리", "; ".join(sr.get("lacking", [])), "steer_chirp")
    dx = R["delay_xcorr"]
    if not np.isfinite(dx.get("value", np.nan)):
        add("겉보기 지연", dx.get("how", ""), "steer_chirp")
    if R["trim"].get("n", 0) < TH.TRIM_MIN_SAMPLES or not np.isfinite(R["trim"].get("value", np.nan)):
        add("트림", f"거의 직진 주행 {R['trim'].get('n', 0) / SG.HZ:.1f} s ({TH.TRIM_MIN_SAMPLES / SG.HZ:.0f} s 필요)", "launch_brake")
    if R["windows"] < TH.REPLAY_MIN_WINDOWS:
        add("응답·타이어 맞춤", f"재생 구간 {R['windows']}개 ({TH.REPLAY_MIN_WINDOWS}개 필요)", None)
    if "imu" not in R["roles"]:
        add("모든 동역학 값", "IMU 없음", "standstill")
    for role, why in R["missing"].items():
        add(f"토픽 역할 '{role}' 없음", why, None)
    hand: List[Dict] = []
    car_m, veh_m, lid_m = hm.get("car", {}), hm.get("vehicle", {}), hm.get("lidar", {})
    if "steer_gain_wheel" not in car_m:
        hand.append(dict(item="명령 조향각 -> 실제 바퀴 각", why="bag 에 바퀴 각 측정이 없음 (k 는 타이어 강성과 맞바뀜)",
                         how="정지 상태에서 좌우 풀락과 중간 단계의 바퀴 각을 각도기·사진으로 재서 setup.yaml `hand_measured.wheel_angle_table` 에 적음",
                         test="steer_edges"))
    if "m" not in veh_m:
        hand.append(dict(item="질량", why="bag 으로 식별 불가 (힘이 질량당으로만 보임)",
                         how="저울로 재서 setup.yaml `hand_measured.mass_kg`", test="scale"))
    if "lr" not in veh_m:
        hand.append(dict(item="무게중심 앞뒤 위치", why="bag 으로 식별 불가",
                         how="바퀴별 저울 값으로 계산해 `hand_measured.cg_from_rear_axle_m` (뒤차축에서 무게중심까지)", test="scale"))
    if "Iz" not in veh_m:
        hand.append(dict(item="요 관성", why="bag 으로 식별 불가",
                         how="이중 줄 매달기·진자로 재서 `hand_measured.yaw_inertia_kgm2`", test="scale"))
    if "mount_x" not in lid_m:
        hand.append(dict(item="LiDAR 장착 위치 (실측)", why="/tf_static 은 설정값이지 잰 값이 아님",
                         how="줄자로 base_link(뒤차축) 기준 x, y, z 를 재서 `hand_measured.lidar_mount_m: [x, y, z]`", test=None))
    hand.append(dict(item="한계에서 앞이 먼저 빠지는가 뒤인가", why="주행 신호만으로는 안 갈림",
                     how="그립 램프(T6)를 영상으로 찍어 사람이 판정 (도구에 입력 경로 없음)", test="grip_ramp"))
    return dict(fixable=fix, not_from_driving=hand)


# ------------------------------------------------------------------------------------------------------------------
def make_figs(R, S, W, figs) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from ..bag_map import _use_korean_font
        _use_korean_font(plt)
    except Exception:
        pass
    paths = []
    Z = S.get("_raw", {})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    # 1 servo command vs applied
    a = ax[0, 0]
    if "servo_cmd" in Z and "servo_applied" in Z:
        j = SG.pair_after(Z["servo_cmd"][:, 0], Z["servo_applied"][:, 0], 0.02)
        ok = j >= 0
        a.plot(Z["servo_cmd"][ok, 1], Z["servo_applied"][j[ok], 1], ".", ms=1.5, alpha=0.4)
        a.axhline(R["calib"]["clamp_lo_servo"], color="r", lw=0.8); a.axhline(R["calib"]["clamp_hi_servo"], color="r", lw=0.8)
        a.set_xlabel("/commands/servo/position"); a.set_ylabel("/sensors/servo_position_command")
        lo, hi = SG.Calib(**{k: v for k, v in R["calib"].items() if k in SG.Calib.__dataclass_fields__}).steer_window_rad()
        a.set_title(f"서보 창 {lo:+.3f} ~ {hi:+.3f} rad, 잘린 명령 {100 * R['calib']['clamp_frac']:.1f} %")
    else:
        a.set_title("서보 명령/적용 토픽 없음")
    # 2 lateral acceleration vs speed
    a = ax[0, 1]
    mv = (S["v_wheel"] > 0.5)
    a.plot(S["v_wheel"][mv], np.abs(S["ay_s"][mv]), ".", ms=1, alpha=0.3)
    lat = R["lat"]
    if np.isfinite(lat.get("value", np.nan)):
        a.axhline(lat["value"], color="r", lw=1)
        a.set_title(f"|횡가속| 대 속도: 0.3 s 버틴 최대 {lat['value']:.1f} m/s² (μ ≥ {lat['mu_lower']:.2f})")
    a.set_xlabel("속도 [m/s]"); a.set_ylabel("|a_y| [m/s²]")
    # 3 accel / brake by speed
    a = ax[0, 2]
    acc = R["accel"]
    for key, col in (("a_max", "C0"), ("a_brake", "C3")):
        d = acc.get(key)
        if isinstance(d, dict) and d.get("by_speed"):
            vv = [0.5 * (t["v"][0] + min(t["v"][1], 10)) for t in d["by_speed"]]
            a.plot(vv, [t["median"] for t in d["by_speed"]], "o-", color=col, label=f"{key} (몸체)")
            a.plot(vv, [t["wheel"] for t in d["by_speed"]], "x--", color=col, alpha=0.5, label=f"{key} (바퀴)")
    a.set_xlabel("속도 [m/s]"); a.set_ylabel("[m/s²]"); a.legend(fontsize=8)
    a.set_title(f"전류 한계 가속 / 회생 한계 제동 (전류 {acc.get('current_max_A', float('nan')):.0f} / {acc.get('current_min_A', float('nan')):.0f} A)")
    # 4 replay examples
    a = ax[1, 0]
    pred = R.get("_pred")
    if pred is not None and len(W["r"]):
        ay1 = np.abs(W["ay"][:, WN.WARM:WN.WARM + 101]).max(1)
        i = int(np.argmax(ay1))
        tt = np.arange(WN.HOR + 1) / SG.HZ
        a.plot(tt, W["r"][i, WN.WARM:], "k", label="실측")
        a.plot(tt, pred["a"]["r"][i], label="(a) 선형 자전거")
        a.plot(tt, pred["b"]["r"][i], label="(b) 단일트랙")
        a.set_title(f"재생 예 (횡가속 최대 구간, t0 {W['t0'][i]:.1f} s)"); a.set_xlabel("예측 시간 [s]"); a.set_ylabel("요레이트 [rad/s]")
        a.legend(fontsize=8)
    # 5 LiDAR dropout by range
    a = ax[1, 1]
    li = R.get("lidar")
    if li:
        tab = li["dropout_by_range"]
        a.bar([f"{t['r'][0]}-{t['r'][1]}" for t in tab], [100 * t["miss_frac"] for t in tab])
        a.set_xlabel("거리 [m]"); a.set_ylabel("짧은 끊김 누락 [%]")
        sig = ", ".join(f"{s['r'][0]}-{s['r'][1]} m {s['sigma_mm']:.1f}" for s in li.get("still_noise", []))
        a.set_title(f"LiDAR 유효 {100 * li['valid_frac']:.1f} %; 정지 잡음 σ [mm] {sig or '정지 없음'}", fontsize=9)
    # 6 mu profile
    a = ax[1, 2]
    mp = R.get("mu_profile")
    if mp:
        xs = [q["x"] for q in mp["points"]]
        a.errorbar(xs, [1e4 * q["dJ"] for q in mp["points"]], yerr=[2e4 * q["se"] for q in mp["points"]], fmt="o-")
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel("최대 마찰 μ (앞뒤 공통)"); a.set_ylabel("오차 증가 ×1e4 (±2 표준오차)")
        a.set_title(f"마찰 식별: {mp['verdict']} {mp['interval'][0]:.2f}~{mp['interval'][1]:.2f}")
    fig.suptitle(f"bag_profile: {R['name']} ({R['place'].get('code')})")
    fig.tight_layout()
    p = os.path.join(figs, "overview.png")
    fig.savefig(p, dpi=90); plt.close(fig)
    paths.append(p)
    return paths


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, nd) for x in v) + "]"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "—"
        return f"{v:.{nd}f}" if (abs(v) >= 10 ** -nd or v == 0) else f"{v:.3g}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmt(x, nd)}" for k, x in v.items())
    return str(v)


def summary_md(R, car, place, missing, figs) -> str:
    L = [f"# bag_profile 요약: {R['name']}", "",
         f"- bag: `{R['bag']}` ({R['duration_s']:.0f} s, .db3 {R.get('bag_files', 1)}개), 장소 코드 **{R['place'].get('code')}**"
         f" (지도 {R['place'].get('size_m')}, md5 {R['place'].get('grid_md5')})",
         f"- 재생 구간 {R['windows']}개 (충돌로 뺀 구간 {R['impact_excluded']}개), 처리 {R['seconds']:.0f} s",
         f"- **구간 나누기: {R.get('segment_mode')}**" + (f" -- {R['note_split']}" if R.get("note_split") else ""), ""]
    if R["missing"]:
        L += ["## 없는 토픽", ""] + [f"- **{k}**: {v}" for k, v in R["missing"].items()] + [""]
    L += ["## 계산에 쓴 구간", "", "| 값 | 구간을 찾은 방식 |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in R.get("methods", {}).items()]
    L.append("")
    if R.get("segments"):
        L += ["표시된 구간 (/sysid/segment):", ""] + [f"- {x['t']:.1f} s: {x['label']}" for x in R["segments"]]
        L += [f"- 경고: {w}" for w in R.get("segment_warnings", [])] + [""]
    subs = R["calib"].get("substitutions") or []
    L += ["## 이 bag에서 못 재서 프로필·기본값으로 대체된 것", ""]
    L += [f"- {x}" for x in subs] if subs else ["- 없음 (서보 변환식, ERPM 환산, 자이로·가속도 단위 모두 이 bag에서 정함)"]
    L.append("")
    if R.get("setup_file"):
        L += [f"## setup.yaml 대조 (`{R['setup_file']}`)", "",
              "| 항목 | setup.yaml | bag | 차이 | 판정 | 비고 |", "|---|---|---|---|---|---|"]
        for r in R.get("setup_compare", []):
            d = r.get("diff")
            L.append(f"| {r['item']} | {_fmt(r['setup'])} | {_fmt(r['bag'])} | {_fmt(d) if not isinstance(d, str) else d} | "
                     f"{'**' + r['verdict'] + '**' if r['ok'] is False else r['verdict']} | {r.get('note', '')} |")
        if R.get("setup_ignored"):
            L += ["", "읽지 않은 키(오타 확인): " + ", ".join(R["setup_ignored"])]
        man = R.get("setup_manual") or {}
        if man:
            L += ["", "수동 실측으로 넣은 값 (출처 실측(수동)): " + ", ".join(f"{g}.{k}" for g, d in man.items() for k in d)]
        L.append("")
    else:
        L += ["## setup.yaml", "", "없음: bag 폴더 안이나 옆에 `setup.yaml` 을 두거나 `--setup` 으로 지정하면 설정값 대조와 수동 실측 입력이 된다.", ""]
    if R.get("setup_check"):
        L += ["## 차 프로필과 설정 비교", "", "| 항목 | bag | 프로필 | 일치 |", "|---|---|---|---|"]
        for r in R["setup_check"]:
            L.append(f"| {r['item']} | {_fmt(r['bag'])} | {_fmt(r['profile'])} | {'예' if r['ok'] else '**아니오**'} |")
        L.append("")
    def rows(title, groups):
        out = [f"## {title}", "", "| 값 | 수치 | 오차/구간 | 단위 | 출처 | 상태 | 방법 |", "|---|---|---|---|---|---|---|"]
        for g, d in groups:
            for k, e in d.items():
                out.append(f"| {g}.{k} | {_fmt(e.get('value'))} | {_fmt(e.get('err'))} | {e.get('unit', '')} | "
                           f"{e.get('source_label') or SOURCE_KO.get(e['source'], e['source'])} | "
                           f"{STATUS_KO.get(e['status'], e['status'])} | {e.get('how', '')[:90]} |")
        return out + [""]
    L += rows("차 보정 기록 (car_calibration.yaml)", [(g, car.get(g, {})) for g in ("car", "vehicle", "actuator", "lidar", "odom")])
    L += ["새 추정기 두 개(코스트다운 `car.coast_*`, 조향 응답 `car.steer_*`)는 **실차 미검증**이다: 합성 신호로 값을 되찾는지만 시험했고 "
          "실차 녹화로는 한 번도 돌려 보지 않았다.", ""]
    cd, sr = R.get("coastdown", {}), R.get("steer_response", {})
    L += ["- 코스트다운: **" + cd.get("status", "?") + "** -- " + cd.get("how", ""),
          "- 조향 응답: **" + sr.get("status", "?") + "** -- " + sr.get("how", ""), ""]
    L += rows("장소 프로필 (place_profile.yaml)", [("place", place["place"])])
    fe = R.get("fit_errors")
    if fe:
        L += ["## 재생 오차 (이 bag, 중앙값)", "", "| 모델 | 요레이트 0.5/1/2 s | 위치 2 s |", "|---|---|---|"]
        for m, d in fe.items():
            L.append(f"| {m} | {d[50]['yaw'][0]:.3f} / {d[100]['yaw'][0]:.3f} / {d[200]['yaw'][0]:.3f} | {_fmt(d[200]['pos'][0], 2)} |")
        L += ["", "(a), (b)는 이 bag으로 맞춘 값(표본 안 오차). car_profile은 --car 프로필 값 그대로(표본 밖)."]
        L.append("")
    fix, hand = missing["fixable"], missing["not_from_driving"]
    L += ["## 이 bag으로 못 잰 것 (a): 이런 녹화로 잴 수 있었는데 조건이 모자란 것 -- 다시 찍으면 됨", ""]
    if fix:
        L += ["| 값 | 무엇이 얼마나 모자랐나 | 시험 | 시험이 정하는 것 |", "|---|---|---|---|"]
        for m in fix:
            t = TESTS.get(m["test"]) if m.get("test") else None
            L.append(f"| {m['item']} | {m['why']} | {t[0] if t else '—'} | {t[1] if t else '—'} |")
    else:
        L.append("**이 bag으로 잴 수 있는 것은 다 쟀음.**")
    L += ["", "## 주행 녹화로는 원래 안 나오는 것 (b): 손으로 재서 setup.yaml 의 `hand_measured:` 에 적을 것", ""]
    if hand:
        L += ["| 값 | 왜 안 나오나 | 넣는 방법 |", "|---|---|---|"]
        L += [f"| {m['item']} | {m['why']} | {m['how']} |" for m in hand]
    L += [""] + [f"![{os.path.basename(p)}](figs/{os.path.basename(p)})" for p in figs]
    L += ["", "## 이 실행에 쓴 문턱값 (f1sim/bag_profile/thresholds.py)", "", TH.table_md()]
    return "\n".join(L) + "\n"


def write_outputs(R, out, S, W, carprof, log):
    car = car_record(R)
    place = place_record(R)
    miss = missing_list(R, S)
    try:
        figs = make_figs(R, S, W, os.path.join(out, "figs"))
    except Exception as e:                       # a figure must never cost the profile
        log(f"[그림] 실패: {e}")
        figs = []
    with open(os.path.join(out, "car_calibration.yaml"), "w") as f:
        f.write("# bag_profile 차 보정 기록. source: measured=실측 estimated=추정 spec=규격 assumed=가정;\n"
                "# status: measured=측정됨 lower_bound=하한만 not_identified=식별 안 됨 not_applied=시뮬에 없음.\n"
                "# source_label: 실측 / 추정 / 설정값(규격) / 가정 / 실측(수동, setup.yaml hand_measured).\n"
                "# vehicle/actuator/lidar/odom 의 값은 f1sim.profiles.loader 로 Config 에 얹을 수 있음 (apply: false 제외).\n")
        yaml.safe_dump(_jsonable(car), f, allow_unicode=True, sort_keys=False, width=140)
    with open(os.path.join(out, "place_profile.yaml"), "w") as f:
        f.write("# bag_profile 장소 프로필 (노면·벽·지도). 마찰은 레이스 bag 으로는 하한만 나옴.\n")
        yaml.safe_dump(_jsonable(place), f, allow_unicode=True, sort_keys=False, width=140)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(summary_md(R, car, place, miss, figs))
    R2 = {k: v for k, v in R.items() if not k.startswith("_")}
    R2["not_measured"] = miss
    R2["thresholds"] = {n: v for n, v, u, m in TH.TABLE}
    R2["fingerprint"] = fingerprint(R)
    json.dump(_jsonable(R2), open(os.path.join(out, "bag_profile.json"), "w"), indent=1, ensure_ascii=False)
    log(f"[쓰기] car_calibration.yaml, place_profile.yaml, summary.md, bag_profile.json -> {out}")
