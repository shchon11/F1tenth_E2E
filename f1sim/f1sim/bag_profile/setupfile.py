"""setup.yaml: the car's recorded configuration and the values somebody measured by hand.

Format: docs/bag_recording_guide.md section 2.3. Two jobs here:
  * `compare` sets every configured value against what the bag says (match / mismatch with the difference / not
    comparable and why) -- this is what catches a flipped servo, a gyro in deg/s, a different regen current;
  * `manual_entries` turns the `hand_measured:` block (wheel-angle table, mass, centre of gravity, yaw inertia, LiDAR
    mount) into profile entries with source 'measured', origin 'manual' (shown as 실측(수동)), which no recording can give.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import yaml

from . import thresholds as TH

LASER_FRAMES = ("laser", "lidar", "laser_frame")
L_WHEELBASE = 0.3302
LR_DEFAULT = 0.17145          # CoG -> rear axle of the stored profile (models.LR)

#: setup.yaml keys the tool understands (anything else is listed as ignored, so a typo does not pass silently)
KNOWN = {
    **{k: None for k in ("date", "place", "surface", "tyre", "battery_v_start", "battery_v_end", "changed_since_last")},
    "vesc": {"speed_to_erpm_gain", "servo_min", "servo_max", "steering_to_servo_gain", "steering_to_servo_offset",
             "current_motor_max_A", "current_regen_min_A"},
    "imu": {"driver", "gyro_unit", "accel_unit"},
    "lidar": {"tf_base_to_laser", "frame_name"},
    "hand_measured": {"wheel_angle_table", "mass_kg", "wheelbase_m", "cg_from_rear_axle_m", "yaw_inertia_kgm2",
                      "lidar_mount_m"},
}


def find(bag_path: str, explicit: Optional[str] = None) -> Optional[str]:
    """--setup path, else setup.yaml inside the bag folder, else next to it (the bag folder's parent)."""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--setup {explicit}: 없는 파일")
        return explicit
    p = bag_path.rstrip("/")
    d = os.path.dirname(p) if os.path.isfile(p) else p
    for cand in (os.path.join(d, "setup.yaml"), os.path.join(os.path.dirname(d), "setup.yaml")):
        if os.path.isfile(cand):
            return cand
    return None


def load(path: str) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: 최상위가 매핑이어야 함")
    d["_path"] = os.path.abspath(path)
    ignored = []
    for k, v in d.items():
        if k == "_path":
            continue
        if k not in KNOWN:
            ignored.append(k)
        elif isinstance(KNOWN[k], set) and isinstance(v, dict):
            ignored += [f"{k}.{kk}" for kk in v if kk not in KNOWN[k]]
    d["_ignored"] = ignored
    return d


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _row(item, setup, bag, tol=None, unit="", mode="eq", note=""):
    """one comparison row. mode eq: |setup-bag| <= tol; str: equal strings."""
    r = dict(item=item, setup=setup, bag=bag, unit=unit, tol=tol, note=note)
    if setup is None:
        return None
    if bag is None:
        r.update(verdict="비교 불가", ok=None, diff=None)
        return r
    if mode == "str":
        ok = str(setup).strip().lower() == str(bag).strip().lower()
        r.update(verdict="일치" if ok else "불일치", ok=bool(ok), diff=None if ok else f"{setup} <-> {bag}")
        return r
    d = float(bag) - float(setup)
    ok = abs(d) <= tol
    r.update(verdict="일치" if ok else "불일치", ok=bool(ok), diff=d)
    return r


def compare(setup: dict, R: dict) -> List[dict]:
    """setup.yaml vs the bag's own estimates. R is the report dict (calib, accel, tf_static)."""
    rows: List[dict] = []
    c = R["calib"]
    v, imu, lid = setup.get("vesc") or {}, setup.get("imu") or {}, setup.get("lidar") or {}
    fit_ok = "fit" in str(c.get("servo_src", ""))

    def add(r):
        if r:
            rows.append(r)
    add(_row("ERPM 환산", _num(v.get("speed_to_erpm_gain")), _num(c.get("erpm_gain")), TH.SETUP_TOL_ERPM, "ERPM/(m/s)"))
    sg = _row("서보 이득", _num(v.get("steering_to_servo_gain")), _num(c.get("servo_g")) if fit_ok else None,
              TH.SETUP_TOL_SERVO_GAIN, "서보/rad")
    if sg and sg["ok"] is None:
        sg["note"] = "이 bag에서 변환식을 못 맞춤 (프로필 값 또는 항등을 쓴 것)"
    add(sg)
    so = _row("서보 오프셋", _num(v.get("steering_to_servo_offset")), _num(c.get("servo_o")) if fit_ok else None,
              TH.SETUP_TOL_SERVO_OFFSET, "서보")
    if so and so["ok"] is None:
        so["note"] = "이 bag에서 변환식을 못 맞춤"
    add(so)
    # servo window ends: equal when the bag saw the clamp at that end; else the bag only shows a range that must lie
    # inside the configured window
    for key, bagv, seen, side in (("servo_min", c.get("clamp_lo_servo"), c.get("clamp_seen_lo"), "lo"),
                                  ("servo_max", c.get("clamp_hi_servo"), c.get("clamp_seen_hi"), "hi")):
        sv = _num(v.get(key))
        bv = _num(bagv)
        if sv is None:
            continue
        if bv is None:
            add(_row(f"서보 창 {key}", sv, None))
            continue
        if seen:
            add(_row(f"서보 창 {key}", sv, bv, TH.SETUP_TOL_SERVO_RANGE, "서보"))
        else:
            inside = (bv >= sv - TH.SETUP_TOL_SERVO_RANGE) if side == "lo" else (bv <= sv + TH.SETUP_TOL_SERVO_RANGE)
            rows.append(dict(item=f"서보 창 {key}", setup=sv, bag=bv, unit="서보", tol=TH.SETUP_TOL_SERVO_RANGE,
                             verdict="일치(하한만 확인)" if inside else "불일치", ok=bool(inside),
                             diff=None if inside else bv - sv,
                             note="이 bag에서 그쪽 끝이 잘린 적이 없어, bag 의 범위가 설정 창 안인지만 확인" if inside
                             else "bag 의 서보 위치가 설정 창을 벗어남"))
    acc = R.get("accel", {})
    for key, fitkey, ext_key, sign in (("current_motor_max_A", "current_max_fit", "current_max_A", 1),
                                       ("current_regen_min_A", "current_min_fit", "current_min_A", -1)):
        sv = _num(v.get(key))
        fit = acc.get(fitkey)
        if sv is None:
            continue
        if not fit or not np.isfinite(fit.get("value", np.nan)):
            add(_row(f"전류 한계 {key}", sv, None))
            continue
        bv = float(fit["value"])
        if fit["status"] == "measured":
            add(_row(f"전류 한계 {key}", sv, bv, TH.SETUP_TOL_CURRENT_A, "A"))
        else:
            inside = (abs(bv) <= abs(sv) + TH.SETUP_TOL_CURRENT_A) and (np.sign(bv) == np.sign(sv) or bv == 0)
            rows.append(dict(item=f"전류 한계 {key}", setup=sv, bag=bv, unit="A", tol=TH.SETUP_TOL_CURRENT_A,
                             verdict="일치(하한만 확인)" if inside else "불일치", ok=bool(inside),
                             diff=None if inside else bv - sv,
                             note="bag 안 최댓값이 한계에 붙어 평평하게 유지된 적이 없어 하한만: 설정 한계보다 크지 않음을 확인" if inside
                             else "bag 의 전류가 설정 한계를 넘음 (설정이 다르거나 단위 문제)"))
    add(_row("자이로 단위", imu.get("gyro_unit"), c.get("gyro_unit"), mode="str"))
    add(_row("가속도 단위", imu.get("accel_unit"), str(c.get("accel_unit", "")).split(" ")[0], mode="str"))
    tf = next((t for t in R.get("tf_static", []) if t[1] in LASER_FRAMES or t[1] == lid.get("frame_name")), None)
    if lid.get("frame_name"):
        add(_row("LiDAR 프레임 이름", lid["frame_name"], tf[1] if tf else None, mode="str"))
    cfg = lid.get("tf_base_to_laser")
    if cfg is not None:
        for k, nm in enumerate("xyz"):
            add(_row(f"LiDAR tf {nm}", _num(cfg[k]) if len(cfg) > k else None, float(tf[2 + k]) if tf else None,
                     TH.SETUP_TOL_TF_M, "m"))
    return rows


def manual_entries(setup: dict) -> Dict[str, Dict[str, dict]]:
    """{group: {key: entry kwargs}} from `hand_measured:` -- value, unit, how, extra; report.E builds the entries."""
    hm = setup.get("hand_measured") or {}
    out: Dict[str, Dict[str, dict]] = {"car": {}, "vehicle": {}, "lidar": {}}
    where = f"setup.yaml hand_measured ({os.path.basename(setup.get('_path', 'setup.yaml'))})"
    tab = hm.get("wheel_angle_table")
    if tab:
        rows, xs, ys = [], [], []
        for r in tab:
            cmd = _num(r.get("cmd_rad"))
            angs = [_num(r.get(k)) for k in ("left_deg", "right_deg") if _num(r.get(k)) is not None]
            if cmd is None or not angs:
                continue
            a = float(np.deg2rad(np.mean(angs)))
            rows.append(dict(cmd_rad=cmd, wheel_rad=a, left_deg=_num(r.get("left_deg")), right_deg=_num(r.get("right_deg"))))
            xs.append(cmd); ys.append(a)
        if rows:
            out["car"]["wheel_angle_table"] = dict(value=rows, unit="rad", how=f"손으로 잰 명령 조향각 -> 바퀴 각 ({where})")
            x, y = np.array(xs), np.array(ys)
            if len(x) >= 2 and np.sum(x ** 2) > 0:
                k = float(np.sum(x * y) / np.sum(x ** 2))
                res = float(np.sqrt(np.mean((y - k * x) ** 2)))
                out["car"]["steer_gain_wheel"] = dict(value=k, err=res, unit="바퀴 각 / 명령 조향각",
                                                      how=f"위 표를 원점을 지나는 직선으로 맞춤, 잔차 rms {res:.4f} rad ({len(x)}점)")
    if _num(hm.get("mass_kg")) is not None:
        out["vehicle"]["m"] = dict(value=float(hm["mass_kg"]), unit="kg", how=f"저울 ({where})")
    wb = _num(hm.get("wheelbase_m")) or L_WHEELBASE
    if _num(hm.get("cg_from_rear_axle_m")) is not None:
        lr = float(hm["cg_from_rear_axle_m"])
        out["vehicle"]["lr"] = dict(value=lr, unit="m", how=f"무게중심 뒤차축 기준 거리 ({where})")
        out["vehicle"]["lf"] = dict(value=wb - lr, unit="m", how=f"축거 {wb:.4f} m - lr ({where})")
    if _num(hm.get("yaw_inertia_kgm2")) is not None:
        out["vehicle"]["Iz"] = dict(value=float(hm["yaw_inertia_kgm2"]), unit="kg m^2", how=f"매달기 측정 ({where})")
    lm = hm.get("lidar_mount_m")
    if lm is not None and len(lm) >= 3:
        lr_used = float(hm["cg_from_rear_axle_m"]) if _num(hm.get("cg_from_rear_axle_m")) is not None else LR_DEFAULT
        out["lidar"]["mount_x"] = dict(value=float(lm[0]) - lr_used, unit="m (ahead of CoG)",
                                       how=f"줄자로 잰 base_link x {float(lm[0]):.3f} - lr {lr_used:.4f} ({where})")
        out["lidar"]["mount_z"] = dict(value=float(lm[2]), unit="m", how=f"줄자로 잰 높이 ({where})")
        out["car"]["lidar_mount_base_link"] = dict(value=[float(q) for q in lm[:3]], unit="m (base_link x, y, z)",
                                                   how=f"줄자로 잰 LiDAR 위치 ({where})")
    return {g: d for g, d in out.items() if d}
