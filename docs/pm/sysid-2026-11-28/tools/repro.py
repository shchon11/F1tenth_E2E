"""Reproducibility table from the per-bag bag_profile runs: one row per bag, grouped by (car setup x place)."""
import json, os, glob
import numpy as np
from common import WORK, PLACE, OUT

rows = []
for key in sorted(os.listdir(os.path.join(WORK, "profiles"))):
    f = os.path.join(WORK, "profiles", key, "bag_profile.json")
    if not os.path.exists(f):
        continue
    R = json.load(open(f))
    c = R["calib"]; acc = R.get("accel", {}); lat = R.get("lat", {}); li = R.get("lidar", {}) or {}
    place = next((p for p, ks in PLACE.items() if key in ks), "?")
    g = lambda d, *ks: (lambda v: v)(__import__("functools").reduce(lambda a, k: (a or {}).get(k) if isinstance(a, dict) else None, ks, d))
    lo = hi = None
    if c.get("servo_g") and c.get("clamp_lo_servo") is not None:
        a_ = (c["clamp_lo_servo"] - c["servo_o"]) / c["servo_g"]; b_ = (c["clamp_hi_servo"] - c["servo_o"]) / c["servo_g"]
        lo, hi = min(a_, b_), max(a_, b_)
    tf = [t for t in R.get("tf_static", []) if t[1] in ("laser", "lidar")]
    drop68 = next((x["miss_frac"] for x in li.get("dropout_by_range", []) if x["r"] == [6, 8]), None)
    sig04 = [x["sigma_mm"] for x in li.get("still_noise", []) if x["r"][1] <= 4]
    rows.append(dict(
        key=key, place=place, windows=R.get("windows"), dur=R.get("duration_s"),
        servo_g=c.get("servo_g"), servo_o=c.get("servo_o"), win_lo=lo, win_hi=hi, edge_hit=[c.get("clamp_seen_lo"), c.get("clamp_seen_hi")],
        erpm=c.get("erpm_gain"), gyro=c.get("gyro_unit"), accel=c.get("accel_unit"),
        i_max=acc.get("current_max_A"), i_min=acc.get("current_min_A"),
        a_brake=g(acc, "a_brake", "value"), a_brake_s=g(acc, "a_brake", "seconds"), a_max=g(acc, "a_max_vswitch_fit", "a_max"),
        delay_xc=g(R, "delay_xcorr", "value"), lin_tau=g(R, "lin_fit", "p", "tau"), lin_b=g(R, "lin_fit", "p", "b"),
        gain5=g(R, "lin_fit", "gain_5mps"), straight=g(R, "trim", "value"),
        k=g(R, "tyre_fit", "p", "k"), tau_d=g(R, "tyre_fit", "p", "tau_d"), tau_s=g(R, "tyre_fit", "p", "tau_s"),
        sf=g(R, "tyre_fit", "p", "sf"), sr=g(R, "tyre_fit", "p", "sr"),
        sf_v=g(R, "sf_profile", "verdict"), sr_v=g(R, "sr_profile", "verdict"), sf_i=g(R, "sf_profile", "interval"), sr_i=g(R, "sr_profile", "interval"),
        mu_v=g(R, "mu_profile", "verdict"), mu_i=g(R, "mu_profile", "interval"),
        ay_held=lat.get("value"), slides=len(lat.get("slides_t", []) or []),
        lidar_x=tf[0][2] if tf else None, valid=li.get("valid_frac"), drop68=drop68, noise04=float(np.median(sig04)) if sig04 else None,
        map_scale=g(R, "bag_map", "scale"), map_src=g(R, "bag_map", "source"), map_skip=g(R, "bag_map", "why"),
        gyro_ev=g(c, "gyro_evidence", "vs_pose_heading_rate", "slope"), seconds=R.get("seconds")))
json.dump(rows, open(os.path.join(WORK, "repro_rows.json"), "w"), indent=1)
f = lambda v, n=3: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
print("| bag | 장소 | 구간 | 서보 g / o | 서보 창 [rad] | ERPM | 자이로 | 회생 최소 [A] | 제동 [m/s²] | 지연 xcorr [s] | (a) τ / b | 5 m/s 이득 | (b) k / τd / τs | sf / sr | μ 판정 | 버틴 a_y | LiDAR x | 6~8 m 누락 | 정지 잡음 [mm] | 지도 축척 |")
print("|" + "---|" * 20)
for r in rows:
    print(f"| {r['key']} | {r['place']} | {r['windows']} | {f(r['servo_g'])} / {f(r['servo_o'])} | {f(r['win_lo'])} ~ {f(r['win_hi'])} | {f(r['erpm'], 0)} | {r['gyro']} | "
          f"{f(r['i_min'], 0)} | {f(r['a_brake'], 2)} | {f(r['delay_xc'])} | {f(r['lin_tau'])} / {f(r['lin_b'])} | {f(r['gain5'])} | "
          f"{f(r['k'], 2)} / {f(r['tau_d'])} / {f(r['tau_s'])} | {f(r['sf'], 2)} / {f(r['sr'], 2)} | {r['mu_v'] or '—'} {f(r['mu_i'][0], 2) if r['mu_i'] else ''} | "
          f"{f(r['ay_held'], 1)} | {f(r['lidar_x'])} | {f(r['drop68'], 3)} | {f(r['noise04'], 1)} | {f(r['map_scale'])} |")

# grouped: per (car setup x place) median [min, max] of the values that should repeat
print()
KEYS = [("servo_g", 3), ("servo_o", 3), ("win_lo", 3), ("win_hi", 3), ("erpm", 0), ("i_min", 0), ("a_brake", 2), ("delay_xc", 3),
        ("lin_tau", 3), ("lin_b", 3), ("gain5", 3), ("k", 2), ("tau_d", 3), ("tau_s", 3), ("sf", 2), ("sr", 2), ("ay_held", 1),
        ("lidar_x", 3), ("noise04", 1), ("drop68", 4), ("map_scale", 3)]
places = ["comp_map08x23", "pre_map12x16", "pre_map17x18", "pre_map16x07"]
print("| 값 | " + " | ".join(places) + " |")
print("|---|" + "---|" * len(places))
for k, n in KEYS:
    cells = []
    for p in places:
        static = k in ("servo_g", "servo_o", "win_lo", "win_hi", "erpm", "i_min", "lidar_x", "noise04", "drop68", "map_scale", "ay_held")
        v = [r[k] for r in rows if r["place"] == p and r.get(k) is not None and (static or (r["windows"] or 0) >= 30)]
        v = [x for x in v if isinstance(x, (int, float))]
        cells.append("—" if not v else (f"{np.median(v):.{n}f} [{min(v):.{n}f}, {max(v):.{n}f}] ({len(v)})"))
    print(f"| {k} | " + " | ".join(cells) + " |")
