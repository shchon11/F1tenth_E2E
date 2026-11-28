"""Write f1sim/f1sim/profiles/*.yaml from the fitted results (one source of numbers for REPORT.md and the profiles).

Profiles (car setup x place):
  default_comp_car_pre_place  DEFAULT CANDIDATE: competition-car values + pre-competition place values (PM 11-28)
  comp_car_map08x23           competition car at the competition venue (structure selection / validation, NOT default)
  pre_car_map12x16 / pre_car_map17x18 / pre_car_map16x07   the pre-competition car at its three places (as recorded)
"""
import json, os
import numpy as np
import yaml
from common import WORK, REPO

P = os.path.join(REPO, "f1sim/f1sim/profiles")
c3 = json.load(open(f"{WORK}/fit_sim3_comp.json"))["over"]
pre = json.load(open(f"{WORK}/pre_sim_over.json"))
I = json.load(open(f"{WORK}/ident_comp.json"))
bp = json.load(open(f"{WORK}/fit1_comp.json"))["st_C13"]["p"]
rows = {r["key"]: r for r in json.load(open(f"{WORK}/repro_rows.json"))}
PP = json.load(open(f"{WORK}/pre_place.json"))
COMP = [k for k in rows if k.startswith("202608") and rows[k]["windows"]]
PRE = {"map12x16": ["20260725-014941", "20260725-015428", "20260725-020031", "20260725-020831"],
       "map17x18": ["20260714-223333", "20260714-223707"], "map16x07": ["20260727-204907"]}


def med(keys, f, cond=lambda r: True):
    v = [rows[k][f] for k in keys if k in rows and rows[k].get(f) is not None and cond(rows[k])]
    return (float(np.median(v)), float(np.min(v)), float(np.max(v)), len(v)) if v else (None, None, None, 0)


def E(value, err=None, unit="", source="estimated", status="measured", how="", note="", apply=None, tier=None):
    e = dict(value=(round(float(value), 6) if isinstance(value, (float, np.floating)) else value), unit=unit, source=source, status=status)
    if err is not None:
        e["err"] = [None if x is None else round(float(x), 6) for x in err] if isinstance(err, (list, tuple)) else round(float(err), 6)
    if tier:
        e["tier"] = tier
    if how:
        e["how"] = how
    if note:
        e["note"] = note
    if apply is not None:
        e["apply"] = apply
    return e


rel = lambda name: (I[name]["interval"][0] / I[name]["best"], I[name]["interval"][1] / I[name]["best"])
# ---------- car values: competition car ----------
g_c = med(COMP, "servo_g"); o_c = med(COMP, "servo_o"); er_c = med(COMP, "erpm"); imin_c = med(COMP, "i_min"); imax_c = med(COMP, "i_max")
ab_c = med(COMP, "a_brake", lambda r: (r["a_brake_s"] or 0) >= 0.5)
win_c = (med(COMP, "win_lo")[0], med(COMP, "win_hi")[0])
lidx_c = med(COMP, "lidar_x")[0]
noise_c = med(COMP, "noise04")


def car_comp():
    k_lo, k_hi = rel("k"); sf_lo, sf_hi = rel("sf"); sr_lo, sr_hi = rel("sr")
    act = {
        "steer_gain": E(c3["steer_gain"], [c3["steer_gain"] * k_lo, c3["steer_gain"] * k_hi], "-", tier="car",
                        how="(c2*) simulator refit, competition TRAIN bags, 1 s yaw-rate error; interval: (b') profile ratio",
                        note="명령 조향각 대비 바퀴 각 이득. 타이어 강성과 일부 맞바뀜(저속 원선회 T2로 확정)"),
        "steer_bias": E(c3["steer_bias"], [I["b"]["interval"][0], I["b"]["interval"][1]], "rad", tier="car",
                        how="(c2*) refit; interval: (b') profile", note="세션마다 바뀔 수 있음: 새 세션 직진 10 s로 다시 맞춤"),
        "cmd_delay": E(c3["cmd_delay"], I["tau_d"]["interval"], "s", tier="car",
                       how="(b') profile picked 0.043 (0.04-0.05); the simulator rounds it to 1 ms", note="순수 지연. 차 설정(스택·VESC)에 붙음"),
        "servo_tau": E(c3["servo_tau"], I["tau_s"]["interval"], "s", tier="car", how="(c2*) refit; interval: (b') profile",
                       note="요 관성 Iz와 맞바뀜 (Iz/m ×1.5 이면 0.020)"),
        "motor_tau": E(c3["motor_tau"], unit="s", tier="car", how="(c2) longitudinal refit on the speed error (sim P loop)",
                       note="시뮬 속도 루프 구조(P만)에 맞춘 실효값; 1차+지연 모델보다 속도 오차 1.2배"),
        "motor_delay": E(c3["motor_delay"], unit="s", tier="car", how="same fit", note="속도 채널 추가 지연 (cmd_delay 위에)"),
    }
    veh = {
        "a_max": E(c3["a_max"], unit="m/s^2", tier="car", how="(c2) speed-replay fit with v_switch 7.0",
                   note=f"전류 한계 구간 직접 측정 중앙값 {med(COMP, 'a_max')[0]:.2f} (bag {med(COMP, 'a_max')[3]}개)"),
        "a_brake": E(5.0, [ab_c[1], ab_c[2]], "m/s^2", source="measured", tier="car",
                     how=f"regen-limited braking (current <= 90 % of {imin_c[0]:.0f} A), bag_profile per bag: median {ab_c[0]:.2f} over {ab_c[3]} bags",
                     note="시뮬 기본값 5.0 유지(11-09 최강 200 ms 측정); bag별 중앙값은 5.0보다 낮음 -> err 범위로 표시"),
    }
    lid = {"mount_x": E(lidx_c - 0.17145, unit="m (ahead of CoG)", source="measured", tier="car",
                        how=f"/tf_static base_link->laser x {lidx_c:.3f} minus lr 0.17145"),
           "mount_z": E(0.110, unit="m", source="measured", tier="car", how="/tf_static z")}
    if noise_c[0]:
        lid["noise_std"] = E(noise_c[0] / 1000, [noise_c[1] / 1000, noise_c[2] / 1000], "m", source="measured", tier="car",
                             how="1.4826 MAD per beam over genuinely still 1 s blocks, 0-4 m, bag_profile per bag")
    car = {
        "servo_map_gain": E(g_c[0], [g_c[1], g_c[2]], "servo/rad", "measured", tier="car", how="/ackermann_cmd vs /commands/servo/position"),
        "servo_map_offset": E(o_c[0], [o_c[1], o_c[2]], "servo", "measured", tier="car", how="same"),
        "steer_window": E([round(win_c[0], 4), round(win_c[1], 4)], unit="rad (command)", source="measured", status="not_applied", tier="car",
                          how="VESC clamp: /sensors/servo_position_command range, both edges hit in every race bag",
                          note="시뮬 s_max는 좌우 대칭(±0.4189)이라 넣을 수 없음. 오른쪽 −0.254에서 잘림"),
        "erpm_gain": E(er_c[0], unit="ERPM/(m/s)", source="measured", tier="car", how="/commands/motor/speed / speed"),
        "gyro_unit": E("rad/s", source="measured", tier="car", how="slope vs localisation heading rate 0.9-1.0"),
        "regen_current_min": E(imin_c[0], [imin_c[1], imin_c[2]], "A", "measured", tier="car", how="p0.1 of current_motor"),
        "drive_current_max": E(imax_c[0], [imax_c[1], imax_c[2]], "A", "measured", tier="car", how="p99.9 of current_motor"),
    }
    return act, veh, lid, car


STRUCT = {  # values that carry the adopted structure (b' expressed in the simulator) -- the same in every profile
    "h": E(0.0, unit="m", source="estimated", tier="structure",
           how="(d) and (c1) (longitudinal load transfer with linear-in-load tyres) predicted WORSE than (b)/(c2) on val and final",
           note="실제 무게중심 높이(0.074)가 아니라 '종방향 하중 이동을 끄는' 실효값. 시뮬에서 h는 하중 이동·IMU 레버암·접촉 모멘트에만 쓰임"),
    "contact_z_end": E(0.031 - 0.074, unit="m", source="assumed", tier="structure",
                       how="shifted with h so the contact pitch moment (h - contact_z_end) is unchanged", note="접촉 모멘트 보존용 이동"),
    "contact_z_side": E(0.060 - 0.074, unit="m", source="assumed", tier="structure",
                        how="shifted with h so the contact roll moment (h - contact_z_side) is unchanged", note="접촉 모멘트 보존용 이동"),
    "C_f": E(1.3, unit="-", source="assumed", status="not_identified", tier="structure",
             how="C 1.0 / 1.3 / free(1.0-2.2) fit equally on val and final", note="교과서 횡방향 값. 모양은 레이스 bag으로 구분 안 됨"),
    "C_r": E(1.3, unit="-", source="assumed", status="not_identified", tier="structure", how="same"),
    "E_f": E(0.0, unit="-", source="assumed", status="not_identified", tier="structure"),
    "E_r": E(0.0, unit="-", source="assumed", status="not_identified", tier="structure"),
}
GEOM = {
    "m": E(3.74, unit="kg", source="spec", status="not_identified", tier="geometry", apply=False,
           how="f1tenth_gym default; the bags see forces per unit mass only", note="저울로 잴 것 (필요할 때 잴 목록)"),
    "Iz": E(0.04712, [0.6 * 0.04712, 1.5 * 0.04712], "kg m^2", "spec", "not_identified", tier="geometry", apply=False,
            how=f"Iz/m x0.6: train error +{100 * I['sensitivity']['iz_m x0.6']['dJ_rel']:.1f} %, x1.5: {100 * I['sensitivity']['iz_m x1.5']['dJ_rel']:.1f} % (servo_tau absorbs it)",
            note="데이터는 1.5배 쪽을 약간 선호하나 서보 지연과 맞바뀜"),
    "lf": E(0.15875, unit="m", source="spec", status="not_identified", tier="geometry", apply=False,
            how="lr +-0.02 m changes the train error by < 0.3 %", note="무게중심 앞뒤 위치: 바퀴별 저울로 잴 것"),
    "lr": E(0.17145, unit="m", source="spec", status="not_identified", tier="geometry", apply=False, how="same"),
    "c_roll": E(0.1, unit="m/s^2", source="assumed", status="not_identified", tier="place", apply=False,
                how="no coast-down in any bag", note="코스트다운 T4로 잴 것"),
    "c_drag": E(0.01, unit="1/m", source="assumed", status="not_identified", tier="place", apply=False, how="same"),
}


def tyres(mu, mu_note, mu_status, mu_err, mu_source="estimated"):
    """the place-attached tyre-on-floor values: peak friction (front 1.04x rear) and stiffness slopes kept from the
    competition fit (mu*B*C constant: the pre bags cannot tell their stiffness from the competition one)."""
    sf_lo, sf_hi = rel("sf"); sr_lo, sr_hi = rel("sr")
    Bf = c3["B_f"] * c3["mu"] / mu; Br = c3["B_r"] * c3["mu"] / mu
    return {
        "mu": E(mu, mu_err, "-", mu_source, mu_status, tier="place", note=mu_note,
                how="peak friction of the rear axle (front = mu_f_scale x mu); see REPORT.md section 4 / 6"),
        "mu_ref": E(mu, unit="-", source=mu_source, status=mu_status, tier="place",
                    how="= mu, so B is the raw stiffness factor and randomising mu moves only the peak (mu_stiffness_exp 0)"),
        "mu_f_scale": E(c3["mu_f_scale"], unit="-", source="estimated", status="lower_bound", tier="place",
                        how="front peak 1.4 = lower bound of its profile (C 1.3), rear 1.35 inside 1.1-1.7",
                        note="앞 최대 마찰은 하한(1.4)만 나옴; 비율 1.04 = 앞이 아주 조금 늦게 포화"),
        "B_f": E(Bf, [Bf * sf_lo, Bf * sf_hi], "1/rad", "estimated", "measured", tier="place",
                 how=f"front slope mu*scale*B*C = {c3['mu'] * c3['mu_f_scale'] * c3['B_f'] * 1.3:.2f}/rad from the competition (c2*) fit, kept; interval: (b') sf profile ratio",
                 note="앞 강성(조향 컴플라이언스 포함). 사전 장소 bag으로는 대회장 값과 구분 안 됨"),
        "B_r": E(Br, [Br * sr_lo, Br * sr_hi], "1/rad", "estimated", "measured", tier="place",
                 how=f"rear slope mu*B*C = {c3['mu'] * c3['B_r'] * 1.3:.2f}/rad from the competition (c2*) fit, kept; interval: (b') sr profile ratio"),
    }


FP_COMP = {"servo map gain": round(g_c[0], 4), "servo map offset": round(o_c[0], 4), "ERPM gain": round(er_c[0], 1),
           "gyro unit": "rad/s", "regen current min [A]": round(imin_c[0], 1), "total delay (a) [s]": round(med(COMP, "lin_tau")[0], 3),
           "LiDAR mount x [m, base_link]": round(lidx_c, 3)}
MODEL_B = {k: E(bp[k], source="estimated", status="measured" if k not in ("mf",) else "lower_bound", how="(b') competition TRAIN fit")
           for k in ("k", "b", "tau_d", "tau_s", "sf", "sr", "mr")}
MODEL_B["mf"] = E(1.4, source="estimated", status="lower_bound", how="(b') front peak: lower bound of the profile (best 2.25)")


def dump(name, header, doc):
    path = os.path.join(P, name + ".yaml")
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=150)
    print("wrote", path)


HDR = ("# {title}\n# 만든 곳: docs/pm/sysid-2026-11-28 (REPORT.md). 기본값(params.py)은 바꾸지 않음; f1sim.profiles.loader 로 얹음.\n"
       "# source: measured=실측 estimated=추정 spec=규격 assumed=가정 | status: measured=측정됨 lower_bound=하한만 "
       "not_identified=식별 안 됨 not_applied=시뮬에 없음 | tier: car=차 설정 place=장소 structure=구조 geometry=제원\n"
       "# apply: false 인 값은 기록만 하고 Config 에 넣지 않음.\n")
act_c, veh_c, lid_c, car_c = car_comp()
# ---------- default: competition car + pre-competition place ----------
pre_ay = {p: med(ks, "ay_held")[2] for p, ks in PRE.items()}
mu_def = tyres(1.0, f"사전 장소: 모델 없이 본 하한 {max(pre_ay.values()) / 9.81:.2f} (0.3 s 버틴 최대 a_y {max(pre_ay.values()):.1f} m/s²); "
                    "map16x07 의 유일한 실제 스핀이 μ≈1.0 에서 재현되고 1.35 에서는 안 됨 (사건 1개, 약한 근거). 대회장은 ≥1.1",
               "lower_bound", [round(max(pre_ay.values()) / 9.81, 3), 1.35])
default = {
    "_meta": dict(name="default_comp_car_pre_place", default_candidate=True,
                  combines="car values: competition car (08-26/27 bags); place values: pre-competition test places (07-14/25/27 bags)",
                  recorded_together=False,
                  validity="REPORT.md section 6: the competition-car structure + the pre car's own delay/trim predicts the pre places as well as a local refit (held-out 1 s yaw 0.119 vs 0.187 nominal); place values the pre bags cannot pin are listed in measure_when_needed",
                  structure="(c2*): simulator single track, longitudinal load transfer off (h 0), C 1.3, refitted values"),
    "vehicle": {**{k: v for k, v in mu_def.items()}, **{k: v for k, v in STRUCT.items() if k in ("h", "contact_z_end", "contact_z_side", "C_f", "C_r", "E_f", "E_r")},
                "a_max": veh_c["a_max"], "a_brake": veh_c["a_brake"], **GEOM},
    "actuator": act_c, "lidar": lid_c, "car": car_c, "model_b": MODEL_B, "fingerprint": FP_COMP,
    "randomization": {"vehicle.mu": dict(range=[0.88, 1.35], note="곱셈(기본값 1.0 기준): 사전 장소 하한 0.88 ~ 대회장 수준 1.35. 식별 안 된 값이라 넓게"),
                      "actuator.steer_bias": dict(range=[-0.035, -0.010], note="절대값: 세션 간 트림 변화(대회 −0.024, 사전 −0.008)를 덮음")},
    "measure_when_needed": [
        dict(item="최대 마찰 μ (사전 장소)", why="하한 0.88만 실측, 1.0은 스핀 1건 재현에서 나온 약한 추정", test="T6 일정 반경 속도 램프 (5분)", when="한계 주행·추월 학습 전"),
        dict(item="트림 steer_bias", why="세션마다 바뀜 (대회 −0.024, 사전 차 −0.008)", test="아무 주행 10 s (bag_profile 이 자동으로 맞춤)", when="매 세션"),
        dict(item="노면이 바뀐 뒤 강성 B_f/B_r", why="사전 장소 bag으로는 대회장 값과 구분 안 됨 (한계 근처 구간 63개뿐)", test="T2 저속 원선회 + T7 레이스 랩", when="새 타이어·새 노면"),
        dict(item="구름·공기 저항 c_roll/c_drag", why="코스트다운 기록 없음", test="T4 코스트다운 (1분)", when="코스트 구간이 많은 전략을 학습할 때"),
        dict(item="질량·요 관성·무게중심", why="bag 으로 식별 불가 (Iz는 서보 지연과 맞바뀜)", test="저울·이중 줄 매달기", when="차를 바꿨을 때"),
        dict(item="서보 창(대칭 s_max 미반영)", why="시뮬 구조가 좌우 대칭만 지원", test="(측정은 끝남: −0.254/+0.412)", when="시뮬 코드 수정 시")],
}
dump("default_comp_car_pre_place", HDR.format(title="기본 프로필 후보: 대회 차 설정의 차 값 + 대회 이전 시험 장소의 장소 값 (직접 녹화된 bag 없음, 조합)"), default)
# ---------- competition car at the competition venue (not default) ----------
comp = {
    "_meta": dict(name="comp_car_map08x23", default_candidate=False, note="대회장(다시 갈 일 없음): 구조 선택·검증용. 기본값으로 쓰지 말 것",
                  bags=COMP, structure="(c2*)", final_check="REPORT.md section 2: 1 s yaw 0.136 vs (a) 0.147, 2 s position 0.45 vs 0.46 m, spin 0 % vs nominal 13 %"),
    "vehicle": {**tyres(1.35, "뒤 최대 마찰: profile 1.1~1.7 (C 1.3 모양 가정 아래); 공통 μ 하한 1.15", "measured", [1.1, 1.7]),
                **{k: v for k, v in STRUCT.items()}, "a_max": veh_c["a_max"], "a_brake": veh_c["a_brake"], **GEOM},
    "actuator": act_c, "lidar": lid_c, "car": car_c, "model_b": MODEL_B, "fingerprint": FP_COMP,
}
dump("comp_car_map08x23", HDR.format(title="대회 차 × 대회장 map08x23 (기본값 아님)"), comp)
# ---------- the pre car at its places ----------
g_p = med(sum(PRE.values(), []), "servo_g"); o_p = med(sum(PRE.values(), []), "servo_o"); er_p = med(sum(PRE.values(), []), "erpm")
imin_p = med(sum(PRE.values(), []), "i_min"); ab_p = med(sum(PRE.values(), []), "a_brake", lambda r: (r["a_brake_s"] or 0) >= 0.3)
lidx_p = med(sum(PRE.values(), []), "lidar_x")[0]
for place, keys in PRE.items():
    pp = PP[place]
    over = pre["1.0"]
    act = dict(act_c)
    act["cmd_delay"] = E(over["cmd_delay"], [0.085, 0.115], "s", tier="car",
                         how="simulator grid on the map12x16 fit bags (pre car), steer_bias refitted at each", note="사전 차 설정: 대회 차보다 약 +57 ms")
    act["steer_bias"] = E(over["steer_bias"], unit="rad", tier="car", how="simulator refit on the map12x16 fit bags")
    ay = med(keys, "ay_held")
    mu_status = "lower_bound"
    note = f"모델 없이 본 하한 {ay[2] / 9.81:.2f} (0.3 s 버틴 최대 a_y {ay[2]:.1f} m/s²); 모델 profile: {pp['mu_verdict']} {pp['mu_interval']}"
    if place == "map16x07":
        note += "; 유일한 실제 스핀(t≈8~9 s)은 μ 1.0 에서 일부 재현, 1.35 에서는 안 됨"
    doc = {
        "_meta": dict(name=f"pre_car_{place}", default_candidate=False, bags=keys,
                      note="사전 차 설정(서보 −0.53, 지연 +57 ms, 회생 −5 A, 자이로 deg/s) × 사전 장소. 차 값은 지금 차가 아님",
                      structure="(c2*) with the pre car's delay / trim / braking"),
        "vehicle": {**tyres(1.0, note, mu_status, [round(ay[2] / 9.81, 3), None]), **{k: v for k, v in STRUCT.items()},
                    "a_brake": E(ab_p[0] if ab_p[0] else 2.0, [ab_p[1], ab_p[2]] if ab_p[0] else None, "m/s^2", "measured", tier="car",
                                 how=f"regen-limited braking at {imin_p[0]:.0f} A, bag_profile median over {ab_p[3]} pre bags"), **GEOM},
        "actuator": act,
        "lidar": {"mount_x": E(med(keys, "lidar_x")[0] - 0.17145, unit="m (ahead of CoG)", source="measured", tier="car",
                               how=f"/tf_static x {med(keys, 'lidar_x')[0]:.3f} minus lr (07-27 bag: 0.297 like the competition car)"),
                  "mount_z": E(0.110, unit="m", source="measured", tier="car")},
        "car": {"servo_map_gain": E(g_p[0], [g_p[1], g_p[2]], "servo/rad", "measured", tier="car"),
                "servo_map_offset": E(o_p[0], [o_p[1], o_p[2]], "servo", "measured", tier="car"),
                "erpm_gain": E(er_p[0], unit="ERPM/(m/s)", source="measured", tier="car"),
                "gyro_unit": E("rad/s" if place == "map16x07" else "deg/s", source="measured", tier="car",
                               how="slope against the localisation heading rate: 52-59 (deg/s bags) or 0.8 (07-27)"),
                "regen_current_min": E(imin_p[0], [imin_p[1], imin_p[2]], "A", "measured", tier="car")},
    }
    dump(f"pre_car_{place}", HDR.format(title=f"사전 차 × 사전 장소 {place} (녹화 그대로의 조합, 기본값 아님)"), doc)
