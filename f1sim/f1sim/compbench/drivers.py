"""Who drives: E2E checkpoints (by alias or path), and the published baselines with a named setting.

A driver string is one of
  - an alias below (`oct`, `scan31b_u475`, ...) or a .pt path: the checkpoint's format picks the engine
    (plan_steps2 -> engine_exam; a contract-v2 `learn.model_v2` checkpoint -> engine_v2)
  - `<kind>@<setting>` for a baseline: kind in ftg / disparity / forzaeth / raceline, setting one of
      k26comp  -- the setting tuned on the neighbouring map korea_2026_competition (baseline_settings.json, 10-19/21)
      tuned    -- the setting tuned on this suite's map by `python -m f1sim.compbench tune` (DATA/tuned/k26scan.json)
      default  -- the method's published defaults (sensor kinds only)
      cand<i>  -- candidate i of a tuning file (tuning only)
"""
import os
import json

from . import REPO, DATA, REPO_FILES

ALIASES = {
    "oct": ("checkpoints/korea26_v2_ppo_s2421_u512.pt",
            "10월 대회 선정 정책 (v2 wp4 계획 + MAP, 2026-10-23, 옛 Korea 2026 지도 korea_2026_competition에서 학습)"),
    "scan31b_u475": ("~/f1sim_runs/pick_scan31b_u475/state_latest.pt",
                     "11-24 예선 정책 (bag-scan 지도, 사용자가 '제일 잘해'라고 함; 콘솔 2번째 줄)"),
    "hs43b_u900": ("~/f1sim_runs/pick_hs43b_u900/state_latest.pt", "11-28 콘솔 '★ 예선 가장 빠름' (ideal 시험으로 고름)"),
    "qual26r_u525": ("~/f1sim_runs/pick_qual26r_u525/state_latest.pt", "11-17/18 예선 정책 (옛 Korea 2026 지도에서 학습, bag-scan 지도는 처음)"),
    "race33b_u975": ("~/f1sim_runs/pick_race33b_u975/state_latest.pt", "11-27 콘솔 '★ 1대1 앞지르기' (본선 세계, 자기 복사본 상대로 학습)"),
    "rc42a_u950": ("~/f1sim_runs/pick_rc42a_u950/state_latest.pt", "11-28 콘솔 '1대1 덜 박음' (본선 세계)"),
    "scan31b2_u600": ("~/f1sim_runs/pick_scan31b2_u600/state_latest.pt", "11-24 scan31b의 다른 시드 계열"),
    "exam4ref_u500": ("~/f1sim_runs/exam4_ref/plan_u500.pt", "scan31b u500: exam4 자기 복사본 상대의 원본"),
}
BASELINE_KINDS = ("ftg", "disparity", "forzaeth", "raceline", "aurorus")
PLANNER_KINDS = ("forzaeth", "raceline", "forzaeth_pred", "lane_switch")
BASELINE_LABELS = {"ftg": "Follow-the-Gap (LiDAR만)", "disparity": "Disparity Extender (LiDAR만)",
                   "forzaeth": "ForzaETH 모듈형 (자기 위치추정·검출·스플라인 회피·MAP)",
                   "raceline": "레이싱라인 + MAP (상자 회피 없음)",
                   "aurorus": "AuRoRus 회피 주행 (자기 선 + LiDAR 회피, 자기 위치는 참값)"}


def tuned_file(map_id: str = "k26scan") -> str:
    return os.path.join(DATA, "tuned", f"{map_id}.json")


def _baseline_setting(kind: str, setting: str, solo: bool) -> dict:
    if setting == "default":
        if kind in PLANNER_KINDS:
            raise ValueError(f"{kind}@default: a planner needs a speed profile; use k26comp or tuned")
        return dict(options={}, source="published defaults")
    if setting == "k26comp":
        from f1sim.learn import baseline_settings as BS
        s = BS.lookup("korea_2026_competition", kind, solo=solo)
        if s is None:
            raise KeyError(f"no korea_2026_competition setting for {kind}")
        s["source"] = f"korea_2026_competition {s.get('source')}"
        return s
    d = json.load(open(tuned_file()))
    if setting == "tuned":
        s = dict(d[kind]["solo" if solo else "race"] if ("solo" in d[kind]) else d[kind]["pick"])
        return s
    if setting.startswith("cand"):
        return dict(d[kind]["candidates"][int(setting[4:])])
    raise ValueError(f"unknown baseline setting {setting!r}")


def resolve(driver: str, solo: bool = True) -> dict:
    """{"id", "engine": "exam" | "v2", "label", and "ckpt" or "ego" (the baseline spec engine_exam builds)}"""
    if "@" in driver and driver.split("@")[0].replace("_gt", "") in BASELINE_KINDS:
        kind, setting = driver.split("@", 1)
        gt = kind.endswith("_gt")                  # the planner on ground truth: what an opponent slot of that kind runs
        kind = kind.replace("_gt", "")
        s = _baseline_setting(kind, setting, solo)
        if kind in PLANNER_KINDS:
            ego = {"kind": kind, "stack_params": s.get("stack_params") or {}, "limits": list(s.get("limits") or [9.0, 7.5, 6.0])}
            if "speed_scale" in s:
                ego["stack_params"] = dict(ego["stack_params"], speed_scale=float(s["speed_scale"]))
            if gt:
                ego["estimated"] = False
        else:
            ego = {"kind": kind, "options": dict(s.get("options") or {}), "weights": s.get("weights") or ""}
        return dict(id=driver.replace("@", "_at_"), engine="exam", ego=ego, setting_source=s.get("source", setting),
                    label=f"{BASELINE_LABELS.get(kind, kind)}{' (참값 GT: 상대 차 자리와 같은 구현)' if gt else ''} [{setting}]")
    path, label = ALIASES.get(driver, (driver, ""))
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(REPO_FILES, p)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"driver {driver}: {p}")
    import torch
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck.get("format") == "plan_steps2":
        eng = "exam"
    elif (ck.get("extra") or {}).get("arch") == "v2" or (ck.get("meta") or {}).get("arch") == "v2":
        eng = "v2"
    else:
        raise ValueError(f"driver {driver}: unknown checkpoint format {ck.get('format')!r}")
    did = driver if driver in ALIASES else os.path.basename(os.path.dirname(p)) + "_" + os.path.splitext(os.path.basename(p))[0]
    upd = ck.get("update") or (ck.get("extra") or {}).get("update")
    return dict(id=did, engine=eng, ckpt=p, label=label or did, update=upd)
