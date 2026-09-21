"""Refresh the panel schema from actual CLI parsers; never starts a training body.

Run with the training Python environment::

    python -m f1sim.viewer.console.training_schema_export [--check]

Only this maintenance command imports learner modules / torch. The panel reads JSON.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TARGETS = {
    "ppo": ("f1sim.learn.ppo", None),
    "dagger": ("f1sim.learn.dagger", None),
    "grip_collect": ("f1sim.learn.policy_grip_data", "collect"),
    "grip_fit": ("f1sim.learn.policy_grip_data", "fit"),
    "grip_final": ("f1sim.learn.policy_grip_data", "final"),
    "grip_pilot": ("f1sim.learn.adaptive_grip_train", None),
}
SCHEMA_PATH = Path(__file__).with_name("training_schema.json")

#: Field labels. The rule, applied throughout: **a term of art keeps its English name** --
#: algorithms, hyper-parameters, architecture pieces and this project's own nouns (rollout,
#: minibatch, GAE, GRU, LiDAR, teacher, raceline, plan, reward, penalty) -- while the words around
#: them that merely say *which* one or *how much* stay Korean. Translating `rollout` to 롤아웃 and
#: `PPO` to 정책 강화학습 makes a field impossible to match to the flag or the paper it came from,
#: which is the only reason anyone reads these labels.
LABELS = {
    "init_estimator": "초기 grip estimator (이어받기)", "replay_data": "TRAIN replay 데이터",
    "replay_fraction": "학습 batch 의 replay 비율",
    "solo_fraction": "단독·무장애물 데이터 비율 (0.3 = 30%)",
    "solo_tracks": "단독 주행 맵 (비우면 선택한 맵의 장애물 제거)",
    "name": "실험 이름", "init": "시작 / 재개 checkpoint", "reference": "원본 D3 reference policy",
    "envs": "병렬 차량 수 (envs)", "tracks": "맵 / 장애물 배치", "obstacle_draws": "맵별 장애물 배치 수",
    "horizon": "rollout 길이 (horizon)", "total": "전체 학습 env step", "epochs": "epoch 수",
    "iters": "DAgger 반복 수", "steps": "수집 step", "batch": "batch 크기",
    "minibatch": "PPO minibatch 크기", "lr": "시작 learning rate", "lr_end": "최종 learning rate",
    "learning_rate": "learning rate", "gamma": "discount γ", "lam": "GAE λ",
    "clip": "PPO clip ε", "ent": "entropy 계수", "vf": "value loss 계수",
    "max_grad": "max grad norm", "sim_backend": "시뮬레이터 실행 방식",
    "controller": "학습 제어기", "adaptation": "controller adaptation", "estimator": "grip estimator",
    "research_estimator": "연구용 미승인 estimator 허용", "fresh_opt": "fresh optimizer 로 시작",
    "cond": "friction conditioning (--cond)", "kl_scope": "reference KL 적용 범위",
    "kl_coef": "reference KL 계수",
    "kl_decay": "KL 감소 step", "cap0": "초기 speed cap (m/s)", "cap1": "최종 speed cap (m/s)",
    "cap_steps": "speed cap 증가 step", "critic_warmup": "critic warmup update",
    "cap_gate": "speed cap 증가 충돌 기준", "cap_gate_quantile": "speed cap 증가 평가 분위수",
    "cap_gate_min_km": "speed cap 증가 최소 평가 거리 (km)", "device": "학습 장치",
    "wandb": "W&B 기록 모드", "wandb_group": "W&B 그룹", "wandb_id": "W&B 재개 ID",
    "wandb_new": "새 W&B 실행 생성", "log_every": "로그 주기", "save_every": "checkpoint 저장 주기",
    "seed": "random seed", "amp": "mixed precision (bf16)", "metrics_jsonl": "지표 JSONL 출력 파일",
    "steps_base": "재개 누적 step 기준", "yield_to_viewer": "시각화 실행 중 GPU 양보 비율",
    "progress_reward": "progress reward (m당)", "alive_reward": "alive reward (제어 step당)",
    "collision_penalty": "벽 충돌 penalty", "steer_penalty": "조향 변화 penalty",
    "proximity_penalty": "벽 근접 penalty", "safe_dist": "벽 안전 거리 (m)",
    "wrong_way_penalty": "역주행 penalty", "collision_speed_penalty": "충돌 속도 penalty",
    "proximity_speed_ref": "근접 penalty 속도 기준 (m/s)", "plan_clearance_penalty": "plan 여유 penalty",
    "plan_margin": "plan 벽 여유 (m)", "lap_time_bonus": "랩타임 bonus", "lap_bonus": "완주 bonus",
    "overtake_bonus": "추월 bonus", "overtake_sustained": "선두 유지 거리 · 시간 · bonus",
    "car_safe_gap": "상대차 안전 간격 (m)", "car_proximity_penalty": "상대차 근접 penalty",
    "car_contact_penalty": "상대차 접촉 penalty", "sideslip_penalty": "sideslip penalty",
    "ttc_penalty": "TTC penalty", "ttc_safe": "TTC 기준 (s)",
    "episode_s": "episode 길이 (s)", "scan_stack": "LiDAR 누적 frame", "scan_stride": "LiDAR frame 간격",
    "hist_len": "차량 상태 이력 길이", "action_mode": "action space", "memory": "policy memory",
    "memory_hidden": "policy GRU hidden 크기", "memory_critic": "critic memory",
    "scan_channels": "LiDAR 추가 channel",
    "scan_memory_tau": "scan memory 시간 (s)", "scan_deltas": "LiDAR 시간 차분 입력",
    "temporal_encoder": "temporal encoder", "scan_stem": "LiDAR encoder (scan stem)",
    "frontend": "frontend checkpoint",
    "floor_att": "노면 자세 입력", "opp_token": "상대차 상태 입력", "opp_future_model": "상대차 미래 예측",
    "procedural_obstacles": "procedural 장애물 episode 비율", "procedural_density": "장애물 밀도 (10m당)",
    "procedural_max_props": "장애물 최대 개수", "procedural_raceline_margin": "장애물 raceline 여유 (m)",
    "procedural_raceline_corridor": "raceline 위 장애물 허용 (corridor off)",
    "spawn_runway": "spawn 활주로 (m)",
    "collision_mode": "충돌 처리 (terminate / soft)",
    "raceline_margin": "raceline 경계 여유 (m)", "teacher_grip": "teacher 마찰 정보",
    "teacher_recover_time": "teacher 방향 복구 시간 (s)", "teacher_kind": "teacher 종류",
    "teacher_speed": "teacher 속도 배율", "teacher_horizon": "teacher 예측 시간 (s)",
    "teacher_cand_iters": "teacher 후보 개선 반복", "teacher_offsets": "teacher 횡방향 후보 (m)",
    "teacher_speeds": "teacher 속도 후보 배율", "teacher_cost": "teacher cost 가중치 5종",
    "beta0": "초기 teacher 실행 비율 (beta)", "speed_cap": "최대 주행 속도 (m/s)",
    "speed_loss": "speed imitation loss", "chunk_length": "GRU chunk length",
    "hard_frac": "hard sample 비율",
    "hard_power": "hard sample 가중 지수", "keep_iters": "DAgger 데이터 보관 반복 수",
    "start_iter": "DAgger 재개 반복 번호", "eval_steps": "평가 step", "eval_every": "평가 주기",
    "eager": "컴파일 없이 실행 (eager)", "race_size": "레이스당 차량 수", "opponent": "상대차 기본 종류",
    "mixed_teacher_frac": "혼합 상대차 teacher 비율", "opp_speed": "상대차 속도 배율 범위",
    "opp_pool": "상대 policy pool", "opp_slots": "차량별 상대차 설정", "spawn_gap": "출발 종방향 간격 (m)",
    "spawn_order": "학습 차량 출발 위치", "spawn_alongside_sep": "나란한 차량 횡간격 (m)",
    "spawn_alongside_gap": "나란한 차량 종간격 (m)", "opp_events": "상대차 행동 event",
    "opp_event_rate": "상대차 event 빈도 (10s당)", "out": "결과 출력 경로", "data": "수집 데이터 폴더",
    "policy": "데이터 수집 policy", "candidate": "평가할 estimator", "split_seeds": "학습·보정·개발·최종 seed",
    "jobs_per_split": "분할별 수집 작업 수", "stride": "샘플 간격", "chunk_steps": "저장 chunk 길이",
    "batch_size": "batch 크기", "hidden": "estimator GRU hidden 크기", "train_episodes": "학습 episode 수",
    "cal_episodes": "보정 episode 수", "test_episodes": "평가 episode 수",
    # -- the grip dial and its budget (docs/research/mintime-teacher-speed-head-2026-09-19.md)
    "dial_margin": "dial margin (학습 중 무작위 하향폭)",
    "dial_exact": "dial 을 정확히 알려줄 비율",
    "grip_budget_penalty": "grip budget penalty (dial 초과)",
    "grip_quantile": "grip pinball τ",
    "speed_mode": "plan 속도 표현 (speed mode)",
    # -- the teacher's own limits, which decide the line it drives
    "raceline_objective": "raceline 목적함수",
    "teacher_a_lat": "teacher 횡가속 한계 (m/s²)",
    "teacher_a_acc": "teacher 구동가속 한계 (m/s²)",
    "teacher_a_brake": "teacher 제동 한계 (m/s²)",
    # -- recurrence and the auxiliary heads
    "seq_len": "BPTT 구간 길이 (step)", "seq_burn": "BPTT warm-up step",
    "init_log_std": "초기 action log_std", "name_seed_fresh": "모듈별 이름 기반 재초기화",
    "aux_grip": "aux head · 노면 가중치", "aux_opp": "aux head · 상대차 가중치",
    "aux_opp_mask": "aux head · 상대차 mask 가중치", "aux_floor": "aux head · 노면 자세 가중치",
    "aux_floor_width": "aux head · 노면 자세 폭", "aux_future": "aux head · 미래 예측 가중치",
    "aux_future_k": "aux head · 미래 예측 step 수", "aux_future_width": "aux head · 미래 예측 폭",
    "aux_motion": "aux head · motion 가중치",
    "motion_memory": "motion memory 사용", "motion_channels": "motion channel",
    "motion_hidden": "motion GRU hidden 크기", "motion_critic": "critic 에도 motion 적용",
    "critic_priv_adapter": "critic privileged adapter",
    "learning_rate": "learning rate (추정기)",
    "sideslip_penalty": "sideslip (슬립각) penalty",
    "temporal_encoder": "temporal encoder 종류",
}
PATHS = {"init_estimator": "input_file", "replay_data": "input_file", "init": "input_file", "reference": "input_file", "estimator": "input_file",
         "frontend": "input_file", "policy": "input_file", "candidate": "input_file",
         "data": "input_dir", "out": "output_dir", "metrics_jsonl": "output_file"}
CSV_LISTS = {"scan_channels": "str", "opp_events": "str", "opp_pool": "path",
             "teacher_offsets": "float", "teacher_speeds": "float", "teacher_cost": "float",
             "split_seeds": "int"}
REWARDS = {"progress_reward", "alive_reward", "collision_penalty", "steer_penalty", "proximity_penalty", "safe_dist", "wrong_way_penalty",
           "collision_speed_penalty", "proximity_speed_ref", "plan_clearance_penalty", "plan_margin",
           "lap_time_bonus", "lap_bonus", "overtake_bonus", "overtake_sustained", "car_safe_gap",
           "sideslip_penalty", "ttc_penalty", "ttc_safe", "car_proximity_penalty", "car_contact_penalty"}


def _group(dest):
    if dest in REWARDS:
        return "PPO 보상"
    if dest in {"name", "init", "init_estimator", "reference", "policy", "candidate", "data", "out", "seed"}:
        return "기본"
    if dest.startswith(("teacher", "opp_", "spawn_", "mixed_")) or dest == "opponent":
        return "teacher·상대차"
    if dest in {"envs", "tracks", "obstacle_draws", "race_size", "episode_s", "speed_cap", "raceline_margin", "solo_fraction", "solo_tracks"} or dest.startswith("procedural_"):
        return "환경"
    if dest.startswith(("scan_", "memory", "motion_", "aligned_")) or dest in {"action_mode", "temporal_encoder", "adaptation", "controller", "estimator", "research_estimator", "cond", "hist_len", "frontend", "floor_att", "critic_priv_adapter"}:
        return "모델·제어"
    if dest.startswith("wandb") or dest in {"device", "sim_backend", "amp", "eager", "log_every", "save_every", "metrics_jsonl", "yield_to_viewer"}:
        return "출력/실행"
    if dest.startswith(("aux_", "name_seed")):
        return "고급"
    return "학습"


class _Captured(BaseException):
    def __init__(self, parser):
        self.parser = parser


def capture_parser(module_name):
    """Intercept before any main-body parsing, device creation, dataset writes or training."""
    module = importlib.import_module(module_name)
    original_strftime = module.time.strftime if hasattr(module, "time") else None

    def capture(parser, *args, **kwargs):
        raise _Captured(parser)

    def fixed_time(fmt, *args):
        return "{timestamp}" if fmt == "%m%d_%H%M" else original_strftime(fmt, *args)

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            if original_strftime:
                with patch.object(module.time, "strftime", fixed_time):
                    module.main()
            else:
                module.main()
        except _Captured as found:
            return found.parser
    raise RuntimeError(f"{module_name}.main did not expose its ArgumentParser")


def _json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json(x) for x in value]
    return value


def field_from_action(action):
    actions = {argparse._StoreAction: "store", argparse._StoreTrueAction: "store_true",
               argparse._StoreFalseAction: "store_false", argparse._AppendAction: "append"}
    if type(action) not in actions:
        raise TypeError(f"unsupported parser action {type(action).__name__}: {action.dest}")
    dtype = {None: "str", str: "str", int: "int", float: "float", Path: "path"}.get(action.type)
    if dtype is None:
        raise TypeError(f"unsupported parser type {action.type}: {action.dest}")
    dest = action.dest
    item_kind = dtype
    kind = "bool" if actions[type(action)] in ("store_true", "store_false") else dtype
    if action.choices is not None:
        kind = "choice"
    separator = None
    if kind != "bool" and (action.nargs is not None or isinstance(action, argparse._AppendAction)):
        kind, separator = "list", "space"
    if dest in CSV_LISTS:
        kind, item_kind, separator = "list", CSV_LISTS[dest], "comma"
    if dest in PATHS:
        kind = "path"
    if dest == "opp_slots":
        kind = "json"
    default = _json(action.default)
    result = dict(flag=action.option_strings[0], flags=action.option_strings, dest=dest,
                  label=LABELS.get(dest, dest.replace("_", " ")), kind=kind, item_kind=item_kind,
                  default=default, nargs=action.nargs, choices=_json(action.choices),
                  required=action.required, group=_group(dest), help=action.help or "",
                  action=actions[type(action)], nullable=action.default is None,
                  metavar=_json(action.metavar))
    if dest == "solo_fraction":
        result["help"] = "학습 차량이 수집한 Teacher 라벨 개수 기준입니다. 0.3은 약 30%를 상대차·배치 장애물 없는 환경에서 수집합니다. 스텝 반올림 후 실제 비율은 실행 기록에 남습니다."
    elif dest == "solo_tracks":
        result["help"] = "비워 두면 위에서 선택한 맵을 사용합니다. 따로 지정한 맵도 배치 장애물을 모두 제거하고 상대차 없이 수집합니다."
    if separator:
        result["separator"] = separator
    if dest in PATHS:
        result["path_mode"] = PATHS[dest]
    if isinstance(default, str) and "{timestamp}" in default:
        result["default_factory"] = "timestamp"
    return result


def _privileged_contract(cfg, env_cfg):
    """Measure the actual critic packing and registered adapters on tiny CPU tensors."""
    import torch
    from ... import dynamics
    from ...gym_env import F1VecEnv
    from ...learn.conditioning import PRIV_ADAPTERS

    zeros = torch.zeros(2)
    track = SimpleNamespace(length=torch.ones(1),
                            pose_at_s=lambda s, tid: (torch.zeros(len(s), 2), torch.zeros_like(s)))
    sim = SimpleNamespace(track=track, tid=torch.zeros(2, dtype=torch.long),
                          control_dt=1 / cfg.sim.control_rate,
                          other_idx=torch.tensor([[1], [0]]),
                          P={name: torch.ones(2) for name in F1VecEnv.PRIV_PARAMS})
    env = SimpleNamespace(B=2, device=torch.device("cpu"), ecfg=env_cfg, sim=sim,
                          speed_cap=torch.ones(2), PRIV_PARAMS=F1VecEnv.PRIV_PARAMS)
    env._priv = lambda result: F1VecEnv._priv(env, result)
    result = SimpleNamespace(state=torch.zeros(2, dynamics.STATE_DIM), s=zeros,
                             lateral=zeros, progress=zeros, wall_dist=torch.ones(2))
    dims = {}
    for name, cars in (("solo", 1), ("race", 2)):
        env.M = cars
        dims[name] = F1VecEnv.privileged(env, result).shape[1]
    adapters = {}
    for name, adapter in PRIV_ADAPTERS.items():
        matches = []
        for width in sorted(set(dims.values())):
            try:
                output = adapter(torch.zeros(1, width))
            except ValueError:
                continue
            matches.append({"input": width, "output": output.shape[1]})
        if len(matches) != 1:
            raise ValueError(f"privileged adapter {name} needs an explicit UI shape contract: {matches}")
        adapters[name] = matches[0]
    return dims, adapters


def _observation_contract(common, obs):
    """Trainer observation fields that the current CLI cannot override."""
    from ...gym_env import EnvConfig
    from ...params import Config
    from ...mpc import ACT_DIM
    from ...learn.memory import memory_spec

    cfg, env_cfg = Config(), EnvConfig()
    spec = common.obs_spec(SimpleNamespace(
        ecfg=env_cfg, n_beams=len(range(0, cfg.lidar.n_beams, env_cfg.scan_subsample)),
        range_max=cfg.lidar.range_max, act_dim=obs.ObsSpec().act_dim,
        opp_token=env_cfg.opp_token))
    fixed_names = ("n_beams", "range_max", "v_max", "gyro_scale", "accel_scale",
                   "att_scale", "action_history", "hist_stride")
    restorable_names = ("scan_stack", "scan_stride", "hist_len", "opp_token", "opp_future_model")
    memory = memory_spec()
    privileged_dims, priv_adapters = _privileged_contract(cfg, env_cfg)
    return {"fixed": {name: getattr(spec, name) for name in fixed_names},
            "restorable": {name: name for name in restorable_names},
            "action_dims": {"direct": spec.act_dim, "plan": ACT_DIM},
            "memory": {name: memory[name] for name in ("layers", "critic")},
            "privileged_dims": privileged_dims, "priv_adapters": priv_adapters}


def export_schema():
    documents = {}
    parsers = {}
    for key, (module, subcommand) in TARGETS.items():
        if module not in parsers:
            parsers[module] = capture_parser(module)
        parser = parsers[module]
        if subcommand:
            subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
            parser = subparsers.choices[subcommand]
        fields = [field_from_action(a) for a in parser._actions if not isinstance(a, argparse._HelpAction)]
        if key == "dagger":
            from ...interactive_teacher import TeacherCost
            names = [f.name for f in dataclasses.fields(TeacherCost)]
            labels = {"progress": "진행", "wall": "벽", "opp": "상대차",
                      "clear": "여유", "smooth": "부드러움"}
            cost = TeacherCost()
            next(f for f in fields if f["dest"] == "teacher_cost").update(
                ui_length=len(names), component_names=names,
                component_labels=[labels.get(name, name) for name in names],
                component_defaults=[getattr(cost, name) for name in names])
        if key == "grip_final":
            next(f for f in fields if f["dest"] == "out")["path_mode"] = "output_file"
        documents[key] = dict(key=key, module=module, subcommand=subcommand, fields=fields)
    from ...learn import opponent_config, common, obs
    from ... import opponent_events, opponent_slots
    contracts = dict(slot_superseded=_json(opponent_config.SLOT_SUPERSEDED),
                     scan_channels=list(obs.SCAN_CHANNELS), aligned_channels=list(obs.ALIGNED_CHANNELS),
                     timed_events=list(opponent_events.EVENT_NAMES), reactive_events=list(opponent_events.REACTIVE_NAMES),
                     reactive_prob_fields=opponent_events.REACTIVE_PROB_FIELD,
                     slot_fields=[f.name for f in dataclasses.fields(opponent_slots.OpponentSlot)],
                     slot_kinds=[dict(name=k.name, teacher=k.teacher, checkpoint=k.checkpoint) for k in opponent_slots.KINDS],
                     slot_grip=list(opponent_slots.GRIP_LABELS), slot_spawns=list(opponent_slots.SLOT_SPAWNS),
                     train_tracks=list(common.TRAIN_TRACKS),
                     observation=_observation_contract(common, obs))
    return dict(format=1, schemas=documents, contracts=contracts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = export_schema()
    if args.check:
        if json.loads(SCHEMA_PATH.read_text()) != result:
            raise SystemExit("training schema drift: regenerate with python -m f1sim.viewer.console.training_schema_export")
        print("training schema matches all CLI actions")
    else:
        SCHEMA_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {sum(len(s['fields']) for s in result['schemas'].values())} fields: {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
