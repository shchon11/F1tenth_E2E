"""Torch-free training form contract, validation and argv construction.

The sibling JSON is generated from the learner ArgumentParsers, not a selected list
of UI flags. Regenerate it with ``training_schema_export`` when a parser changes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _document():
    return json.loads(Path(__file__).with_name("training_schema.json").read_text(encoding="utf-8"))


def get_schema(kind: str) -> dict:
    """Return an independent schema; the run-name timestamp is evaluated now."""
    try:
        result = copy.deepcopy(_document()["schemas"][kind])
    except KeyError:
        raise ValueError(f"알 수 없는 학습 종류: {kind}") from None
    for field in result["fields"]:
        if field.get("default_factory") == "timestamp":
            field["default"] = field["default"].replace("{timestamp}", datetime.now().strftime("%m%d_%H%M"))
    return result


def kinds() -> tuple[str, ...]:
    return tuple(_document()["schemas"])


def get_observation_contract(kind: str) -> dict:
    """Fixed observation normalizers and restorable fields for policy trainers.

    Generated from the actual trainer defaults; reading this does not import torch.
    A checkpoint with a different fixed field cannot be resumed by these CLIs.
    """
    if kind not in ("ppo", "dagger"):
        raise ValueError(f"정책 관측 계약이 없는 학습 종류: {kind}")
    return copy.deepcopy(_document()["contracts"]["observation"])


def _scalar(value, kind):
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError("켜짐 / 꺼짐 값이 필요합니다")
        return value
    if kind in ("int", "float"):
        if isinstance(value, bool):
            raise ValueError("숫자가 필요합니다")
        if kind == "int":
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    pass
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("유한한 숫자가 필요합니다")
        if kind == "int" and number != int(number):
            raise ValueError("정수가 필요합니다")
        return int(number) if kind == "int" else number
    if not isinstance(value, (str, Path)):
        raise ValueError("문자열이 필요합니다")
    if "\x00" in str(value):
        raise ValueError("NUL 문자는 사용할 수 없습니다")
    return str(value)


def _json_finite(value):
    if isinstance(value, str) and "\x00" in value:
        raise ValueError("NUL 문자는 사용할 수 없습니다")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON 안의 숫자는 유한해야 합니다")
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("JSON 객체의 키는 문자열이어야 합니다")
        for v in value.values():
            _json_finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _json_finite(v)


def _coerce(field, value, *, resolve_files=True):
    if value is None:
        if field["default"] is not None:
            # None explicitly omits an optional control, so it inherits the CLI default.
            value = field["default"]
        else:
            return None
    kind = field["kind"]
    if kind == "list":
        if isinstance(value, str):
            value = value.strip()
            value = ([x.strip() for x in value.split(",")] if field["separator"] == "comma" else value.replace(",", " ").split()) if value else []
        if not isinstance(value, (list, tuple)):
            raise ValueError("값 목록이 필요합니다")
        result = [_scalar(v, field["item_kind"]) for v in value]
        nargs = field["nargs"]
        if isinstance(nargs, int) and len(result) != nargs:
            raise ValueError(f"값 {nargs}개가 필요합니다")
        if nargs == "+" and not result:
            raise ValueError("한 개 이상의 값이 필요합니다")
        return result
    if kind == "json":
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            if text.startswith("@"):
                if not resolve_files:
                    return _scalar(text, "str")
                text = Path(text[1:]).expanduser().read_text(encoding="utf-8")
            value = json.loads(text)
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, (list, tuple)) or not value or not all(isinstance(v, dict) for v in value):
            raise ValueError("상대차 JSON 객체의 비어 있지 않은 목록이 필요합니다")
        _json_finite(value)
        return list(value)
    result = _scalar(value, field["item_kind"] if kind == "choice" else kind)
    if field["choices"] is not None and result not in field["choices"]:
        raise ValueError(f"허용된 값: {field['choices']}")
    return result


def _values(schema, values, *, resolve_files=True):
    if not isinstance(values, dict):
        return {}, ["설정은 키와 값의 객체여야 합니다"]
    fields = {f["dest"]: f for f in schema["fields"]}
    errors = [f"지원하지 않는 설정: {key}" for key in values if key not in fields]
    result = {}
    for dest, field in fields.items():
        try:
            result[dest] = _coerce(field, values.get(dest, field["default"]), resolve_files=resolve_files)
        except (TypeError, ValueError, OSError) as exc:
            errors.append(f"{field['label']} ({field['flag']}): {exc}")
    return result, errors


def coerce_values(kind: str, values: dict) -> dict:
    """Normalize portable settings before UI mutation; no launch or path checks.

    Defaults are filled in, while missing required values stay unset. An ``@file``
    slot table stays a reference until launch validation resolves its contents.
    """
    result, errors = _values(get_schema(kind), values, resolve_files=False)
    if errors:
        raise ValueError("\n".join(errors))
    return copy.deepcopy(result)


def _path_error(value, mode):
    path = Path(value).expanduser()
    if mode == "input_file" and not path.is_file():
        return f"입력 파일이 없습니다: {path}"
    if mode == "input_dir" and not path.is_dir():
        return f"입력 폴더가 없습니다: {path}"
    if mode == "output_dir" and path.exists() and not path.is_dir():
        return f"출력 폴더 자리에 파일이 있습니다: {path}"
    if mode == "output_file" and path.exists() and not path.is_file():
        return f"출력 파일 자리에 폴더가 있습니다: {path}"
    return None


def _validate_slots(slots, race, errors):
    contract = _document()["contracts"]
    if len(slots) != race - 1:
        errors.append("차량별 상대차 행 수는 레이스당 차량 수 - 1이어야 합니다 (--opp-slots / --race-size)")
    for index, slot in enumerate(slots, 1):
        prefix = f"상대차 {index}"
        extra = set(slot) - set(contract["slot_fields"])
        if extra:
            errors.append(f"{prefix}: 알 수 없는 설정 {sorted(extra)}")
        kind = next((k for k in contract["slot_kinds"] if k["name"] == slot.get("kind", "raceline")), None)
        if not kind:
            errors.append(f"{prefix}: 알 수 없는 상대차 종류")
            continue
        checkpoint = slot.get("checkpoint")
        if kind["checkpoint"]:
            if not isinstance(checkpoint, (str, Path)) or not checkpoint or not Path(checkpoint).expanduser().is_file():
                errors.append(f"{prefix}: 정책 체크포인트 파일이 필요합니다")
        elif checkpoint:
            errors.append(f"{prefix}: 정책 상대차만 체크포인트를 사용합니다")
        if slot.get("label_grip", "true") not in contract["slot_grip"] or slot.get("spawn", "ahead") not in contract["slot_spawns"]:
            errors.append(f"{prefix}: 마찰 정보 또는 출발 위치가 잘못되었습니다")
        try:
            scale = slot.get("speed_scale", 1.)
            lo, hi = scale if isinstance(scale, (list, tuple)) else (scale, scale)
            if not (math.isfinite(float(lo)) and math.isfinite(float(hi)) and 0 < float(lo) <= float(hi)):
                raise ValueError("속도 배율은 0 < 최소 <= 최대여야 합니다")
            if slot.get("speed_cap") is not None and not (math.isfinite(float(slot["speed_cap"])) and float(slot["speed_cap"]) > 0):
                raise ValueError("속도 제한은 양수여야 합니다")
            rate = float(slot.get("event_rate", 0.))
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("이벤트 빈도는 0 이상이어야 합니다")
            events = slot.get("events") or []
            if isinstance(events, str):
                events = [x.strip() for x in events.split(",") if x.strip()]
            if not set(events) <= set(contract["timed_events"] + contract["reactive_events"]):
                raise ValueError("알 수 없는 이벤트")
            timed = set(events) & set(contract["timed_events"])
            if bool(timed) != (rate > 0):
                raise ValueError("주기적 이벤트와 양수 빈도를 함께 지정해야 합니다")
            reactive = slot.get("reactive") or {}
            if not isinstance(reactive, dict) or not set(reactive) <= set(contract["reactive_events"]):
                raise ValueError("알 수 없는 반응 행동")
            if any(not 0 <= float(p) <= 1 for p in reactive.values()):
                raise ValueError("반응 행동 확률은 0~1이어야 합니다")
            if not kind["teacher"] and (events or any(float(p) > 0 for p in reactive.values()) or slot.get("label_grip", "true") != "true"):
                raise ValueError("Teacher 상대차만 이벤트와 마찰 정보를 사용합니다")
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")


def _cross_validate(kind, a, errors):
    def require(condition, message):
        if not condition:
            errors.append(message)

    contract = _document()["contracts"]
    # Positive sizes are necessary to construct batches and schedules. Seeds, offsets,
    # rewards and research coefficients are deliberately not clamped to arbitrary ranges.
    positive = ("envs", "race_size", "horizon", "total", "epochs", "minibatch", "batch", "iters",
                "steps", "learning_rate", "batch_size", "hidden", "memory_hidden",
                "scan_stack", "scan_stride", "episode_s", "chunk_length", "keep_iters", "log_every",
                "save_every", "eval_steps", "eval_every", "stride", "chunk_steps", "jobs_per_split")
    for key in positive:
        if key in a:
            require(a[key] > 0, f"--{key.replace('_', '-')} 값은 양수여야 합니다")
    for key in ("lr", "lr_end"):
        if key in a:
            require(a[key] >= 0, f"--{key.replace('_', '-')} 값은 0 이상이어야 합니다")
    race = a.get("race_size", 1)
    if race > 0 and "envs" in a:
        require(a["envs"] % race == 0, "--envs는 --race-size의 배수여야 합니다")
    if kind in ("ppo", "dagger"):
        require(bool(a["tracks"]), "주행 맵을 선택하세요 (--tracks)")
        for key in ("spawn_gap", "opp_speed"):
            lo, hi = a[key]
            require(0 < lo <= hi, f"--{key.replace('_', '-')} 범위는 0 < 최소 <= 최대여야 합니다")
        require(set(a["scan_channels"]) <= set(contract["scan_channels"]), "알 수 없는 LiDAR 추가 채널 (--scan-channels)")
        slots = a["opp_slots"]
        if slots:
            _validate_slots(slots, race, errors)
            for flag, dest, default in contract["slot_superseded"]:
                actual = a[dest]
                comparable = [] if default == "" and isinstance(actual, list) else default
                if actual != comparable:
                    errors.append(f"--opp-slots와 {flag} 전역 설정은 함께 사용할 수 없습니다")
        else:
            pool = a["opp_pool"]
            require(not pool or a["opponent"] == "pool", "상대 정책 풀은 --opponent pool에서만 사용합니다")
            if a["opponent"] == "pool":
                require(race > 1 and bool(pool), "정책 풀 상대차에는 2대 이상 차량과 비어 있지 않은 풀이 필요합니다")
            for entry in pool:
                if entry not in ("self", "teacher"):
                    require(Path(entry).expanduser().is_file(), f"상대 정책 파일이 없습니다: {entry}")
            events = a["opp_events"]
            require(set(events) <= set(contract["timed_events"] + contract["reactive_events"]), "알 수 없는 상대차 이벤트 (--opp-events)")
            teacher = race > 1 and (a["opponent"] in ("teacher", "mixed") or (a["opponent"] == "pool" and "teacher" in pool))
            require(not events or teacher, "상대차 이벤트에는 Teacher가 운전하는 상대차가 필요합니다")
            if set(events) & set(contract["timed_events"]):
                require(a["opp_event_rate"] > 0, "주기적 상대차 이벤트의 빈도는 양수여야 합니다")
            for event in set(events) & set(contract["reactive_events"]):
                require(0 < a[contract["reactive_prob_fields"][event]] <= 1, f"{event} 반응 확률은 0보다 크고 1 이하여야 합니다")
            require(a["spawn_order"] == "behind" or race > 1, "출발 순서 변경에는 상대차가 필요합니다")
        if a["opp_token"] != "off":
            require(race > 1, "상대차 상태 입력에는 2대 이상 차량이 필요합니다")
    if kind == "dagger":
        solo = a.get("solo_fraction", 0.)
        require(0 <= solo < 1, "단독·무장애물 데이터 비율은 0 이상 1 미만입니다")
        require(not a.get("solo_tracks") or solo > 0, "별도 단독 주행 맵에는 양수 데이터 비율이 필요합니다")
        if solo > 0:
            solo_steps = math.floor(a["steps"] * solo + .5)
            require(0 < solo_steps < a["steps"], "수집 스텝이 너무 작아 단독·교통 데이터를 함께 수집할 수 없습니다")
            require(race >= 2 and a["teacher_kind"] == "interactive" and a["action_mode"] == "plan",
                    "단독·교통 혼합 수집에는 2대 이상 레이스와 Interactive plan Teacher가 필요합니다")
            require(a["opp_token"] == "off", "단독·교통 혼합 수집에서는 상대차 상태 입력을 끄세요")
        if a["teacher_kind"] == "interactive":
            require(a["action_mode"] == "plan" and race > 1, "Interactive teacher에는 plan 행동과 상대차가 필요합니다")
        require(not (a["memory"] != "off" and a["hard_frac"] > 0), "GRU 학습과 --hard-frac은 함께 사용할 수 없습니다")
        require(a["start_iter"] >= 0, "재개 반복 번호는 0 이상이어야 합니다")
        require(a["start_iter"] == 0 or bool(a["init"]), "DAgger 재개에는 시작 체크포인트가 필요합니다")
        require(not a["teacher_cost"] or len(a["teacher_cost"]) == 5, "Teacher 비용 가중치는 5개입니다")
    if kind == "ppo":
        adaptive = a["adaptation"] != "off"
        if adaptive:
            require(a["controller"] == "auto" and bool(a["init"]) and a["memory"] == "gru", "controller adaptation 에는 auto 제어기, 시작 checkpoint 와 GRU 가 필요합니다")
            require(a["cond"] == "none" and a["action_mode"] == "plan" and a["opp_token"] == "off", "controller adaptation 은 plan action 과 일반 관측을 사용합니다 (--cond none / --opp-token off)")
            require(a["init_log_std"] is None, "controller adaptation 에서는 기존 action noise 를 유지합니다 (--init-log-std 비우기)")
            if race > 0 and a["horizon"] > 0 and a["envs"] >= race:
                quantum = a["horizon"] * (a["envs"] // race)
                require(a["total"] == int(a["total"]) and int(a["total"]) % quantum == 0, "controller adaptation 의 전체 step 은 horizon × 학습 차량 수의 배수여야 합니다")
        else:
            require(not a["reference"] and not a["research_estimator"] and a["kl_scope"] is None, "reference policy · 연구용 estimator · KL scope 는 controller adaptation 모드에서 설정하세요")
            require(not ((a["memory"] != "off" or a["scan_channels"]) and a["controller"] != "legacy"), "controller adaptation 밖에서 GRU / 추가 channel 은 legacy 제어기를 사용합니다")
        if a["memory"] != "off":
            require(a["cond"] == "none", "GRU 와 friction conditioning 을 함께 사용할 수 없습니다")
            require(a["minibatch"] >= a["horizon"], "GRU minibatch 크기는 rollout 길이 이상이어야 합니다")
        if a["motion_memory"] or a["aux_opp_mask"] > 0 or a["aux_motion"] > 0:
            require(a["memory"] == "gru" and a["motion_memory"] and bool(set(a["scan_channels"]) & set(contract["aligned_channels"])), "motion 학습에는 GRU, motion memory, 정렬된 LiDAR channel 이 필요합니다")
        fraction = a["procedural_obstacles"]
        require(0 <= fraction <= 1, "procedural 장애물 비율은 0~1 입니다")
        if fraction:
            require(a["procedural_density"] > 0, "절차적 장애물 밀도는 양수여야 합니다")
        else:
            require(a["procedural_density"] == 1 and a["procedural_max_props"] == 0 and a["procedural_raceline_margin"] == .25, "장애물 밀도·개수·여유 변경에는 procedural 장애물을 활성화하세요")
        if a["opp_token"] != "off":
            require(a["action_mode"] == "plan", "상대차 상태 입력에는 plan 행동이 필요합니다")
        if a["overtake_sustained"] is not None:
            distance, seconds, bonus = a["overtake_sustained"]
            require(distance > 0 and seconds > 0 and (bonus <= 0 or race > 1), "선두 유지 거리·시간은 양수이며 보너스에는 상대차가 필요합니다")
        if a["ttc_penalty"] > 0:
            require(race > 1 and a["ttc_safe"] > 0, "TTC 보상에는 상대차와 양수 시간 기준이 필요합니다")
        else:
            require(a["ttc_safe"] == 1., "TTC 시간 기준 변경에는 TTC 패널티를 활성화하세요")
        require(a["overtake_bonus"] <= 0 or race > 1, "추월 보너스에는 상대차가 필요합니다")
    if kind == "grip_collect":
        require(a["steps"] >= 8, "마찰 데이터 수집은 8 스텝 이상 필요합니다")
        from ... import tracks
        selected = contract["train_tracks"] if a["tracks"].strip() == "train" else [x.strip() for x in a["tracks"].split(",") if x.strip()]
        try:
            bases = {tracks.parse(name).track for name in selected}
            allowed = {tracks.parse(name).track for name in contract["train_tracks"]}
            require(len(bases) >= 4 and bases <= allowed, "분할 누출을 막기 위해 TRAIN 목록에서 서로 다른 기본 맵을 4개 이상 선택하세요")
        except ValueError as exc:
            errors.append(f"맵 설정: {exc}")
        require(len(a["split_seeds"]) == 4 and len(set(a["split_seeds"])) == 4 and min(a["split_seeds"], default=-1) >= 0, "수집 분할 시드는 서로 다른 0 이상 정수 4개여야 합니다")
        seeds = sorted(a["split_seeds"])
        require(all(right - left >= a["jobs_per_split"] for left, right in zip(seeds, seeds[1:])),
                "수집 분할 시드 범위가 겹칩니다: 인접한 분할 시드는 --jobs-per-split 이상 떨어져야 합니다")
    if kind == "grip_pilot":
        require(1 <= a["epochs"] <= 10 and 40 <= a["steps"] <= 240, "제어된 파일럿은 1~10 에포크, 40~240 스텝 범위입니다")
        require(min(a["train_episodes"], a["cal_episodes"], a["test_episodes"]) >= 8, "파일럿 각 분할은 8 에피소드 이상 필요합니다")
        if a["out"]:
            out = Path(a["out"]).expanduser()
            require(not out.exists() or not any(out.iterdir()), "파일럿 출력 폴더는 비어 있어야 합니다")
    if kind == "grip_fit":
        out = Path(a["out"]).expanduser()
        require(not out.exists() or not any(out.iterdir()), "추정기 학습 출력 폴더는 비어 있어야 합니다")
        if "replay_fraction" in a:
            if a["replay_data"]:
                require(0 < a["replay_fraction"] < 1, "재생 데이터 비율은 0보다 크고 1보다 작아야 합니다")
                require(0 < round(a["batch_size"] * a["replay_fraction"]) < a["batch_size"], "배치에 현재 TRAIN과 재생 TRAIN 샘플이 각각 한 개 이상 필요합니다")
            else:
                require(a["replay_fraction"] == .5, "재생 비율 변경에는 재생 데이터가 필요합니다")
    if kind == "grip_final" and a["out"]:
        require(not Path(a["out"]).expanduser().exists(), "최종 평가 보고서를 덮어쓸 수 없습니다; 새 출력 파일을 선택하세요")
        require(not (Path(a["data"]).expanduser() / "final_opened.json").exists(), "이미 최종 평가에 사용한 데이터입니다; 최종 평가는 한 번만 실행할 수 있습니다")


def validate(kind: str, values: dict) -> list[str]:
    """Report typed/cross-field errors without importing or launching a learner."""
    schema = get_schema(kind)
    a, errors = _values(schema, values)
    for field in schema["fields"]:
        value = a.get(field["dest"])
        if field["required"] and (value is None or value == "" or value == []):
            errors.append(f"{field['label']} ({field['flag']}) 값을 지정하세요")
        if field["kind"] == "path" and value:
            error = _path_error(value, field["path_mode"])
            if error:
                errors.append(f"{field['label']}: {error}")
    # Cross-field checks require well-typed, present values; avoid cascading tracebacks.
    if not errors:
        _cross_validate(kind, a, errors)
    return errors


def _tuple_argument(value):
    # argparse does not recognize negative scientific notation inside fixed-nargs
    # groups as a number. Decimal expands the existing float text without rounding.
    return format(Decimal(str(value)), "f") if isinstance(value, float) and value < 0 else str(value)


def build_argv(kind: str, values: dict, python: str = sys.executable) -> list[str]:
    """Build an argv list, never shell text. Refuse unknown or invalid settings."""
    errors = validate(kind, values)
    if errors:
        raise ValueError("\n".join(errors))
    schema = get_schema(kind)
    a, _ = _values(schema, values)
    argv = [str(python), "-m", schema["module"]]
    if schema["subcommand"]:
        argv.append(schema["subcommand"])
    superseded = {row[1] for row in _document()["contracts"]["slot_superseded"]} if a.get("opp_slots") else set()
    for field in schema["fields"]:
        value = a[field["dest"]]
        if value is None or field["dest"] in superseded:
            continue
        flag = field["flag"]
        if field["kind"] == "bool":
            if value == (field["action"] == "store_true"):
                argv.append(flag)
            continue
        if value == "" and field["default"] in (None, ""):
            continue
        if field["kind"] == "path":
            value = str(Path(value).expanduser().absolute())
        if field["kind"] == "json":
            value = copy.deepcopy(value)
            for slot in value:
                if slot.get("checkpoint"):
                    slot["checkpoint"] = str(Path(slot["checkpoint"]).expanduser().absolute())
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        elif field["kind"] == "list":
            if field["separator"] == "comma":
                if field["item_kind"] == "path":
                    value = [x if x in ("self", "teacher") else str(Path(x).expanduser().absolute()) for x in value]
                value = ",".join(str(x) for x in value)
            elif field["action"] == "append":
                for item in value:
                    argv.extend((flag, str(item)))
                continue
            else:
                argv.append(flag)
                argv.extend(_tuple_argument(x) for x in value)
                continue
        text = str(value)
        # A comma list starting with a negative offset is otherwise parsed as an option.
        if text.startswith("-"):
            argv.append(f"{flag}={text}")
        else:
            argv.extend((flag, text))
    return argv


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parse_argv(kind: str, args: list[str]) -> dict:
    """Parse flags (without Python/module prefix) for recipes and saved launch configs."""
    schema = get_schema(kind)
    parser = _Parser(add_help=False, allow_abbrev=False)
    for field in schema["fields"]:
        options = dict(dest=field["dest"], default=field["default"], required=False)
        if field["kind"] == "bool":
            options["action"] = field["action"]
        else:
            options["type"] = {"int": int, "float": float}.get(field["item_kind"], str)
            if field.get("separator") == "comma":
                options["type"] = str
            if field["nargs"] is not None:
                options["nargs"] = field["nargs"]
            if field["choices"] is not None:
                options["choices"] = field["choices"]
            if field["action"] == "append":
                options["action"] = "append"
        parser.add_argument(*field["flags"], **options)
    result, errors = _values(schema, vars(parser.parse_args(args)))
    if errors:
        raise ValueError("\n".join(errors))
    return result
