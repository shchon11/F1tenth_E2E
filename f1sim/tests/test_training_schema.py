"""Panel coverage is the real parser contract, not a hand-picked menu of flags."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from f1sim.viewer.console import training_schema as schema


def test_panel_import_and_validation_are_torch_free():
    code = """
import sys
from f1sim.viewer.console.training_schema import get_schema, validate, build_argv, parse_argv
assert not validate('ppo', {})
assert not validate('dagger', {})
assert parse_argv('ppo', ['--total', '1024'])['total'] == 1024
assert build_argv('ppo', {})[1:3] == ['-m', 'f1sim.learn.ppo']
assert 'torch' not in sys.modules
assert not any(k.startswith('PyQt') for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


@pytest.fixture(scope="module")
def actual_parsers():
    from f1sim.viewer.console.training_schema_export import TARGETS, capture_parser
    return {module: capture_parser(module) for module in {m for m, _ in TARGETS.values()}}


def test_every_actual_parser_action_and_default_is_checked_in():
    from f1sim.viewer.console.training_schema_export import export_schema, SCHEMA_PATH
    assert export_schema() == json.loads(SCHEMA_PATH.read_text())


def test_every_real_action_is_reachable_and_typed(actual_parsers):
    from f1sim.viewer.console.training_schema_export import TARGETS
    for kind, (module, subcommand) in TARGETS.items():
        parser = actual_parsers[module]
        if subcommand:
            parser = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices[subcommand]
        actions = {a.dest: a for a in parser._actions if not isinstance(a, argparse._HelpAction)}
        fields = schema.get_schema(kind)["fields"]
        assert {f["dest"] for f in fields} == set(actions)
        for f in fields:
            action = actions[f["dest"]]
            assert f["flags"] == action.option_strings
            assert f["nargs"] == action.nargs
            assert f["required"] == action.required
            assert f["choices"] == (list(action.choices) if action.choices is not None else None)
            assert f["kind"] in {"bool", "int", "float", "str", "path", "choice", "list", "json"}
            assert f["group"] in {"기본", "환경", "teacher·상대차", "모델·제어", "학습", "PPO 보상", "출력/실행", "고급"}
            assert f["label"]


def _job_values(kind, tmp_path):
    weights = tmp_path / "policy with spaces.pt"
    weights.write_bytes(b"placeholder: parser test never loads it")
    data = tmp_path / "dataset"
    data.mkdir(exist_ok=True)
    return {
        "ppo": {"name": "panel_roundtrip"},
        "dagger": {"name": "panel_roundtrip"},
        "grip_collect": {"policy": str(weights), "estimator": str(weights), "out": str(tmp_path / "collect"), "tracks": "train"},
        "grip_fit": {"data": str(data), "out": str(tmp_path / "fit"), "seed": 123, "epochs": 2},
        "grip_final": {"data": str(data), "candidate": str(weights), "out": str(tmp_path / "final.json")},
        "grip_pilot": {"out": str(tmp_path / "pilot")},
    }[kind]


@pytest.mark.parametrize("kind", schema.kinds())
def test_every_job_builds_and_roundtrips_through_actual_parser(kind, tmp_path, actual_parsers):
    values = _job_values(kind, tmp_path)
    argv = schema.build_argv(kind, values, python="/python with spaces")
    spec = schema.get_schema(kind)
    assert argv[:3] == ["/python with spaces", "-m", spec["module"]]
    parsed = vars(actual_parsers[spec["module"]].parse_args(argv[3:]))
    parsed.pop("command", None)
    raw, errors = schema._values(spec, parsed)
    assert errors == []
    expected, errors = schema._values(spec, values)
    assert errors == []
    assert raw == expected
    flags = argv[4:] if spec["subcommand"] else argv[3:]
    assert schema.parse_argv(kind, flags) == expected


def test_common_settings_and_reward_terms_have_written_labels():
    """Every field a person actually sets must carry a label somebody wrote.

    This used to assert the label contained a Hangul character, which is not the same thing and
    is now wrong: a term of art keeps its English name on purpose -- `controller adaptation`,
    `max grad norm`, `action space` -- so that the label, the CLI flag and the paper agree. What
    must not happen is a field falling through to `dest.replace("_", " ")`, which is the fallback
    `training_schema_export` uses when no label was written for it.
    """
    fields = {f["dest"]: f for f in schema.get_schema("ppo")["fields"]}
    for dest in ("envs", "tracks", "adaptation", "controller", "estimator", "init", "lr",
                 "collision_penalty", "lap_bonus", "overtake_bonus", "ttc_penalty",
                 "sideslip_penalty"):
        label = fields[dest]["label"]
        assert label, dest
        assert label != dest.replace("_", " "), f"{dest} has no written label"


def test_unknown_inputs_never_become_extra_cli():
    assert "지원하지 않는 설정" in schema.validate("ppo", {"extra": "--surprise"})[0]
    with pytest.raises(ValueError):
        schema.build_argv("ppo", {"extra": "--surprise"})
    with pytest.raises(ValueError):
        schema.parse_argv("ppo", ["--surprise"])
    with pytest.raises(ValueError):
        schema.parse_argv("ppo", ["--env", "4"])


@pytest.mark.parametrize("values", [{"lr": float("nan")}, {"total": float("inf")}, {"envs": 3.5}, {"amp": "false"}, {"opp_speed": [1.]}, {"opp_speed": [1, float("inf")]}, {"total": True}])
def test_nonfinite_wrong_shape_and_wrong_types_refused(values):
    assert schema.validate("ppo", values)


def test_paths_required_and_correct_kind(tmp_path):
    assert schema.validate("grip_fit", {})
    assert schema.validate("ppo", {"init": str(tmp_path / "missing.pt")})
    assert schema.validate("ppo", {"init": str(tmp_path)})
    assert schema.validate("grip_fit", {"data": str(tmp_path), "out": str(tmp_path / "out"), "seed": 0, "epochs": 2}) == []


def test_fixed_tuple_negative_csv_and_json_are_single_argv_values(tmp_path):
    values = {"teacher_kind": "interactive", "action_mode": "plan", "race_size": 2, "envs": 4,
              "teacher_offsets": [-.7, 0., .7], "teacher_cost": [2, 5, 8, 1, .01],
              "opp_slots": [{"kind": "raceline", "speed_scale": [.7, .9], "spawn": "ahead"}]}
    argv = schema.build_argv("dagger", values)
    assert "--teacher-offsets=-0.7,0.0,0.7" in argv
    slots = argv[argv.index("--opp-slots") + 1]
    assert json.loads(slots) == values["opp_slots"]
    assert "--opponent" not in argv and "--opp-speed" not in argv
    assert schema.parse_argv("dagger", argv[3:])["teacher_offsets"] == [-.7, 0., .7]


def test_shell_metacharacters_remain_data(tmp_path):
    sentinel = tmp_path / "never-created"
    name = f"x; touch {sentinel}; $(echo injection)"
    argv = schema.build_argv("ppo", {"name": name})
    assert argv[argv.index("--name") + 1] == name
    assert not sentinel.exists()


def test_conflicting_per_car_and_global_settings_are_errors():
    values = {"envs": 4, "race_size": 2, "opp_slots": [{"kind": "raceline"}]}
    assert schema.validate("ppo", values) == []
    for key, value in {"opponent": "teacher", "opp_speed": [.8, 1.], "spawn_order": "ahead", "opp_event_rate": 1.}.items():
        assert any("--opp-slots" in e for e in schema.validate("ppo", values | {key: value}))
    assert schema.validate("ppo", values | {"race_size": 3, "envs": 6})


def test_slot_file_roundtrip(tmp_path):
    slots = tmp_path / "slots.json"
    slots.write_text(json.dumps([{"kind": "self"}]))
    argv = schema.build_argv("ppo", {"envs": 2, "race_size": 2, "opp_slots": f"@{slots}"})
    assert json.loads(argv[argv.index("--opp-slots") + 1]) == [{"kind": "self"}]


def test_valid_research_ranges_not_arbitrarily_capped():
    assert schema.validate("ppo", {"envs": 6, "race_size": 3, "opp_speed": [1.1, 1.3], "collision_penalty": 1000., "seed": 101}) == []


def test_interactive_teacher_and_batch_validation():
    assert schema.validate("dagger", {"teacher_kind": "interactive"})
    assert schema.validate("dagger", {"teacher_kind": "interactive", "action_mode": "plan", "envs": 4, "race_size": 2}) == []
    assert schema.validate("ppo", {"envs": 5, "race_size": 2})
    assert schema.validate("dagger", {"start_iter": 3})


def test_adaptive_new_and_resume_contracts(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not loaded by UI")
    values = {"adaptation": "speed", "controller": "auto", "memory": "gru", "action_mode": "plan",
              "init": str(checkpoint), "reference": str(checkpoint), "fresh_opt": True,
              "total": 256, "envs": 4, "horizon": 32}
    assert schema.validate("ppo", values) == []
    # Resume/full-stage reference identity may live in the checkpoint: do not demand a new one.
    assert schema.validate("ppo", values | {"adaptation": "full", "reference": ""}) == []
    assert schema.validate("ppo", values | {"total": 255})
    assert schema.validate("ppo", values | {"memory": "off"})
    assert schema.validate("ppo", values | {"cond": "true_mu"})
    assert schema.validate("ppo", values | {"init_log_std": -2.})
    assert schema.validate("ppo", {"research_estimator": True})


def test_pilot_caps_and_output_guard(tmp_path):
    out = tmp_path / "pilot"
    out.mkdir()
    assert schema.validate("grip_pilot", {"out": str(out)}) == []
    assert schema.validate("grip_pilot", {"out": str(out), "epochs": 11})
    assert schema.validate("grip_pilot", {"out": str(out), "steps": 20})
    (out / "existing.json").write_text("{}")
    assert schema.validate("grip_pilot", {"out": str(out)})


def test_final_eval_is_single_use(tmp_path):
    values = _job_values("grip_final", tmp_path)
    assert schema.validate("grip_final", values) == []
    marker = Path(values["data"]) / "final_opened.json"
    marker.write_text("{}")
    assert schema.validate("grip_final", values)
    marker.unlink()
    Path(values["out"]).write_text("{}")
    assert schema.validate("grip_final", values)


def test_collector_train_maps_aliases_and_splits(tmp_path):
    values = _job_values("grip_collect", tmp_path)
    assert schema.validate("grip_collect", values) == []
    assert schema.validate("grip_collect", values | {"tracks": "real/map16x07"})
    assert schema.validate("grip_collect", values | {"split_seeds": [1, 1, 2, 3]})
    assert schema.validate("grip_collect", values | {"split_seeds": [-1, 1, 2, 3]})
    assert schema.validate("grip_collect", values | {"tracks": "real/icra22,real/bb21-1,real/bb21-2,real/bb21-3"}) == []


def test_schema_returns_independent_copy():
    first = schema.get_schema("ppo")
    first["fields"].clear()
    assert len(schema.get_schema("ppo")["fields"]) > 100
    with pytest.raises(ValueError):
        schema.get_schema("unknown")


def test_zero_lr_and_endpoint_are_valid_ablations():
    assert schema.validate("ppo", {"lr": 0., "lr_end": 0.}) == []
    assert schema.validate("dagger", {"lr": 0.}) == []
    assert schema.validate("ppo", {"lr": -1.})


def test_paths_expand_home_and_pin_launch_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "a checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    argv = schema.build_argv("ppo", {"init": "~/a checkpoint.pt", "metrics_jsonl": "logs/progress.jsonl"})
    assert argv[argv.index("--init") + 1] == str(checkpoint)
    assert argv[argv.index("--metrics-jsonl") + 1] == str(tmp_path / "logs/progress.jsonl")
    parsed = schema.parse_argv("ppo", argv[3:])
    assert parsed["init"] == str(checkpoint)
    values = {"envs": 2, "race_size": 2, "opp_slots": [{"kind": "policy", "checkpoint": "a checkpoint.pt"}]}
    argv = schema.build_argv("ppo", values)
    assert json.loads(argv[argv.index("--opp-slots") + 1])[0]["checkpoint"] == str(checkpoint)
    assert values["opp_slots"][0]["checkpoint"] == "a checkpoint.pt"  # builder does not mutate UI state
    argv = schema.build_argv("ppo", {"envs": 2, "race_size": 2, "opponent": "pool", "opp_pool": ["self", "teacher", "a checkpoint.pt"]})
    assert argv[argv.index("--opp-pool") + 1] == f"self,teacher,{checkpoint}"


@pytest.mark.parametrize("slots", [[{"kind": "policy", "checkpoint": 123}], [{"kind": "policy", "checkpoint": "bad\x00path"}], [{"kind": "raceline", "speed_scale": [1, "inf"]}], [{"kind": "raceline", "event_rate": "nan"}]])
def test_malformed_slots_return_errors_not_exceptions(slots):
    assert schema.validate("ppo", {"envs": 2, "race_size": 2, "opp_slots": slots})


def test_large_integer_configuration_is_not_rounded():
    exact = 9007199254740993
    assert schema.parse_argv("ppo", ["--seed", str(exact)])["seed"] == exact


def test_fit_refuses_nonempty_output(tmp_path):
    values = _job_values("grip_fit", tmp_path)
    out = Path(values["out"])
    out.mkdir()
    (out / "candidate.pt").write_bytes(b"previous")
    assert schema.validate("grip_fit", values)


def test_negative_scientific_fixed_tuple_roundtrips_actual_parser(actual_parsers):
    values = {"overtake_sustained": [1., 2., -1e-30]}
    argv = schema.build_argv("ppo", values)
    parsed = actual_parsers["f1sim.learn.ppo"].parse_args(argv[3:])
    assert parsed.overtake_sustained == values["overtake_sustained"]


def test_observer_warmstart_and_replay_contract(tmp_path, actual_parsers):
    values = _job_values("grip_fit", tmp_path)
    initial = tmp_path / "initial estimator.pt"
    replay = tmp_path / "controlled train.pt"
    initial.write_bytes(b"parser placeholder")
    replay.write_bytes(b"parser placeholder")
    values.update(init_estimator=str(initial), replay_data=str(replay), replay_fraction=.25)
    argv = schema.build_argv("grip_fit", values)
    args = actual_parsers["f1sim.learn.policy_grip_data"].parse_args(argv[3:])
    assert args.init_estimator == initial and args.replay_data == replay and args.replay_fraction == .25
    assert schema.validate("grip_fit", values | {"replay_data": None})
    assert schema.validate("grip_fit", values | {"replay_fraction": 1.})
    assert schema.validate("grip_fit", values | {"batch_size": 1})
    assert schema.validate("grip_fit", values | {"batch_size": 2, "replay_fraction": .01})
    assert schema.validate("grip_fit", values | {"replay_data": None, "replay_fraction": .5}) == []


def test_teacher_cost_components_follow_actual_dataclass(actual_parsers):
    import dataclasses
    from f1sim.interactive_teacher import TeacherCost

    field = next(f for f in schema.get_schema("dagger")["fields"] if f["dest"] == "teacher_cost")
    names = [f.name for f in dataclasses.fields(TeacherCost)]
    defaults = TeacherCost()
    assert field["ui_length"] == len(names) == 5
    assert field["component_names"] == names
    assert field["component_labels"] == ["진행", "벽", "상대차", "여유", "부드러움"]
    assert field["component_defaults"] == [getattr(defaults, name) for name in names]
    # UI layout metadata must not change the actual single-CSV CLI contract.
    assert field["nargs"] is None and field["default"] == "" and field["separator"] == "comma"
    automatic = schema.build_argv("dagger", {"teacher_cost": ""})
    assert actual_parsers["f1sim.learn.dagger"].parse_args(automatic[3:]).teacher_cost == ""
    costs = [2.0, 31.0, 9.0, .8, .2]
    argv = schema.build_argv("dagger", {"teacher_cost": costs})
    parsed = actual_parsers["f1sim.learn.dagger"].parse_args(argv[3:])
    assert [float(x) for x in parsed.teacher_cost.split(",")] == costs
    assert schema.parse_argv("dagger", argv[3:])["teacher_cost"] == costs


def test_collector_split_seed_ranges_cannot_overlap(tmp_path, actual_parsers):
    values = _job_values("grip_collect", tmp_path)
    errors = schema.validate("grip_collect", values | {"split_seeds": [1, 2, 3, 4], "jobs_per_split": 12})
    assert any("시드 범위가 겹칩니다" in error for error in errors)
    from f1sim.learn.policy_grip_data import make_specification
    args = actual_parsers["f1sim.learn.policy_grip_data"].parse_args(
        ["collect", "--out", values["out"], "--policy", values["policy"],
         "--estimator", values["estimator"], "--tracks", "train",
         "--split-seeds", "1,2,3,4", "--jobs-per-split", "12"])
    with pytest.raises(ValueError, match="seed ranges overlap"):
        make_specification(args)
    assert schema.validate("grip_collect", values | {"split_seeds": [1, 13, 25, 37], "jobs_per_split": 12}) == []
    assert schema.validate("grip_collect", values | {"split_seeds": [37, 1, 25, 13], "jobs_per_split": 12}) == []
    assert schema.validate("grip_collect", values | {"split_seeds": [1, 12, 25, 37], "jobs_per_split": 12})
    assert schema.validate("grip_collect", values | {"split_seeds": [1, 2, 3, 4], "jobs_per_split": 1}) == []


def test_coerce_values_is_portable_and_does_not_mutate_input(tmp_path):
    values = {"envs": "5", "race_size": "2", "init": str(tmp_path / "absent.pt"),
              "opp_slots": f"@{tmp_path / 'absent-slots.json'}", "opp_speed": "0.7,1.2"}
    original = values.copy()
    result = schema.coerce_values("ppo", values)
    assert result["envs"] == 5 and result["race_size"] == 2  # cross-field launch invalidity allowed
    assert result["opp_speed"] == [.7, 1.2]
    assert result["init"] == values["init"] and result["opp_slots"] == values["opp_slots"]
    assert result["controller"] == "legacy"
    assert values == original
    assert schema.coerce_values("grip_fit", {})["epochs"] is None  # required-at-launch only
    assert schema.validate("ppo", result)  # launch still checks the actual files/constraints


@pytest.mark.parametrize("values", [{"controller": "unknown"}, {"unknown_field": 1}, {"opp_speed": [1.]}, {"lr": "nan"}, {"amp": "yes"}])
def test_coerce_values_rejects_invalid_shapes_types_keys_and_choices(values):
    with pytest.raises(ValueError):
        schema.coerce_values("ppo", values)


def test_observation_contract_uses_actual_trainer_defaults():
    from types import SimpleNamespace
    from f1sim.gym_env import EnvConfig
    from f1sim.params import Config
    from f1sim.mpc import ACT_DIM
    from f1sim.learn import common
    from f1sim.learn.obs import ObsSpec
    from f1sim.learn.memory import memory_spec

    cfg, env_cfg = Config(), EnvConfig()
    actual = common.obs_spec(SimpleNamespace(
        ecfg=env_cfg, n_beams=len(range(0, cfg.lidar.n_beams, env_cfg.scan_subsample)),
        range_max=cfg.lidar.range_max, act_dim=ObsSpec().act_dim, opp_token=env_cfg.opp_token))
    for kind in ("ppo", "dagger"):
        contract = schema.get_observation_contract(kind)
        assert set(contract["fixed"]) == {"n_beams", "range_max", "v_max", "gyro_scale", "accel_scale", "att_scale", "action_history", "hist_stride"}
        assert contract["fixed"] == {key: getattr(actual, key) for key in contract["fixed"]}
        assert contract["action_dims"] == {"direct": actual.act_dim, "plan": ACT_DIM}
        assert contract["memory"] == {key: memory_spec()[key] for key in ("layers", "critic")}
        fields = {field["dest"] for field in schema.get_schema(kind)["fields"]}
        assert set(contract["restorable"].values()) <= fields
        assert contract["restorable"]["opp_future_model"] == "opp_future_model"
        contract["fixed"].clear()
        assert schema.get_observation_contract(kind)["fixed"]
    with pytest.raises(ValueError):
        schema.get_observation_contract("grip_fit")


def test_observation_contract_getter_is_torch_free():
    code = """
import sys
from f1sim.viewer.console.training_schema import get_observation_contract
assert get_observation_contract('ppo')['fixed']['n_beams'] > 0
assert get_observation_contract('dagger')['memory']['layers'] > 0
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_privileged_contract_and_adapters_follow_actual_runtime():
    from types import SimpleNamespace
    import torch
    from f1sim.gym_env import F1VecEnv
    from f1sim.learn.conditioning import PRIV_ADAPTERS

    contract = schema.get_observation_contract("ppo")
    # The true-mu index is the actual _priv block width; the packing then adds
    # the registered randomized parameters and its one speed-cap column.
    for name, cars in (("solo", 1), ("race", 2), ("race", 4)):
        actual_prefix = F1VecEnv.priv_mu_index.fget(SimpleNamespace(M=cars))
        assert contract["privileged_dims"][name] == actual_prefix + len(F1VecEnv.PRIV_PARAMS) + 1
    assert set(contract["priv_adapters"]) == set(PRIV_ADAPTERS)
    for name, layout in contract["priv_adapters"].items():
        result = PRIV_ADAPTERS[name](torch.zeros(2, layout["input"]))
        assert result.shape == (2, layout["output"])
        assert layout["input"] == contract["privileged_dims"]["solo"]
        assert layout["output"] == contract["privileged_dims"]["race"]
    assert schema.get_observation_contract("dagger")["privileged_dims"] == contract["privileged_dims"]


def test_dagger_solo_data_share_roundtrips_actual_parser(actual_parsers):
    values = {"solo_fraction": .3, "solo_tracks": "real/icra22#bare", "race_size": 2,
              "envs": 4, "teacher_kind": "interactive", "action_mode": "plan"}
    assert schema.validate("dagger", values) == []
    argv = schema.build_argv("dagger", values)
    parsed = actual_parsers["f1sim.learn.dagger"].parse_args(argv[3:])
    assert parsed.solo_fraction == .3 and parsed.solo_tracks == "real/icra22#bare"
    fields = {f["dest"]: f for f in schema.get_schema("dagger")["fields"]}
    assert fields["solo_fraction"]["group"] == "환경"
    assert "라벨 개수 기준" in fields["solo_fraction"]["help"]
    assert schema.parse_argv("dagger", argv[3:])["solo_fraction"] == .3


def test_dagger_solo_mix_constraints_are_visible_before_launch():
    base = {"solo_fraction": .3, "race_size": 2, "envs": 4,
            "teacher_kind": "interactive", "action_mode": "plan"}
    for changed in ({"solo_fraction": -0.1}, {"solo_fraction": 1.}, {"race_size": 1},
                    {"teacher_kind": "raceline"}, {"opp_token": "tokens"}):
        assert schema.validate("dagger", dict(base, **changed))


def test_dagger_solo_rounding_and_disabled_custom_maps_are_rejected():
    assert schema.validate("dagger", {"solo_tracks": "real/icra22"})
    assert schema.validate("dagger", {"solo_fraction": .3, "steps": 1, "race_size": 2,
                                      "teacher_kind": "interactive", "action_mode": "plan"})
