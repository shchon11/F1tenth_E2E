"""Panel contracts: every real setting, no shell, persistent modes and faithful resume."""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from f1sim.viewer.console import training as T
from f1sim.viewer.console import training_schema as S
from f1sim.viewer.console.training_setup import MODES, TrainingSetupForm


def saved_policy_contract(phase="ppo"):
    fixed = S.get_observation_contract("ppo")["fixed"]
    spec = dict(fixed, scan_stack=1, scan_stride=1, hist_len=0, opp_token="off", opp_future_model="pred", act_dim=8)
    model = {"n_stack": 1, "n_beams": fixed["n_beams"], "proprio_dim": 26, "priv_dim": 17,
             "act_dim": 8, "scan_deltas": False, "scan_stem": "plain", "temporal_encoder": "cnn"}
    return {"phase": phase, "observation_spec": spec, "observation_proprio_dim": 26,
            "model_contract": model, "action_mode": "plan", "adaptation": "off"}


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["panel-test"])


@pytest.fixture
def form(app, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    widget = TrainingSetupForm()
    widget.seg_stage.set_current("expert")
    widget._stage_chosen("expert")
    widget.combo_mode.setCurrentIndex(widget.combo_mode.findData("ppo"))
    yield widget
    widget.deleteLater()


def fill_required(form, tmp_path):
    policy = tmp_path / "policy with spaces.pt"
    policy.write_bytes(b"fake; launch tests never load a model")
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    values = form.values()
    values.update({"out": str(tmp_path / (form.mode + (".json" if form.mode == "grip_final" else "")))} if "out" in form.fields else {})
    for key, field in form.fields.items():
        if field.get("path_mode") == "input_file" and (field["required"] or key in ("init", "reference", "estimator")):
            values[key] = str(policy)
        elif field.get("path_mode") == "input_dir" and field["required"]:
            values[key] = str(data)
    if "tracks" in form.fields:
        values["tracks"] = "train"
    if form.mode == "ppo":
        values["reference"] = None
    form.set_values(values)
    if form.mode in ("speed", "full"):
        form._remember_metadata(values["init"], {"adaptation": "off" if form.mode == "speed" else "speed",
                                               "original_reference": {"path": str(policy)}})


def test_import_and_construct_do_not_import_torch(tmp_path):
    code = "from PyQt5 import QtWidgets; from f1sim.viewer.console.training_setup import TrainingSetupForm; import sys; a=QtWidgets.QApplication([]); f=TrainingSetupForm(); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode,_title,kind", MODES)
def test_every_mode_has_all_typed_controls_and_launches_argv_only(form, tmp_path, monkeypatch, mode, _title, kind):
    form.set_mode(mode)
    rendered = set(form.editors)
    if form.tracks:
        rendered.add("tracks")
        if "obstacle_draws" in form.fields:
            rendered.add("obstacle_draws")
    if form.opp_table:
        rendered.add("opp_slots")
    assert rendered == {f["dest"] for f in S.get_schema(kind)["fields"]}
    assert not hasattr(form, "edit_extra")
    fill_required(form, tmp_path)
    emitted = []
    form.launch_requested.connect(lambda *args: emitted.append(args))
    form._launch()
    assert emitted, form.launch_note.text()
    name, argv, device = emitted[0]
    assert argv[:3] == [sys.executable, "-m", S.get_schema(kind)["module"]]
    assert name and device
    start = 4 if S.get_schema(kind)["subcommand"] else 3
    values = S.parse_argv(kind, argv[start:])
    assert not S.validate(kind, values)
    manager = T.JobManager(str(tmp_path / "launched-jobs"))
    monkeypatch.setattr(manager, "list_jobs", lambda: [])
    calls = []
    class Process:
        pid = 987654321
    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()
    monkeypatch.setattr(T.subprocess, "Popen", fake_popen)
    job = manager.launch(name, argv, device)
    assert calls[0][0] == argv and "shell" not in calls[0][1]
    assert Path(job.log).is_file()


def test_modes_and_saved_configuration_keep_user_values(form, tmp_path):
    form.set_values({"lr": .000123, "tracks": "real/icra22", "collision_penalty": 17.})
    form.set_mode("dagger")
    form.set_values({"teacher_kind": "interactive", "iters": 23})
    form.set_mode("ppo")
    assert form.values()["lr"] == .000123
    assert form.values()["tracks"] == "real/icra22"
    path = tmp_path / "recipe.json"
    form.save_config(path)
    form.set_values({"lr": .9})
    form.load_config(path)
    assert form.values()["lr"] == .000123
    form.set_mode("dagger")
    assert form.values()["iters"] == 23
    assert form.values()["teacher_kind"] == "interactive"


def test_seed_above_signed_integer_range_is_not_silently_clamped(form, tmp_path):
    form.set_values({"seed": 3_000_000_001})
    path = tmp_path / "large-seed.json"
    form.save_config(path)
    form.load_config(path)
    assert form.values()["seed"] == 3_000_000_001


def test_reference_only_editable_for_adaptation_and_all_flags_are_counted(form):
    assert not form.editors["reference"].isEnabled()
    assert f"{len(form.fields)}개 표시" in form.coverage_note.text()
    form.set_mode("speed")
    assert form.editors["reference"].isEnabled()


def test_search_finds_reward_and_teacher_fields(form):
    form.search.setText("collision_penalty")
    assert not form.rows["collision_penalty"].isHidden()
    assert form.rows["lr"].isHidden()
    form.set_mode("dagger")
    form.search.setText("teacher_kind")
    assert not form.rows["teacher_kind"].isHidden()


def test_teacher_cost_has_five_named_optional_numeric_components(form):
    form.set_mode("dagger")
    editor = form.editors["teacher_cost"]
    assert len(editor.items) == 5
    assert editor.value() is None
    assert {"진행", "벽", "상대차", "여유", "부드러움"} <= {w.text() for w in editor.findChildren(QtWidgets.QLabel)}
    editor.enabled_box.setChecked(True)
    assert editor.value() == [1., 30., 8., 1., .35]
    editor.items[2].set_value(12.5)
    _, argv, _ = form.argv()
    assert argv[argv.index("--teacher-cost") + 1] == "1.0,30.0,12.5,1.0,0.35"
    editor.enabled_box.setChecked(False)
    assert form.values()["teacher_cost"] is None


def test_compact_training_page_does_not_inherit_monitor_minimum_height(app, monkeypatch):
    monkeypatch.setattr(T.JobManager, "list_jobs", lambda _self: [])
    page = T.TrainingPage()
    page.resize(1000, 720)
    page.show()
    app.processEvents()
    assert (page.width(), page.height()) == (1000, 720)
    page.form.set_mode("grip_fit")
    app.processEvents()
    assert (page.width(), page.height()) == (1000, 720)
    page.close()
    page.deleteLater()


def test_invalid_race_budget_blocks_launch(form):
    form.set_values({"envs": 17, "race_size": 3})
    emitted = []
    form.launch_requested.connect(lambda *args: emitted.append(args))
    form._launch()
    assert not emitted and "배수" in form.launch_note.text()


def test_same_stage_ppo_resume_preserves_total_reference_and_optimizer(form, tmp_path):
    form.set_mode("speed")
    fill_required(form, tmp_path)
    original = form.values()
    _, argv, _ = form.argv()
    resume = tmp_path / "ppo_u100.pt"
    resume.write_bytes(b"fake")
    metadata = {"adaptation": "speed", "original_reference": {"path": original["reference"]},
                "stage_schedule": {"stage": "speed", "total_steps": original["total"], "hyperparameters": {}}}
    form.load_job(argv, str(resume), checkpoint_metadata=metadata)
    resumed = form.values()
    assert resumed["total"] == original["total"]
    assert resumed["reference"] == original["reference"]
    assert resumed["adaptation"] == "speed"
    assert resumed["fresh_opt"] is False
    assert resumed["init"] == str(resume)
    assert "--fresh-opt" not in form.argv()[1]
    form.set_mode("dagger")
    form.set_mode("speed")
    assert form._resuming
    path = tmp_path / "resume-config.json"
    form.save_config(path)
    form.load_config(path)
    assert form._resuming and form.values()["total"] == original["total"]


def test_dagger_resume_uses_next_iteration_and_keeps_requested_iterations(form, tmp_path):
    form.set_mode("dagger")
    fill_required(form, tmp_path)
    form.set_values({"iters": 5})
    _, argv, _ = form.argv()
    checkpoint = tmp_path / "student_it7.pt"
    checkpoint.write_bytes(b"fake")
    form.load_job(argv, str(checkpoint), checkpoint_metadata={"phase": "dagger", "iter": 7})
    assert form.values()["start_iter"] == 8
    assert form.values()["iters"] == 5


@pytest.mark.parametrize("module,sub", [("policy_grip_data", "fit"), ("adaptive_grip_train", "")])
def test_observer_launcher_does_not_pollute_empty_output(tmp_path, monkeypatch, module, sub):
    manager = T.JobManager(str(tmp_path / "runs"))
    monkeypatch.setattr(manager, "list_jobs", lambda: [])
    commands = []
    class Process:
        pid = 987654321
    def fake_popen(argv, **kwargs):
        commands.append((argv, kwargs))
        return Process()
    monkeypatch.setattr(T.subprocess, "Popen", fake_popen)
    out = tmp_path / "new observer output"
    argv = [sys.executable, "-m", "f1sim.learn." + module] + ([sub] if sub else []) + ["--out", str(out)]
    job = manager.launch("observer", argv, "cpu")
    assert job.run_dir == str(out)
    assert not out.exists(), "pre-launch log must not trip empty-output guard"
    assert Path(job.log).parent == Path(manager.jobs_dir)
    assert commands[0][0] == argv
    assert "shell" not in commands[0][1]
    assert T.job_stdout_log(str(out), str(tmp_path / "runs")) == job.log


def test_observer_epoch_and_collection_progress_are_real(tmp_path):
    p = T.parse_progress("epoch 1/3 loss=0.015\nepoch 2/3: loss=0.011")
    assert p.kind == "grip" and p.update == 2 and p.n_updates == 3
    assert p.get("loss") == [.015, .011]
    assert T.charts_for(p)[0].keys == ("loss",)
    (tmp_path / "manifest.json").write_text(json.dumps({"specification": {"jobs": [{"id": 0}, {"id": 1}]}, "completed_jobs": [0]}))
    collection = T.read_progress(str(tmp_path))
    assert collection.kind == "grip_collect" and collection.fraction == .5


def test_indexed_cuda_device_is_not_accidentally_forced_to_cpu(tmp_path, monkeypatch):
    manager = T.JobManager(str(tmp_path / "runs"))
    monkeypatch.setattr(manager, "list_jobs", lambda: [])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    captured = {}
    class Process:
        pid = 987654321
    def fake_popen(argv, **kwargs):
        captured.update(kwargs)
        return Process()
    monkeypatch.setattr(T.subprocess, "Popen", fake_popen)
    manager.launch("gpu-index", [sys.executable, "-m", "f1sim.learn.ppo", "--device", "cuda:0"], "cuda:0")
    assert "CUDA_VISIBLE_DEVICES" not in captured["env"]


def test_checkpoint_browser_includes_dagger_and_observer_but_not_data_shards(tmp_path):
    for name in ("ppo_u1.pt", "student_it7.pt", "candidate.pt", "train.pt", "job00000_chunk00000.pt"):
        (tmp_path / name).write_bytes(b"fake")
    assert {name for name, _, _ in T.list_checkpoints(str(tmp_path))} == {"ppo_u1.pt", "student_it7.pt", "candidate.pt"}


def test_saved_invalid_choice_in_inactive_mode_is_atomic(form, tmp_path):
    form.set_mode("dagger")
    form.set_values({"teacher_kind": "interactive"})
    form.set_mode("ppo")
    path = tmp_path / "stale-config.json"
    form.save_config(path)
    original = form.values()
    payload = json.loads(path.read_text())
    payload["modes"]["dagger"]["teacher_kind"] = "removed_teacher_mode"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="removed_teacher_mode|허용된 값"):
        form.load_config(path)
    assert form.mode == "ppo" and form.values() == original
    with pytest.raises(ValueError, match="지원하지 않는 선택값"):
        form.editors["controller"].set_value("removed_controller")


def test_full_stage_entry_and_same_stage_resume_use_different_optimizer_rules(form, tmp_path):
    form.set_mode("full")
    fill_required(form, tmp_path)
    form.set_values({"total": 98304})
    reference = str(tmp_path / "policy with spaces.pt")
    source = tmp_path / "misleading_ppo_final.pt"
    source.write_bytes(b"metadata supplied by inspector test seam")
    form.resume_checkpoint(str(source), metadata={"adaptation": "speed", "original_reference": {"path": reference}})
    assert form.values()["fresh_opt"] is True and not form._resuming
    assert form.values()["total"] == 98304
    assert form.values()["reference"] == reference
    form.resume_checkpoint(str(source), metadata={"adaptation": "full", "original_reference": {"path": reference},
        "stage_schedule": {"stage": "full", "total_steps": 196608, "hyperparameters": {"lr": .0000123}}})
    assert form.values()["fresh_opt"] is False and form._resuming
    assert form.values()["total"] == 196608 and form.values()["lr"] == .0000123


@pytest.mark.parametrize("target,source", [("full", "off"), ("speed", "full")])
def test_impossible_adaptive_stage_transitions_are_refused(form, tmp_path, target, source):
    form.set_mode(target)
    fill_required(form, tmp_path)
    path = tmp_path / "policy with spaces.pt"
    with pytest.raises(ValueError):
        form.resume_checkpoint(str(path), metadata={"adaptation": source})


def test_initial_adaptation_reference_is_required_after_metadata_inspection(form, tmp_path):
    form.set_mode("speed")
    fill_required(form, tmp_path)
    form.set_values({"reference": None})
    path = tmp_path / "policy with spaces.pt"
    form.resume_checkpoint(str(path), metadata={"adaptation": "off", "original_reference": {}})
    with pytest.raises(ValueError, match="원본 D3"):
        form.argv()
    assert "원본 D3" in form.launch_note.text()


def test_changed_checkpoint_metadata_is_not_cached_for_new_file(form, tmp_path):
    path = tmp_path / "changed.pt"
    path.write_bytes(b"first")
    before = path.stat()
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="변경"):
        form._remember_metadata(path, {"file_size": before.st_size, "mtime_ns": before.st_mtime_ns})
    assert form._metadata_for(path) is None


@pytest.mark.parametrize("action", ["close", "deleteLater", "inspector_delete"])
def test_pending_metadata_probe_is_cleaned_up_before_qt_teardown(app, tmp_path, action):
    from PyQt5 import sip
    widget = TrainingSetupForm()
    widget.set_mode("full")
    path = tmp_path / "not-loaded-yet.pt"
    path.write_bytes(b"probe process must be stopped before it reads this")
    widget.resume_checkpoint(str(path))
    inspector = widget.inspector
    process = inspector.process
    assert inspector.busy and process is not None
    if action == "inspector_delete":
        inspector.deleteLater()
    else:
        getattr(widget, action)()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()
    assert sip.isdeleted(process) or process.state() == QtCore.QProcess.NotRunning
    if not sip.isdeleted(widget):
        widget.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_checkpoint_metadata_is_asynchronous_cpu_json_without_tensor_payload(form, app, tmp_path):
    import torch
    form.set_mode("full")
    fill_required(form, tmp_path)
    reference = str(tmp_path / "policy with spaces.pt")
    path = tmp_path / "source.pt"
    contract = saved_policy_contract()
    contract["model_contract"]["memory"] = {"kind": "gru", "hidden_size": 128, "layers": 1, "critic": "own"}
    torch.save({"state_dict": {"actor.fake": torch.ones(2)}, "meta": contract["model_contract"],
        "extra": {"phase": "ppo", "spec": contract["observation_spec"], "action_mode": "plan", "opt": {"state": torch.ones(3)},
        "experiment": {"adaptation": "speed", "original_reference": {"path": reference, "sha256": "pin"}}}}, path)
    loop = QtCore.QEventLoop()
    beats = []
    timer = QtCore.QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: beats.append(1))
    form.inspector.ready.connect(lambda *_args: loop.quit())
    form.inspector.failed.connect(lambda *_args: loop.quit())
    QtCore.QTimer.singleShot(20_000, loop.quit)
    timer.start()
    form.resume_checkpoint(str(path))
    assert not form.btn_launch.isEnabled()
    assert form.inspector.busy
    loop.exec_()
    timer.stop()
    assert beats, "Qt event processing must remain live during CPU checkpoint inspection"
    metadata = form._metadata_for(path)
    assert metadata and metadata["adaptation"] == "speed", form.launch_note.text()
    assert "opt" not in metadata and "state_dict" not in metadata
    assert form.values()["fresh_opt"] is True and form.btn_launch.isEnabled()
    assert not form.inspector.busy


def test_standalone_ppo_resume_restores_recorded_actor_inputs_and_keeps_training_parameters(form, tmp_path):
    path = tmp_path / "arbitrary-checkpoint.pt"
    path.write_bytes(b"worker metadata supplied directly")
    assert form.values()["scan_stack"] == 6 and form.values()["hist_len"] == 20
    form.set_values({"lr": .000654, "total": 123456, "scan_stem": "resnet", "memory": "gru"})
    contract = saved_policy_contract()
    contract["action_mode"] = None           # infer only from a verified supported action dimension
    form.resume_checkpoint(str(path), metadata=contract)
    values = form.values()
    assert (values["scan_stack"], values["hist_len"], values["scan_stem"], values["memory"]) == (1, 0, "plain", "off")
    assert values["action_mode"] == "plan" and values["temporal_encoder"] == "cnn"
    assert values["lr"] == .000654 and values["total"] == 123456
    assert values["race_size"] == 2             # never infer unrecorded race/training parameters
    assert "critic" in form.launch_note.text()
    with pytest.raises(ValueError, match="critic"):
        form.argv()
    form.set_values({"race_size": 1, "overtake_bonus": 0.})
    assert form.argv()[1]
    form.set_values({"hist_len": 20})
    with pytest.raises(ValueError, match="구조"):
        form.argv()


def test_loaded_legacy_resume_configuration_rechecks_checkpoint_contract_before_launch(form, tmp_path, monkeypatch):
    path = tmp_path / "resume.pt"
    path.write_bytes(b"worker metadata supplied directly")
    form.set_values({"init": str(path)})
    form._resuming = True
    requested, emitted = [], []
    monkeypatch.setattr(form.inspector, "inspect", lambda path: requested.append(path))
    form.launch_requested.connect(lambda *args: emitted.append(args))
    form._launch()
    assert requested == [str(path)] and not emitted
    form._checkpoint_ready(str(path), saved_policy_contract())
    assert form.values()["scan_stack"] == 1 and form.values()["hist_len"] == 0
    assert not emitted                    # incompatible retained race is refused
    form.set_values({"race_size": 1, "overtake_bonus": 0.})
    form._launch()
    assert emitted


def test_collection_resume_does_not_request_policy_resume_metadata(form, tmp_path, monkeypatch):
    form.set_mode("grip_collect")
    fill_required(form, tmp_path)
    _, argv, _ = form.argv()
    form.load_job(argv)
    requested, emitted = [], []
    monkeypatch.setattr(form.inspector, "inspect", lambda path: requested.append(path))
    form.launch_requested.connect(lambda *args: emitted.append(args))
    form._launch()
    assert emitted and not requested


def test_explicit_supported_critic_adapter_preserves_resume_shapes(form, tmp_path):
    path = tmp_path / "race-critic.pt"
    path.write_bytes(b"worker metadata supplied directly")
    form.set_values({"race_size": 1, "overtake_bonus": 0., "critic_priv_adapter": "absent_opponent_17_to_21"})
    contract = saved_policy_contract()
    contract["model_contract"]["priv_dim"] = 21
    form.resume_checkpoint(str(path), metadata=contract)
    assert form.values()["race_size"] == 1
    assert form.values()["critic_priv_adapter"] == "absent_opponent_17_to_21"
    assert "나머지 학습 파라미터" in form.launch_note.text()
    assert form.argv()[1]


def test_standalone_dagger_resume_restores_recurrent_contract(form, tmp_path):
    form.set_mode("dagger")
    path = tmp_path / "renamed.pt"
    path.write_bytes(b"worker metadata supplied directly")
    contract = saved_policy_contract("dagger")
    contract["iter"] = 4
    contract["model_contract"].update(memory={"kind": "gru", "hidden_size": 64, "layers": 1, "critic": "own"},
                                     scan_deltas=True, scan_stem="resnet")
    form.resume_checkpoint(str(path), metadata=contract)
    values = form.values()
    assert values["memory"] == "gru" and values["memory_hidden"] == 64
    assert values["scan_deltas"] is True and values["scan_stack"] == 1 and values["hist_len"] == 0
    assert values["action_mode"] == "plan" and values["start_iter"] == 5


@pytest.mark.parametrize("key,value", [("n_beams", 1080), ("gyro_scale", 7.), ("att_scale", .5),
                                      ("range_max", 12.), ("action_history", 3), ("hist_stride", 1)])
def test_unrepresentable_saved_observation_contract_is_refused_without_ui_changes(form, tmp_path, key, value):
    path = tmp_path / "unsupported.pt"
    path.write_bytes(b"worker metadata supplied directly")
    before = form.values()
    contract = saved_policy_contract()
    contract["observation_spec"][key] = value
    with pytest.raises(ValueError, match=key):
        form.resume_checkpoint(str(path), metadata=contract)
    assert form.values() == before


def test_saved_recurrent_critic_and_layer_contracts_are_checked(form, tmp_path):
    path = tmp_path / "memory.pt"
    path.write_bytes(b"worker metadata supplied directly")
    contract = saved_policy_contract()
    contract["model_contract"]["memory"] = {"kind": "gru", "hidden_size": 96, "layers": 1, "critic": "none"}
    form.resume_checkpoint(str(path), metadata=contract)
    assert form.values()["memory_hidden"] == 96 and form.values()["memory_critic"] == "none"
    contract["model_contract"]["memory"]["layers"] = 2
    with pytest.raises(ValueError, match="계층"):
        form.resume_checkpoint(str(path), metadata=contract)


def test_final_report_identity_retains_same_basenames_and_same_parent_results(app, tmp_path, monkeypatch):
    manager = T.JobManager(str(tmp_path / "runs"))
    monkeypatch.setattr(T, "discover_external_jobs", lambda *_: [])
    class Process:
        pid = 987654321
        def poll(self):
            return 0
    monkeypatch.setattr(T.subprocess, "Popen", lambda *args, **kwargs: Process())
    outputs = [tmp_path / "a" / "custom.json", tmp_path / "b" / "custom.json", tmp_path / "a" / "second.json"]
    launched = []
    for index, out in enumerate(outputs):
        argv = [sys.executable, "-m", "f1sim.learn.policy_grip_data", "final", "--out", str(out)]
        job = manager.launch("custom", argv, "cpu")
        launched.append(job)
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"candidate_sha256": str(index), "metrics": {"mae": index / 10}, "promoted": False}))
        assert job.run_dir == str(out) and job.report_path == str(out)
    records = manager.list_jobs()
    assert len(records) == 3
    assert len({job.log for job in records}) == 3
    assert {job.name for job in records} == {str(path) for path in outputs}
    assert {job.run_dir for job in records} == {str(path) for path in outputs}
    page = T.TrainingPage()
    page.jobs = manager
    page.refresh_all()
    opened = []
    monkeypatch.setattr(T.QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    for index, out in enumerate(outputs):
        choice = page.combo_run.findData(str(out))
        assert choice >= 0
        page.combo_run.setCurrentIndex(choice)
        page._tick(force=True)
        progress = page._progress_for(str(out))
        assert progress.finished and progress.kind == "grip_final" and progress.source == str(out)
        assert progress.get("mae") == [index / 10]
        assert T._progress_text(progress) == "최종 평가 완료"
        assert page.btn_report.isEnabled()
        page.btn_report.click()
        assert opened[-1] == str(out)
    page.deleteLater()


def test_dagger_solo_share_is_editable_and_persisted_without_cli(form, tmp_path):
    form.set_mode("dagger")
    assert {"solo_fraction", "solo_tracks"} <= set(form.editors)
    form.set_values({"solo_fraction": .3, "solo_tracks": "real/icra22#bare", "envs": 4,
                     "race_size": 2, "teacher_kind": "interactive", "action_mode": "plan"})
    path = tmp_path / "solo-mix.json"
    form.save_config(path)
    form.set_values({"solo_fraction": 0., "solo_tracks": ""})
    form.load_config(path)
    assert form.values()["solo_fraction"] == .3
    assert form.values()["solo_tracks"] == "real/icra22#bare"
    assert "라벨 개수 기준" in form.fields["solo_fraction"]["help"]


def test_dagger_resume_restores_collection_mix_instead_of_zero_fraction(form, tmp_path):
    form.set_mode("dagger")
    checkpoint = tmp_path / "student_it2.pt"
    checkpoint.write_bytes(b"metadata test seam")
    form.set_values({"solo_fraction": 0., "envs": 4, "race_size": 2, "action_mode": "plan"})
    form.resume_checkpoint(str(checkpoint), metadata={"phase": "dagger", "iter": 2,
        "collection_mix": {"requested_solo_fraction": .3, "solo_fraction": .3,
                           "solo_steps": 3, "traffic_steps": 7,
                           "solo_tracks": ["real:icra2022+bare"], "traffic_teacher": "interactive"}})
    values = form.values()
    assert values["solo_fraction"] == .3 and values["steps"] == 10
    assert values["solo_tracks"] == "real:icra2022+bare"
    assert values["teacher_kind"] == "interactive" and values["start_iter"] == 3
