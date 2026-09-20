"""The training page: choosing maps, and saying what is running.

Two of the user's three complaints land here. "학습 셋 가면 트랙 이름만 있고. 선택." -- the tracks
field was a text box containing the word `train`, so choosing anything else meant typing a hundred
and forty-nine loader names. And "지금 학습 돌리고있는것도 뭐 돌리고있는지도 모르겠고" -- the jobs
list was a name, a state and a pid.
"""
import json
import os
import time

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from f1sim import tracks
from f1sim.viewer.console import training as T


@pytest.fixture(scope="module")
def qapp():
    _prev = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from f1sim.viewer.console import app as A
    yield QtWidgets.QApplication.instance() or A.create_app(["test"])
    if _prev is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev


@pytest.fixture
def picker(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    p = T.TrackPicker()
    try:
        yield p
    finally:
        p.deleteLater()


def _checked(picker):
    return sorted(picker.selected_tracks())


# ================================================================ the track picker
def test_the_three_groups_are_tabs_of_base_maps(picker):
    assert [picker.tabs.tabText(i).split(" (")[0] for i in range(picker.tabs.count())] == \
        ["학습", "검증", "내 환경"]
    for group, lw in picker._lists.items():
        for i in range(lw.count()):
            tid = lw.item(i).data(QtCore.Qt.UserRole)
            assert tracks.group_of(tid) == group or group == "내 환경"


def test_the_default_uses_asset_variants_of_the_training_split(picker):
    """Fresh launches keep the curated maps/seeds and explicitly switch to real assets."""
    assert picker.spec() == ",".join(tracks.asset_scenario(tracks.short(n)) for n in tracks.split_names("train"))
    assert set(_checked(picker)) == set(tracks.split_tracks("train"))
    assert picker.count() == 149
    assert "149개 변형" in picker.summary.text()


def test_one_click_selects_the_whole_training_set(picker):
    picker.clear_selection()
    assert picker.spec() == "" and _checked(picker) == []
    picker.btn_all_train.click()
    assert picker.spec() == ",".join(tracks.asset_scenario(tracks.short(n)) for n in tracks.split_names("train"))


def test_touching_anything_drops_the_split_shortcut_and_emits_the_list(picker):
    picker.clear_selection()
    for lw in picker._lists.values():
        for i in range(lw.count()):
            it = lw.item(i)
            if it.data(QtCore.Qt.UserRole) in ("real/icra22", "rt/spielberg"):
                it.setCheckState(QtCore.Qt.Checked)
    for d, cb in picker.dir_boxes.items():
        cb.setChecked(d in ("", "rev"))
    assert picker.spec() == "real/icra22,real/icra22@rev,rt/spielberg,rt/spielberg@rev"
    assert picker.count() == 4


def test_an_obstacle_family_is_applied_to_every_selected_map(picker):
    picker.clear_selection()
    for lw in picker._lists.values():
        for i in range(lw.count()):
            it = lw.item(i)
            if it.data(QtCore.Qt.UserRole) == "real/icra22":
                it.setCheckState(QtCore.Qt.Checked)
    for d, cb in picker.dir_boxes.items():
        cb.setChecked(d == "")
    picker.obs_boxes[""].setChecked(False)
    picker.obs_boxes["line"].setChecked(True)
    assert picker.spec() == "real/icra22#line:*!assets=mixed:1"
    # `*` is a request for N placements, not one track
    picker.spin_draws.setValue(8)
    assert picker.count() == 8
    picker.combo_seed.setCurrentIndex(picker.combo_seed.findData("fixed"))
    picker.spin_fixed.setValue(44)
    assert picker.spec() == "real/icra22#line:44!assets=mixed:1"
    assert picker.count() == 1


def test_asset_modes_support_racetracks_without_legacy_raster_suffixes(picker):
    """New asset markers support Monza while its old +obs loader remains unchanged."""
    picker.clear_selection()
    for lw in picker._lists.values():
        for i in range(lw.count()):
            it = lw.item(i)
            if it.data(QtCore.Qt.UserRole) in ("rt/monza", "real/bb22-3"):
                it.setCheckState(QtCore.Qt.Checked)
    for d, cb in picker.dir_boxes.items():
        cb.setChecked(d == "")
    picker.obs_boxes[""].setChecked(False)
    picker.obs_boxes["edge"].setChecked(True)
    assert set(picker.spec().split(",")) == {"real/bb22-3#edge:*!assets=mixed:1", "rt/monza#edge:*!assets=mixed:1"}


def test_every_spec_the_picker_emits_is_one_the_trainer_accepts(picker):
    from f1sim.learn import common
    picker.clear_selection()
    for lw in picker._lists.values():
        for i in range(min(3, lw.count())):
            lw.item(i).setCheckState(QtCore.Qt.Checked)
    for cb in picker.obs_boxes.values():
        cb.setChecked(True)
    names = common.track_names(picker.spec(), draws=picker.draws(), seed=0)
    assert len(names) == picker.count()
    assert all(":" in n for n in names), "every entry resolves to a loader name"


def test_an_open_seed_survives_set_spec(picker):
    """`#line:*` has no loader name until a seed is drawn, so the round-trip check that decides
    whether the controls can express a value must compare the short grammar, not the loader's."""
    picker.set_spec("real/bb22-1#line:*")
    assert picker.spec() == "real/bb22-1#line:*"
    assert picker.selected_tracks() == ["real/bb22-1"]
    assert picker.obs_boxes["line"].isChecked() and not picker.obs_boxes[""].isChecked()
    assert picker.combo_seed.currentData() == "random"
    picker.spin_draws.setValue(3)
    assert picker.count() == 3


def test_a_tracks_value_the_controls_cannot_express_is_kept_verbatim(picker):
    """A curated split has a different obstacle seed per map; reducing it to (maps × obstacles)
    would train on a different set. The picker says so instead of approximating."""
    spec = "real/icra22#line:7,real/bb21-2#line:9"
    picker.set_spec(spec)
    assert picker.spec() == spec
    assert "직접 지정한" in picker.summary.text()
    picker.btn_all_train.click()
    assert picker.spec() == ",".join(tracks.asset_scenario(tracks.short(n)) for n in tracks.split_names("train"))


# ================================================================ the argv it builds
@pytest.fixture
def form(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    f = T.RecipeForm()
    try:
        yield f
    finally:
        f.deleteLater()


def test_the_default_recipe_emits_explicit_asset_scenarios(form):
    _name, argv, _dev = form.argv()
    assert argv[argv.index("--tracks") + 1] == form.tracks.spec()
    assert any(tracks.parse(n).asset for n in form.tracks.spec().split(","))
    assert argv[argv.index("--obstacle-draws") + 1] == "8"


def test_historical_recipe_is_explicit_and_training_controls_are_available(form):
    for old_widget in ("combo_controller", "edit_estimator", "ctrl_hint"):
        assert not hasattr(form, old_widget), old_widget
    assert "controller" in form.editors and "estimator" in form.editors
    assert "기본 레이스" in form.combo_recipe.currentText()

    _name, argv, _dev = form.argv()
    assert form.spin_race.value() == 2 and form.spin_race.isEnabled()
    assert form.combo_opp.isEnabled()
    assert argv.count("--controller") == 1
    assert argv[argv.index("--controller") + 1] == "legacy"
    assert "--estimator" not in argv
    assert argv[argv.index("--race-size") + 1] == "2"
    assert form.combo_device.findText("cuda") >= 0


def test_the_narrow_recipes_round_trip_through_the_picker(form):
    for key, expect in (("sgr", T.SGR_TRACKS), ("r10", T.R10_TRACKS)):
        form.combo_recipe.setCurrentIndex(form.combo_recipe.findData(key))
        _name, argv, _dev = form.argv()
        assert argv[argv.index("--tracks") + 1] == expect


def test_a_custom_selection_becomes_a_comma_list_in_the_new_grammar(form):
    form.tracks.clear_selection()
    for lw in form.tracks._lists.values():
        for i in range(lw.count()):
            if lw.item(i).data(QtCore.Qt.UserRole) == "real/korea26":
                lw.item(i).setCheckState(QtCore.Qt.Checked)
    for d, cb in form.tracks.dir_boxes.items():
        cb.setChecked(d == "")
    _name, argv, _dev = form.argv()
    assert argv[argv.index("--tracks") + 1] == "real/korea26"


# ================================================================ what is running
def _job(argv, name="cl_run", pid=None, external=False, started=None):
    return T.Job(name=name, pid=pid or os.getpid(), argv=argv, log="/tmp/x.log",
                 started=started if started is not None else time.time() - 600,
                 run_dir="/tmp/runs/cl_run", external=external)


RECORDED = [
    "/usr/bin/python", "-m", "f1sim.learn.ppo", "--action-mode", "plan", "--horizon", "32",
    "--tracks", "train", "--obstacle-draws", "8", "--envs", "256", "--total", "1048576.0",
    "--lr", "5e-05", "--lr-end", "2e-05", "--kl-coef", "0.05",
    "--init", "/home/u/f1sim_runs/_baselines/frozen_original_48cc698f.pt", "--seed", "701",
    "--race-size", "2", "--opponent", "mixed", "--aux-grip", "1.0", "--aux-opp", "1.0",
    "--device", "cuda", "--wandb", "online", "--wandb-group", "console-origrecipe",
    "--save-every", "10", "--name", "cl_origrecipe_legacy_s701_09121712", "--controller", "legacy",
    "--mixed-teacher-frac", "0.5", "--overtake-bonus", "1.0",
]

LOG = """\
$ python -m f1sim.learn.ppo --tracks train
wandb: Run data is saved locally in /home/u/f1sim_runs/x/wandb/run-20260912_171233-ab12cd34
wandb: View run at https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/ab12cd34
loading 149 tracks + racelines ...
upd 12/64 steps 0.20M cap 9.0 | rew/step 1.234 coll 0.900/km prog 41.2 m lap 12.3 s | gate 0.90 (149.0 tk) | kl_ref 0.012 | 4100 steps/s
"""


def test_the_summary_reads_the_argv_the_process_was_started_with():
    s = T.summarize_job(_job(RECORDED), log_text=LOG, progress=T.parse_progress(LOG))
    assert s.recipe == "기본 레이스 레시피"
    assert s.tracks_text == "학습 분할 (149개 변형, 53개 맵)"
    assert s.race_text == "2대 · 상대차 mixed"
    assert s.controller == "legacy"
    assert s.init == "frozen_original_48cc698f.pt · 48cc698f"
    assert s.lr_text == "5e-05 → 2e-05"
    assert s.total_text == "1.05M"
    assert s.wandb_url == "https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/ab12cd34"
    assert s.progress == "12/64 업데이트 · 0.20M 스텝"
    assert s.eta_s and s.eta_s > 0
    assert s.state == "실행 중"
    # every row the card draws has a label and a value
    assert all(k and v for k, v in s.lines())
    assert [k for k, _ in s.lines()][:3] == ["레시피", "트랙", "레이스"]


def test_an_open_seed_says_how_many_placements_it_becomes():
    """`real/bb22-1#line:*` with `--obstacle-draws 3` is three maps, not one, and the card has to
    say so -- that difference is the whole reason the flag exists."""
    argv = list(RECORDED)
    argv[argv.index("--tracks") + 1] = "real/bb22-1#line:*"
    argv[argv.index("--obstacle-draws") + 1] = "3"
    s = T.summarize_job(_job(argv))
    assert s.tracks_text == "1개 맵 · Blackbox 2022 #1 · 장애물 주행선 위 · 배치 3개씩 (3개 로드)"


def test_the_eta_uses_the_steps_per_update_the_argv_fixes():
    """The progress line prints cumulative steps to one decimal in millions; a short run sits at
    `0.0M` for its whole life, and an ETA divided by that reads 0 minutes forever."""
    log = LOG.replace("upd 12/64 steps 0.20M", "upd 12/512 steps 0.0M")
    s = T.summarize_job(_job(RECORDED), log_text=log, progress=T.parse_progress(log))
    envs, horizon = 256, 32
    expected = (512 - 12) * envs * horizon / 4100
    assert s.eta_s == pytest.approx(expected, rel=1e-6)


def test_a_named_track_list_is_reported_as_maps_not_as_a_string():
    argv = list(RECORDED)
    argv[argv.index("--tracks") + 1] = "real/bb22-1@rev,rt/spielberg,gen/control-1400,real/icra22"
    s = T.summarize_job(_job(argv))
    assert s.tracks_text == ("4개 맵 · Blackbox 2022 #1, Spielberg, 생성 control 1400 외 1개"
                             " · 방향 정방향/역방향")


def test_the_same_map_three_ways_is_named_once_and_counted_as_one_map():
    """A 202-entry list whose first three entries are one map in three directions must not be
    described as "ICRA 2022, ICRA 2022 · 역방향, ICRA 2022 · 거울": that names nothing."""
    argv = list(RECORDED)
    argv[argv.index("--tracks") + 1] = ",".join(
        ["real/icra22", "real/icra22@rev", "real/icra22@mir", "rt/spielberg", "gen/control-1400"])
    s = T.summarize_job(_job(argv))
    assert s.tracks_text == ("5개 시나리오 · 3개 맵 · ICRA 2022, Spielberg, 생성 control 1400"
                             " · 방향 정방향/역방향/거울")


def test_a_suffix_this_version_does_not_know_is_passed_through_rather_than_dropped():
    """Another branch is training with an obstacle family this registry has never heard of. An
    unknown name is data, not an error: it is counted, and the map under it is named even though the
    suffix cannot be.

    The suffix used to be `+hard`, which this registry has since learned -- so the test had quietly
    become a test of `+hard` and its assertion drifted. It names one that is genuinely unknown now.
    """
    argv = list(RECORDED)
    argv[argv.index("--tracks") + 1] = "real:icra2022+blobs1,real:icra2022+blobs1~rev,weird_name"
    s = T.summarize_job(_job(argv))
    assert s.tracks_text == "3개 시나리오 · 2개 맵 · ICRA 2022, weird_name"


def test_the_old_grammar_still_matches_its_recipe():
    """A job launched before the rename must still be recognised by the card."""
    argv = list(RECORDED)
    argv[argv.index("--tracks") + 1] = ("real:icra2022,real:blackbox2021_2,real:blackbox2022_1,"
                                        "rt:Spielberg,gen:control:1400")
    argv[argv.index("--race-size") + 1] = "1"
    argv[argv.index("--opponent") + 1] = "teacher"
    s = T.summarize_job(_job(argv))
    assert s.recipe == "SGR base (5개 맵 단독)"
    assert s.race_text == "1대 (단독)"


def test_a_run_nobody_here_started_is_named_as_such():
    s = T.summarize_job(_job(["python", "-m", "f1sim.learn.ppo", "--tracks", "heldout",
                              "--controller", "fixed_low"], external=True, pid=1))
    assert s.external and s.recipe == "외부 실행"
    assert s.tracks_text == "검증 분할 (12개 변형, 8개 맵)"
    assert s.controller == "fixed_low"


def test_opponent_event_flags_are_shown_when_present():
    argv = list(RECORDED) + ["--opp-events", "brake,shift", "--opp-event-rate", "0.5"]
    s = T.summarize_job(_job(argv))
    assert s.events_text == "brake,shift (10초당 0.5회)"
    assert ("이벤트", "brake,shift (10초당 0.5회)") in s.lines()


def test_argv_flags_reads_switches_and_pairs():
    assert T.argv_flags(["--amp", "--envs", "256", "--wandb-new"]) == \
        {"amp": "", "envs": "256", "wandb-new": ""}


def test_the_page_draws_one_card_per_job(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    runs = tmp_path / "runs"
    (runs / "_console_jobs").mkdir(parents=True)
    (runs / "cl_a").mkdir()
    (runs / "cl_a" / "console-train.log").write_text(LOG)
    json.dump({"name": "cl_a", "pid": os.getpid(), "argv": RECORDED,
               "log": str(runs / "cl_a" / "console-train.log"), "started": time.time() - 60,
               "run_dir": str(runs / "cl_a")},
              open(runs / "_console_jobs" / "cl_a.json", "w"))
    monkeypatch.setattr(T.catalog, "RUNS_DIR", str(runs))
    # Trainers really running on this machine belong to whoever started them; this test is about
    # what the page draws for the jobs it was given.
    monkeypatch.setattr(T, "discover_external_jobs", lambda runs_dir=None: [])
    page = T.TrainingPage()
    try:
        page.jobs = T.JobManager(str(runs))
        page._refresh_jobs()
        assert set(page._job_cards) == {"cl_a"}
        card = page._job_cards["cl_a"]
        assert card.title.text() == "cl_a"
        assert "실행 중" in card.state.text()
        assert "경과" in card.clock.text()
        assert card.facts._rows["트랙"].text() == "학습 분할 (149개 변형, 53개 맵)"
        assert card.facts._rows["W&B"].text().startswith("https://wandb.ai/")
    finally:
        page.deleteLater()


# ================================================================ the 현황판 charts
# "학습 패널에서 학습 런에는 다 뜨는데 현황판 그래프에 아무것도 안뜬다." Every current run's charts
# were empty, and nothing on the page said why. These are about what the page draws now: the chart
# set belongs to the run's kind, the important metric comes first, and a run with no record says so.
PPO_JSONL = [
    {"kind": "ppo", "update": k, "total": 64, "steps": k * 8192, "cap": 9.0,
     "rew_per_step": 0.1 * k, "coll_per_km": 50.0 / k, "prog_m": 10.0 * k,
     "lap_s": float("nan") if k < 3 else 14.0, "gate": 9.0, "tk": 149.0,
     "kl_ref": 0.01 * k, "sps": 4100.0, "wall_s": 12.0 * k,
     "reward/progress_per_step": 0.05, "reward/collision_per_step": -0.2,
     "traffic/car_contacts_per_min": 0.5 * k, "traffic/passes_held_per_min": 0.2 * k,
     "traffic/wall_collisions_per_min": 1.0, "traffic/ttc_share": 0.03}
    for k in range(1, 6)
]
DAGGER_JSONL = [
    {"kind": "dagger", "iter": k, "total": 6, "beta": 0.5 ** k, "samples": 64500 * (k + 1),
     "loss": 0.07 / (k + 1), "student_coll_per_km": 52.6 / (k + 1), "student_prog_mps": 3.1 + 0.2 * k,
     "student_lap_s": 14.8 - 0.3 * k, "teacher_coll_per_km": 7.2, "teacher_prog_mps": 3.6,
     "teacher_lap_s": 13.6, "wall_s": 1165.0 * (k + 1)}
    for k in range(3)
]


def _run_with(tmp_path, name, records=(), files=()):
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    if records:
        (d / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for fn, text in files:
        (d / fn).write_text(text)
    return d


@pytest.fixture
def page(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    (tmp_path / "runs" / "_console_jobs").mkdir(parents=True)
    monkeypatch.setattr(T.catalog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(T, "discover_external_jobs", lambda runs_dir=None: [])
    p = T.TrainingPage()
    p.jobs = T.JobManager(str(tmp_path / "runs"))
    try:
        yield p
    finally:
        p.deleteLater()


def _titles(page):
    return [c.title for c in page._charts.values()]


def test_a_ppo_run_plots_the_metrics_that_matter_first(page, tmp_path):
    """"중요한 메트릭들 위주로 plot하게 해줘": what the policy is judged on, in that order, then
    the traffic the opponents create, then the optimiser's diagnostics."""
    d = _run_with(tmp_path, "cl_ppo", PPO_JSONL)
    page._current_run = str(d)
    page._tick(force=True)
    assert _titles(page)[:4] == ["충돌 / km", "랩 타임", "에피소드 진행", "보상 / 스텝"]
    assert _titles(page)[-1] == "처리량"
    # the traffic charts exist because this run measured traffic, and they sit before kl_ref
    titles = _titles(page)
    assert titles.index("차량 접촉 / 분") < titles.index("KL (원본 대비)")
    assert titles[-3:] == ["커리큘럼 게이트 (충돌/km)", "채점된 트랙 수", "처리량"]
    assert {"추월 성공 / 분", "벽 충돌 / 분", "접촉 위험 시간 비율"} <= set(titles)
    coll = page._charts[("coll_per_km",)]
    assert coll._series[0] == [50.0, 25.0, 50 / 3, 12.5, 10.0] and coll.lower_is_better


def test_a_run_that_measured_no_traffic_gets_no_empty_traffic_charts(page, tmp_path):
    """A solo run has no contacts to count. A flat zero line would read as "no contacts happened"."""
    plain = [{k: v for k, v in r.items() if not k.startswith("traffic/")} for r in PPO_JSONL]
    d = _run_with(tmp_path, "cl_solo", plain)
    page._current_run = str(d)
    page._tick(force=True)
    assert not [t for t in _titles(page) if "분" in t]
    assert _titles(page)[0] == "충돌 / km"


def test_a_dagger_run_gets_its_own_charts_with_the_teacher_beside_the_student(page, tmp_path):
    """52 coll/km is a disaster against a teacher at 7 and ordinary against a teacher at 48; the
    student's curve alone does not say which. The old page had no DAgger charts at all."""
    d = _run_with(tmp_path, "cl_dagger", DAGGER_JSONL)
    page._current_run = str(d)
    page._tick(force=True)
    assert _titles(page) == ["충돌 / km · student vs teacher", "모방 손실 (imitation loss)",
                            "랩 타임 · student vs teacher", "진행 속도 · student vs teacher",
                            "beta (teacher 주행 비율)"]
    ch = page._charts[("student_coll_per_km", "teacher_coll_per_km")]
    assert len(ch._series) == 2 and ch.labels == ["student", "teacher"]
    assert ch._series[0][0] == 52.6 and ch._series[1] == [7.2, 7.2, 7.2]
    assert ch.x_label == "iter →"
    # the tiles are the iteration's, not an update's: a DAgger run has no steps and no steps/s
    assert page.m_upd.name_label.text() == "반복" and page.m_upd.value.text() == "3/6"
    assert page.m_steps.name_label.text() == "샘플" and page.m_sps.name_label.text() == "손실"
    assert page.m_sps.value.text() == "0.0233"


def test_switching_between_the_two_kinds_rebuilds_the_grid(page, tmp_path):
    ppo = _run_with(tmp_path, "cl_p", PPO_JSONL)
    dag = _run_with(tmp_path, "cl_d", DAGGER_JSONL)
    for run, first in ((ppo, "충돌 / km"), (dag, "충돌 / km · student vs teacher"), (ppo, "충돌 / km")):
        page._current_run = str(run)
        page._tick(force=True)
        assert _titles(page)[0] == first
        assert page.charts_grid.count() == len(page._charts)


def test_a_blank_chart_says_which_file_was_missing(page, tmp_path):
    """The bug as reported: empty charts, and no way to tell a broken page from a run that left no
    record. `fl_a0_control_s701` and every `cl_orc_a*` are exactly this -- stdout went to the
    launching worker's own log directory, so the run directory holds checkpoints and nothing else."""
    d = _run_with(tmp_path, "cl_nothing")
    (d / "ppo_latest.pt").write_bytes(b"")
    page._current_run = str(d)
    page._tick(force=True)
    note = page.source_note.text()
    assert "progress.jsonl 없음" in note and "wandb output.log 없음" in note
    assert "progress.jsonl" in note.split("—")[-1]           # and what to do about it
    for ch in page._charts.values():
        assert ch.placeholder.startswith("progress.jsonl 없음")
        assert ch._series == [[]] or all(not s for s in ch._series)


def test_the_source_is_named_when_there_is_one(page, tmp_path):
    d = _run_with(tmp_path, "cl_ppo2", PPO_JSONL)
    page._current_run = str(d)
    page._tick(force=True)
    note = page.source_note.text()
    assert note.startswith("진행 기록: progress.jsonl") and "5개 지점" in note and "PPO" in note


def test_the_run_list_says_per_run_where_its_curve_comes_from(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "_console_jobs").mkdir(parents=True)
    _run_with(tmp_path, "has_jsonl", PPO_JSONL)
    _run_with(tmp_path, "has_console", files=[("console-train.log", LOG)])
    _run_with(tmp_path, "has_nothing")
    monkeypatch.setattr(T.catalog, "RUNS_DIR", str(runs))
    tags = {name: subtitle.rsplit(" · ", 1)[-1] for name, _p, subtitle, _mt in T.list_run_dirs(str(runs))}
    assert tags["has_jsonl"] == "기록 progress.jsonl"
    assert tags["has_console"] == "기록 console-train.log"
    assert tags.get("has_nothing") is None      # no curve and no checkpoint: not a run to watch yet


def test_progress_jsonl_wins_over_every_log_but_an_empty_one_does_not(tmp_path):
    """A run resumed under the new trainer has both. A `progress.jsonl` that exists but is still
    empty -- the first minute of a run -- must not hide a log with thirty updates in it."""
    d = _run_with(tmp_path, "cl_both", PPO_JSONL, files=[("console-train.log", LOG)])
    assert T.read_progress(str(d)).source_kind == "progress"
    (d / "progress.jsonl").write_text("")
    p = T.read_progress(str(d))
    assert p.source_kind == "console" and p.n_points == 1 and p.update == 12


def test_the_wandb_fallback_reads_the_layout_on_disk(tmp_path):
    """101 runs in `~/f1sim_runs` have one of these and nothing else; wandb 0.29 stopped writing
    them, which is how the dashboard went blank in the first place."""
    d = _run_with(tmp_path, "cl_wb")
    wb = d / "wandb" / "run-20260914_031725-w5h31ib5" / "files"
    wb.mkdir(parents=True)
    (wb / "output.log").write_text(LOG)
    (d / "wandb" / "latest-run").symlink_to(wb.parent)       # must not be read a second time
    assert T.wandb_output_logs(str(d)) == [str(wb / "output.log")]
    p = T.read_progress(str(d))
    assert p.source_kind == "wandb" and p.update == 12


def test_the_chart_reads_out_the_value_under_the_cursor(page, tmp_path):
    """A curve is the shape; the number the run is at is what gets written down."""
    d = _run_with(tmp_path, "cl_hover", PPO_JSONL)
    page._current_run = str(d)
    page._tick(force=True)
    ch = page._charts[("coll_per_km",)]
    ch.resize(300, 140)
    assert ch._readout(4) == "10"                            # the last point, as drawn
    ch.mouseMoveEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove, QtCore.QPointF(48.0, 70.0), QtCore.Qt.NoButton,
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier))
    assert ch._hover == 0 and ch._readout(ch._hover) == "50"
    ch.leaveEvent(None)
    assert ch._hover is None
