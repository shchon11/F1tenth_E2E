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
from PyQt5 import QtCore, QtWidgets

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


def test_the_default_is_the_training_split_itself(picker):
    """Not a reconstruction of it. The curated list has per-map obstacle seeds by design, and a run
    launched from the default has to be the run the recipes were measured with."""
    assert picker.spec() == "train"
    assert set(_checked(picker)) == set(tracks.split_tracks("train"))
    assert picker.count() == 149
    assert "149개 변형" in picker.summary.text()


def test_one_click_selects_the_whole_training_set(picker):
    picker.clear_selection()
    assert picker.spec() == "" and _checked(picker) == []
    picker.btn_all_train.click()
    assert picker.spec() == "train"


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
    assert picker.spec() == "real/icra22#line:*"
    # `*` is a request for N placements, not one track
    picker.spin_draws.setValue(8)
    assert picker.count() == 8
    picker.combo_seed.setCurrentIndex(picker.combo_seed.findData("fixed"))
    picker.spin_fixed.setValue(44)
    assert picker.spec() == "real/icra22#line:44"
    assert picker.count() == 1


def test_an_obstacle_a_track_cannot_carry_is_left_out_rather_than_emitted(picker):
    """`rt:Monza+obs3` raises in the loader. The picker must not build the name."""
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
    assert picker.spec() == "real/bb22-3#edge:*"


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
    assert picker.spec() == "train"


# ================================================================ the argv it builds
@pytest.fixture
def form(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    f = T.RecipeForm()
    try:
        yield f
    finally:
        f.deleteLater()


def test_the_default_recipe_emits_the_split_name(form):
    _name, argv, _dev = form.argv()
    assert argv[argv.index("--tracks") + 1] == "train"
    assert argv[argv.index("--obstacle-draws") + 1] == "8"


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
    assert s.recipe == "원본 레이스 레시피 (권장)"
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
    """Another branch is training with a `+hard<seed>` obstacle family this registry has never
    heard of. An unknown name is data, not an error: it is counted, and the map under it is named
    even though the suffix cannot be."""
    argv = list(RECORDED)
    argv[argv.index("--tracks") + 1] = "real:icra2022+hard1,real:icra2022+hard1~rev,weird_name"
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
