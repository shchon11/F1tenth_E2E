"""The 상대차 table widget, on both pages that use it.

One widget, two pages, one spelling: the driving page builds a `SessionConfig` out of it and the
training page emits `--opp-slots` out of it, so what has to be pinned is that a spec survives the
round trip through the widgets unchanged, and that what comes out the other side is something the
trainer's own parser accepts.

The rest is the refusals. A table can name a checkpoint the loader will not open, a race size it
does not fit, or a driver kind this tree does not have; every one of those is a session that fails
after a minute of loading unless the page says so while it is still being typed.

Offscreen Qt (`QT_QPA_PLATFORM=offscreen`), like the other console tests; nothing here touches GL.
"""
import argparse
import dataclasses
import json
import os

import pytest
import torch
from PyQt5 import QtCore, QtWidgets

from f1sim import opponent_slots as osl
from f1sim.learn import opponent_config as oc
from f1sim.viewer.console import opponent_table as OT
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
def table(qapp):
    t = OT.OpponentSlotTable()
    try:
        yield t
    finally:
        t.deleteLater()


FULL = [
    {"kind": "raceline", "speed_scale": [0.6, 1.15], "label_grip": "conservative",
     "speed_cap": 4.5, "events": ["brake", "shift"], "event_rate": 1.5,
     "reactive": {"defend": 0.4, "yield": 0.25, "line": 0.5, "oblivious": 0.1},
     "spawn": "alongside"},
    {"kind": "self", "speed_scale": 0.8, "spawn": "behind"},
]


# ================================================================ the widget
def test_the_table_has_one_row_per_other_car(table):
    for grid in (2, 3, 4, 1):
        table.set_count(grid - 1)
        assert table.count() == grid - 1
        assert table.table.rowCount() == grid - 1


def test_a_spec_round_trips_through_the_widget(table):
    """Widget -> spec -> widget -> spec. Every field, not just the ones a default happens to keep."""
    table.set_slots(FULL)
    assert table.slot_dicts() == FULL
    once = table.slots()
    table.set_slots(once)
    assert table.slots() == once


def test_growing_the_table_keeps_the_rows_already_filled_in(table):
    table.set_slots([FULL[0]])
    table.set_count(3)
    assert table.slot_dicts()[0] == FULL[0]
    assert table.count() == 3


def test_copy_first_row_makes_every_car_the_same_car(table):
    table.set_slots(FULL)
    table.copy_first_row()
    rows = table.slot_dicts()
    assert rows[0] == rows[1] == FULL[0]


@pytest.mark.parametrize("key", [k for k, _t, _f in osl.PRESETS])
def test_every_preset_builds_a_table_the_trainer_would_accept(table, key, monkeypatch):
    """Whatever is (or is not) lying around in `~/f1sim_runs`: the recipe preset names the pool's
    checkpoints when they are on the machine, so the test pins the case where they are not."""
    monkeypatch.setattr(osl, "default_recipe_checkpoints", lambda: ())
    table.set_count(3)
    i = table.combo_preset.findData(key)
    table.combo_preset.setCurrentIndex(i)
    slots = table.slots()
    assert len(slots) == 3
    osl.validate_slots(slots, 4, require_files=False)
    assert table.problem(4) == ""


def test_a_kind_this_tree_does_not_have_is_listed_disabled_with_its_reason(table):
    table.set_count(1)
    combo = table._rows[0].kind
    i = combo.findData("interactive")
    assert i >= 0, "an unavailable kind was dropped from the list instead of being explained"
    model = combo.model()
    assert not (model.item(i).flags() & QtCore.Qt.ItemIsEnabled)
    assert "worker 17" in combo.itemData(i, QtCore.Qt.ToolTipRole)
    # and the row says it in its own text: greyed alone reads as "not applicable here"
    assert "병합 후 활성" in combo.itemText(i)


def test_the_cells_a_kind_cannot_carry_are_switched_off(table):
    table.set_count(1)
    row = table._rows[0]
    row.kind.setCurrentIndex(row.kind.findData("self"))
    assert not row.grip.isEnabled() and not row.rate.isEnabled() and not row.ckpt.isEnabled()
    assert all(not cb.isEnabled() for cb in row.events.values())
    row.kind.setCurrentIndex(row.kind.findData("raceline"))
    assert row.grip.isEnabled() and row.rate.isEnabled() and not row.ckpt.isEnabled()
    row.kind.setCurrentIndex(row.kind.findData("policy"))
    assert row.ckpt.isEnabled() and not row.grip.isEnabled()


def test_a_policy_row_without_a_checkpoint_is_an_objection_not_a_start(table):
    table.set_count(1)
    row = table._rows[0]
    row.kind.setCurrentIndex(row.kind.findData("policy"))
    assert "체크포인트" in table.problem(2)


def test_a_race_size_the_table_does_not_fit_is_an_objection(table):
    table.set_slots(FULL)
    assert "race_size" in table.problem(2)
    assert table.problem(3) == ""


# ================================================================ what a checkpoint says
def _tiny_checkpoint(path, extra=None):
    from f1sim.learn.model import ActorCritic, save_checkpoint
    from f1sim.learn.obs import ObsSpec
    from f1sim.params import Config
    spec = ObsSpec(n_beams=Config().lidar.n_beams)
    with torch.random.fork_rng():
        torch.manual_seed(20260915)
        model = ActorCritic(n_stack=spec.scan_stack, n_beams=spec.n_beams,
                            proprio_dim=spec.proprio_dim, priv_dim=21, act_dim=spec.act_dim)
    meta = {"spec": dataclasses.asdict(spec)}
    meta.update(extra or {})
    save_checkpoint(str(path), model, meta)
    return str(path)


def test_the_checkpoint_cell_names_the_arm_and_the_memory_kind(tmp_path):
    path = _tiny_checkpoint(tmp_path / "a.pt",
                            {"experiment": {"controller": {"arm": "fixed_low"}}})
    note = OT.describe_checkpoint(path)
    assert "fixed_low" in note and "피드포워드" in note
    assert OT.checkpoint_arm(path) == "fixed_low"


def test_an_oracle_checkpoint_is_refused_in_the_cell_with_the_loader_s_reason(tmp_path):
    path = _tiny_checkpoint(tmp_path / "oracle.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["meta"]["opp_token"] = True
    torch.save(ck, path)
    assert "특권" in OT.describe_checkpoint(path)


def test_a_missing_file_is_said_rather_than_waited_for(table, tmp_path):
    table.set_count(1)
    row = table._rows[0]
    row.kind.setCurrentIndex(row.kind.findData("policy"))
    row.set_checkpoint(str(tmp_path / "nope.pt"))
    assert "파일이 없습니다" in table.problem(2)


def test_the_arm_a_checkpoint_records_is_carried_into_the_spec(table, tmp_path):
    path = _tiny_checkpoint(tmp_path / "armed.pt",
                            {"experiment": {"controller": {"arm": "fixed_low"}}})
    table.set_count(1)
    row = table._rows[0]
    row.kind.setCurrentIndex(row.kind.findData("policy"))
    row.set_checkpoint(path)
    assert table.slots()[0].controller == "fixed_low"


# ================================================================ the two pages
def test_the_driving_page_emits_the_table_it_shows(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    from f1sim.viewer.console.window import ConsoleWindow
    w = ConsoleWindow()
    try:
        w.spin_grid.setValue(3)
        assert w.opp_table.count() == 2
        w.opp_table.set_slots(FULL)
        cfg = w.current_config()
        assert cfg.opponent == "slots" and cfg.opponent_slots == FULL
        assert [s.kind for s in cfg.slots()] == ["raceline", "self"]
        w.spin_grid.setValue(1)
        assert w.current_config().opponent_slots is None, "a solo session carries a table"
    finally:
        w.deleteLater()


def test_the_training_page_emits_a_flag_the_trainer_parses(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    form = T.RecipeForm()
    try:
        form.spin_race.setValue(3)
        form.chk_slots.setChecked(True)
        form.opp_table.set_slots(FULL)
        _name, argv, _dev = form.argv()
        # the flags the table replaces are gone, not merely overridden
        assert "--opponent" not in argv and "--opp-speed" not in argv
        assert "--mixed-teacher-frac" not in argv
        text = argv[argv.index("--opp-slots") + 1]
        assert json.loads(text) == FULL
        ap = oc.add_arguments(argparse.ArgumentParser())
        a, _unknown = ap.parse_known_args(argv[3:])
        oc.validate(a)                       # must not raise: the widget speaks the flag's language
        assert oc.env_kwargs(a)["opponent"] == "slots"
        assert [s.kind for s in oc.env_kwargs(a)["opponent_slots"]] == ["raceline", "self"]
    finally:
        form.deleteLater()


def test_the_training_page_leaves_the_recipes_alone_with_the_switch_off(qapp, tmp_path, monkeypatch):
    """The recipes are measured configurations. Off, the command must be the one it always was."""
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    form = T.RecipeForm()
    try:
        _name, argv, _dev = form.argv()
        assert "--opp-slots" not in argv
        assert argv[argv.index("--opponent") + 1] == "mixed"
        assert "--opp-speed" in argv and "--mixed-teacher-frac" in argv
    finally:
        form.deleteLater()


def test_the_recipe_preset_uses_the_pool_s_checkpoints_when_they_are_here(table, tmp_path, monkeypatch):
    """The preset is meant to *be* the training recipe's population, so where the recipe's
    checkpoints exist it names them rather than quietly substituting self-play for them."""
    ck = _tiny_checkpoint(tmp_path / "frozen.pt")
    monkeypatch.setattr(osl, "default_recipe_checkpoints", lambda: (ck,))
    table.set_count(3)
    table.combo_preset.setCurrentIndex(table.combo_preset.findData("recipe"))
    slots = table.slots()
    assert [s.kind for s in slots] == ["raceline", "self", "policy"]
    assert slots[2].checkpoint == ck
    assert slots[0].events == ("brake", "stop", "shift", "weave")
    assert slots[0].speed_scale == (0.6, 1.15)


def test_the_jobs_card_lists_one_line_per_slot(qapp):
    argv = ["python", "-m", "f1sim.learn.ppo", "--tracks", "train", "--race-size", "3",
            "--opp-slots", json.dumps(FULL), "--controller", "legacy"]
    job = T.Job(name="cl_slots", pid=os.getpid(), argv=argv, log="/tmp/x.log",
                started=0.0, run_dir="/tmp/r")
    s = T.summarize_job(job, now=60.0)
    assert "차량별 설정" in s.race_text and "1x raceline" in s.race_text
    rows = dict(s.lines())
    assert "상대차" in rows and "raceline" in rows["상대차"]
    assert len(s.slot_lines) == 2
