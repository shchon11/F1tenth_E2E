"""The console's "내 차" panel (2026-10-17): who drives the ego car, as SessionConfig fields."""
import os

import pytest
from PyQt5 import QtWidgets

from f1sim.learn import common
from f1sim.viewer.console import ego_driver as ED
from f1sim.viewer.console.protocol import SessionConfig


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from f1sim.viewer.console import app as A
    yield QtWidgets.QApplication.instance() or A.create_app(["test"])


def test_the_panel_offers_exactly_the_kinds_the_worker_builds():
    assert ED.PLANNERS == common.EGO_PLANNER_KINDS and ED.EXTERNALS == common.EGO_EXTERNAL_KINDS


def test_policy_is_the_default_and_changes_nothing(qapp):
    p = ED.EgoDriverPanel()
    assert p.config() == {"ego_kind": "policy", "ego_weights": "", "ego_options": {}, "ego_limits": None,
                          "ego_tuned": True}
    assert p.problem() == ""
    SessionConfig(**p.config())                         # the fields exist


def test_tuned_settings_hide_the_manual_ones(qapp):
    """"맵별 튜닝 설정" (on by default, 2026-10-19): the method drives on its per-map tuned setting, so
    the manual pace / line / weights are not offered and a network kind needs no file picked."""
    p = ED.EgoDriverPanel()
    for k in ("ftg", "forzaeth", "tinylidarnet"):
        p.kind.setCurrentIndex(p.kind.findData(k))
        assert not p.tuned.isHidden() and p.pace.isHidden() and p.line.isHidden() and p.weights.isHidden()
        assert p.problem() == "" and p.config()["ego_tuned"] is True
    p.kind.setCurrentIndex(p.kind.findData("policy"))
    assert p.tuned.isHidden()


def test_each_kind_shows_only_its_own_settings_and_round_trips(qapp, tmp_path):
    p = ED.EgoDriverPanel()
    p.tuned.setChecked(False)                            # manual settings
    p.kind.setCurrentIndex(p.kind.findData("ftg")); p.pace.setValue(8.0)
    assert not p.pace.isHidden() and p.weights.isHidden() and p.line.isHidden()
    assert p.config()["ego_options"] == {"max_speed": 8.0}
    p.kind.setCurrentIndex(p.kind.findData("forzaeth")); p.line.setCurrentIndex(3)
    assert p.config()["ego_limits"] == [10.0, 8.0, 7.0] and p.config()["ego_options"] == {}
    p.kind.setCurrentIndex(p.kind.findData("tinylidarnet"))
    assert "가중치" in p.problem()
    w = tmp_path / "tln.pt"; w.write_bytes(b"x")
    p._weights_path = str(w); p._on_kind()
    cfg = p.config()
    assert cfg["ego_weights"] == str(w) and p.problem() == ""
    q = ED.EgoDriverPanel(); q.set_config(cfg)
    assert q.config() == cfg
    q.set_config({"ego_kind": "forzaeth", "ego_limits": [8.0, 7.0, 5.0]})
    assert q.config()["ego_limits"] == [8.0, 7.0, 5.0]
