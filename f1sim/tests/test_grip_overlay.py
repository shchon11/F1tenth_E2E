"""The g-g overlay distinguishes measured truth from the automatic grip estimate."""
import os

import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets                                             # noqa: E402

from f1sim.viewer.console.overlays import DashPanel


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(qapp):
    p = DashPanel()
    p.resize(520, 220)
    p.set_data(5.2, 5.0, 4.6, 0.08, 0.1, 8.0,
               [(1.5, 2.0), (2.0, 3.0), (2.5, 4.0)], 0.82 * 9.81)
    qapp.processEvents()
    yield p
    p.deleteLater()
    qapp.processEvents()


def _estimate(**overrides):
    value = {"used_mu": 0.76, "q10": 0.58, "q50": 0.70, "q90": 0.80,
             "warm": True, "finite": True}
    value.update(overrides)
    return value


def test_warm_estimate_stores_q50_band_and_applied_mu(panel):
    panel.set_grip_estimate(_estimate())
    assert panel._grip_estimate == {"used_mu": 0.76, "q10": 0.58, "q50": 0.70, "q90": 0.80}
    assert panel._grip_status == "추정 μ 0.70 · 적용 0.76"


def test_overestimate_expands_the_shared_plot_scale(panel):
    panel.mu_g = 0.734 * 9.81
    panel.set_grip_estimate(_estimate(used_mu=0.58, q10=0.60, q50=0.91, q90=1.10))
    plot_lim, truth_lim, truth_r, estimate_radii, _ = panel._gg_radii(50.0)
    q10_r, q50_r, q90_r = estimate_radii
    assert plot_lim == pytest.approx(1.10 * 9.81)
    assert truth_lim == pytest.approx(0.734 * 9.81)
    assert q50_r > truth_r and q90_r > q50_r > q10_r
    assert q90_r == pytest.approx(50.0)


def test_cold_or_missing_evidence_has_no_estimate_circle(panel):
    panel.set_grip_estimate(_estimate(warm=False, finite=False, has_evidence=False))
    assert panel._grip_estimate is None
    assert panel._grip_status == "추정 대기"


def test_nonfinite_estimate_clears_the_previous_circle(panel):
    panel.set_grip_estimate(_estimate())
    panel.set_grip_estimate(_estimate(q50=float("nan"), finite=False))
    assert panel._grip_estimate is None
    assert panel._grip_status == "추정 대기"


def test_noninformative_frame_retains_last_known_evidence(panel):
    panel.set_grip_estimate(_estimate())
    panel.set_grip_estimate(_estimate(q50=0.42, used_mu=0.40, informative=0.0,
                                     has_evidence=1.0, warm=1.0, finite=1.0))
    assert panel._grip_estimate["q50"] == 0.70
    assert panel._grip_estimate["used_mu"] == 0.40
    assert panel._grip_status == "추정 μ 0.70 · 적용 0.40"


def test_float_wire_flags_hold_last_evidence(panel):
    panel.set_grip_estimate(_estimate())
    panel.set_grip_estimate(_estimate(q50=0.42, used_mu=0.40, informative=0.0,
                                     has_evidence=1.0, warm=1.0, finite=1.0))
    assert panel._grip_estimate["q50"] == 0.70
    assert panel._grip_estimate["used_mu"] == 0.40
    assert panel._grip_status == "추정 μ 0.70 · 적용 0.40"


def test_new_focus_noninformative_packet_keeps_circle_unavailable_but_shows_applied_mu(panel):
    panel.set_focus(0)
    panel.set_grip_estimate(_estimate())
    panel.set_focus(1)
    panel.set_grip_estimate(_estimate(q50=0.42, used_mu=0.58, informative=0.0,
                                     has_evidence=1.0, warm=1.0, finite=1.0))
    assert panel._grip_estimate is None
    assert panel._grip_status == "추정 대기 · 적용 0.58"


def test_cold_float_evidence_clears_before_noninformative_hold(panel):
    panel.set_grip_estimate(_estimate())
    panel.set_grip_estimate(_estimate(informative=0.0, has_evidence=0.0,
                                     warm=0.0, finite=1.0))
    assert panel._grip_estimate is None
    assert panel._grip_status == "추정 대기"


def test_focus_change_clears_the_previous_car_estimate(panel):
    panel.set_focus(0)
    panel.set_grip_estimate(_estimate())
    panel.set_focus(1)
    assert panel._grip_estimate is None
    assert panel._grip_status == "추정 대기"


def test_fault_hides_estimate_and_remains_held_until_informative_recovery(panel):
    panel.set_grip_estimate(_estimate())
    panel.set_grip_estimate(_estimate(used_mu=0.66, fault=1.0, informative=1.0))
    assert panel._grip_estimate is None
    assert panel._grip_fault is True
    assert panel._grip_status == "센서 확인 · 적용 0.66"

    panel.set_grip_estimate(_estimate(q50=0.42, used_mu=0.40, informative=0.0,
                                     has_evidence=1.0, warm=1.0, finite=1.0))
    assert panel._grip_estimate is None
    assert panel._grip_status == "센서 확인 · 적용 0.40"

    panel.set_grip_estimate(_estimate(q50=0.72, used_mu=0.70, informative=True,
                                     has_evidence=True))
    assert panel._grip_estimate["q50"] == 0.72
    assert panel._grip_fault is False


def test_clear_resets_the_estimate_for_a_new_session(panel):
    panel.set_grip_estimate(_estimate())
    panel.clear()
    assert panel._grip_estimate is None
    assert panel._grip_status == "추정 대기"
