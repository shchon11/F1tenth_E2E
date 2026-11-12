"""The console's obstacle seed, 장애물 편집 and opponent choice (2026-11-03).

The user: "콘솔에서 왜 장애물 시드 못바꾸게해? 그리고 시드만 바꾸는게 아니라 내가 직접 장애물을 추가/제거/위치/
변경해서 바로 돌리게 할 수도 있었으면 좋겠어." and "상대차도 왜 forza eth밖에 선택 못해?".

What is pinned here:

* a seed is a layout -- the same pieces whatever the car count, and the same after a 리셋 (`prop_layout`);
* a layout written by hand is what the simulator collides with and the LiDAR sees, in the generator's own slots;
* the layout file round-trips and refuses what it cannot place;
* the window's seed box, editor and opponent bar end up in the `SessionConfig` the worker builds from;
* the worker's `CMD_SET_LAYOUT` holds a new layout and answers with it.
"""
import functools
import json
import math
import os
import sys
from collections import deque

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from f1sim import Config, maps                                                 # noqa: E402
from f1sim.gym_env import EnvConfig                                            # noqa: E402
from f1sim.learn import common                                                 # noqa: E402
from f1sim.viewer import prop_layout as PL                                     # noqa: E402
from f1sim.viewer.console import layout_file as LF                             # noqa: E402
from f1sim.viewer.console import protocol as P                                 # noqa: E402

TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=1)
def _track():
    return maps.load(TRACK)


def _env(envs=1, race=1, max_props=12, seed=3):
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 181
    e = EnvConfig(action_mode="direct", race_size=race, opponent="policy",          # self-play: no raceline needed
                  collision_mode="soft", compile_tracker=False, max_steps=40, speed_cap=6.0,
                  procedural_obstacles=1.0, procedural_density=1.5, procedural_max_props=max_props,
                  procedural_raceline_corridor="off", procedural_shared=True, movable_obstacles=True,
                  spawn_runway=3.0)
    env = common.make_env([_track()], envs, "cpu", e, cfg=cfg, seed=seed)
    env.sim.tid.fill_(0)
    return env


def _same(a, b, tol=1e-5):
    return len(a) == len(b) and all(p["sid"] == q["sid"] and abs(p["x"] - q["x"]) < tol and abs(p["y"] - q["y"]) < tol
                                    and abs(p["yaw"] - q["yaw"]) < tol for p, q in zip(a, b))


# ------------------------------------------------------------------ a seed is a layout
def test_a_seed_draws_the_same_layout_whatever_the_batch_and_another_seed_differs():
    a = _env(envs=1)
    b = _env(envs=4, seed=99)                    # another batch size, another simulator seed
    pa = PL.draw_seed(a.procedural, 4417, keep=12, tid=a.sim.tid[:1])
    pb = PL.draw_seed(b.procedural, 4417, keep=12, tid=b.sim.tid[:1])
    assert pa and _same(pa, pb)
    for row in range(b.B):                       # every row of the batch holds it
        assert _same(PL.read(b.procedural, row), pa)
    assert not _same(PL.draw_seed(a.procedural, 4418, keep=12, tid=a.sim.tid[:1]), pa)


def test_spare_slots_never_add_crates_to_a_seed_layout():
    """The console gives a session spare slots for 장애물 편집; a seed's draw keeps only the run's own count, and that
    is exactly the layout a generator of that many slots draws."""
    small = _env(max_props=6)
    big = _env(max_props=32)
    ps = PL.draw_seed(small.procedural, 7, keep=6, tid=small.sim.tid[:1])
    pb = PL.draw_seed(big.procedural, 7, keep=6, tid=big.sim.tid[:1])
    assert _same(ps, pb) and len(pb) <= 6
    assert all(int(s) < 0 for s in big.procedural.p_sid[0, 6:].tolist())


def test_a_held_layout_survives_a_reset():
    env = _env(envs=2, race=2)
    hold = PL.LayoutHold(env.procedural, 12, env.sim.tid[:1])
    pieces = hold.set_seed(31)
    hold.install()
    env.reset()
    assert _same(PL.read(env.procedural, 0), pieces) and _same(PL.read(env.procedural, 1), pieces)
    hold.release()
    env.reset()                                  # released: the generator draws again
    assert not _same(PL.read(env.procedural, 0), pieces)


# ------------------------------------------------------------------ a layout made by hand is a physical one
def _ahead(env, d):
    x, y, yaw = (float(v) for v in env.sim.state[0, :3])
    return x + d * math.cos(yaw), y + d * math.sin(yaw), yaw


def _forward_range(env):
    sim = env.sim
    _scan, scan_true, _t = sim.lidar.scan(sim.state[:, :3], None, sim.P, motion_distortion=False, tid=sim.tid,
                                          cars=None, eid=sim.eid)
    j = int(torch.argmin(sim.lidar.angles.abs()))
    return float(scan_true[0, j])


def _crate(proc):
    return LF.match_shape("wooden_crate", {"width": 0.30, "depth": 0.34, "height": 0.36}, PL.catalogue(proc))


def test_a_placed_crate_is_seen_by_the_lidar_and_touched_by_the_car_and_a_removed_one_is_gone():
    env = _env()
    env.reset()
    proc = env.procedural
    hold = PL.LayoutHold(proc, proc.C, env.sim.tid[:1]).install()
    hold.set_pieces([])
    clear = _forward_range(env)
    x, y, yaw = _ahead(env, 1.2)
    hold.set_pieces([{"sid": _crate(proc), "x": x, "y": y, "yaw": yaw + math.pi / 2}])
    seen = _forward_range(env)
    assert seen < min(clear, 1.2) - 0.05, (clear, seen)
    # slot 0 carries the crate's own half-planes, height and mass from the catalogue
    sid = _crate(proc)
    assert torch.equal(proc.p_n[0, 0], proc.sh_n[sid]) and float(proc.p_zhi[0, 0]) == float(proc.sh_z1[sid])
    assert float(proc.p_mass[0, 0]) == float(proc.sh_mass[sid]) > 0
    # on top of the car: the contact test finds it
    x0, y0, _ = _ahead(env, 0.0)
    hold.set_pieces([{"sid": sid, "x": x0, "y": y0, "yaw": 0.0}])
    assert float(env.sim._prop_contact(env.sim.state)[0][0]) > 0.05
    # removed: nothing to see or touch
    hold.set_pieces([])
    assert abs(_forward_range(env) - clear) < 1e-4
    assert float(env.sim._prop_contact(env.sim.state)[0][0]) <= 0.0
    assert int((proc.p_sid[0] >= 0).sum()) == 0 and bool((proc.p_zhi[0] < proc.p_zlo[0]).all())


def test_more_pieces_than_slots_is_refused_before_anything_changes():
    env = _env(max_props=4)
    proc = env.procedural
    hold = PL.LayoutHold(proc, 4, env.sim.tid[:1])
    before = hold.set_seed(5)
    with pytest.raises(ValueError, match="슬롯"):
        hold.set_pieces([{"sid": 0, "x": 0.0, "y": float(i), "yaw": 0.0} for i in range(5)])
    assert _same(hold.pieces, before) and _same(PL.read(proc, 0), before)


# ------------------------------------------------------------------ the file
def _catalogue():
    return PL.catalogue(_env().procedural)


def test_the_layout_file_round_trips_and_refuses_what_it_cannot_place(tmp_path):
    cat = _catalogue()
    pieces = [{"sid": cat[0]["id"], "x": 1.0, "y": 2.0, "yaw": 0.5},
              {"sid": cat[-1]["id"], "x": -3.25, "y": 0.125, "yaw": 4.0}]
    path = LF.save(str(tmp_path / "a.json"), LF.to_portable(pieces, cat), map_id="real/korea26", seed=4417)
    data = LF.load(path)
    assert data["map"] == "real/korea26" and data["seed_origin"] == 4417
    back = LF.from_portable(data["pieces"], cat)
    # the file names a shape, not a catalogue slot: two slots can hold the same shape (12 and 31 are both the 42x34 cm crate)
    shape = {e["id"]: (e["style"], e["dims"]) for e in cat}
    assert [shape[p["sid"]] for p in back] == [shape[p["sid"]] for p in pieces]
    assert back[1]["yaw"] == pytest.approx(math.atan2(math.sin(4.0), math.cos(4.0)))
    bad = dict(data["pieces"][0], style="piano")
    with pytest.raises(ValueError, match="목록에 없는"):
        LF.from_portable([bad], cat)
    (tmp_path / "x.json").write_text(json.dumps({"format": "something-else", "pieces": []}))
    with pytest.raises(ValueError):
        LF.load(str(tmp_path / "x.json"))
    with pytest.raises(ValueError):
        LF.clean_portable([{"style": "wooden_crate", "dims": {}, "x": "a", "y": 0}])


def test_every_catalogue_shape_has_a_korean_name():
    for e in _catalogue():
        assert e["style"] in LF.STYLE_LABEL and LF.STYLE_LABEL[e["style"]] in e["label"]


# ------------------------------------------------------------------ the worker's command
class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


def test_the_worker_holds_a_new_seed_or_pieces_and_answers_with_them():
    from f1sim.viewer import sim_worker
    w = sim_worker.SimWorker(_N(), _N())
    env = _env(envs=2, race=2)
    env.reset()
    session = {"env": env, "gg": deque(), "obs": None}
    st = w._set_layout(session, {"seed": 4417, "reset": True})
    assert st["layout_seed"] == 4417 and not st["layout_custom"] and st.get("reset") and not st.get("error")
    assert _same(st["layout"], PL.read(env.procedural, 0))
    cat = session["prop_catalogue"]
    x, y, yaw = _ahead(env, 1.5)
    portable = [{"style": "wooden_crate", "dims": {"width": 0.30, "depth": 0.34, "height": 0.36},
                 "x": x, "y": y, "yaw": yaw}]
    st = w._set_layout(session, {"pieces": portable, "reset": True})
    assert st["layout_custom"] and len(st["layout"]) == 1
    assert LF.to_portable(st["layout"], cat)[0]["style"] == "wooden_crate"
    env.reset()                                         # 리셋: the hand-made layout is put back
    assert len(PL.read(env.procedural, 0)) == 1
    st = w._set_layout(session, {"pieces": [dict(portable[0], style="piano")]})
    assert "목록에 없는" in st["error"] and len(st["layout"]) == 1           # refused, layout unchanged
    st = w._set_layout({"env": type("E", (), {"procedural": None})()}, {"seed": 1})
    assert "장애물 생성기" in st["error"]


def test_the_config_carries_the_seed_and_the_hand_made_layout():
    cfg = P.SessionConfig(procedural=True, layout_seed=12, prop_layout=[{"style": "s", "dims": {}, "x": 0, "y": 0}])
    back = P.SessionConfig.from_dict(cfg.to_dict())
    assert back.layout_seed == 12 and back.prop_layout == cfg.prop_layout
    assert cfg.affects_simulation(P.SessionConfig(procedural=True, layout_seed=13))
    assert P.SessionConfig().layout_seed is None and P.SessionConfig().prop_layout is None   # scripts: as before


# ------------------------------------------------------------------ the window
pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets                                                     # noqa: E402


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
def window(qapp):
    from f1sim.viewer.console.window import ConsoleWindow
    w = ConsoleWindow()
    w.resize(1400, 900)
    w.show()
    qapp.processEvents()
    yield w
    w.allow_close()
    w.deleteLater()
    qapp.processEvents()


def test_the_seed_box_is_the_seed_the_session_draws(window):
    w = window
    w._on_condition_picked("props")
    w.seed_box.setValue(4417)
    w._on_seed_entered()
    cfg = w.current_config()
    assert cfg.procedural and cfg.layout_seed == 4417 and cfg.prop_layout is None
    w._on_seed_roll()
    lo, hi = w.SEED_RANGE
    assert lo <= w.current_config().layout_seed <= hi and w.current_config().layout_seed == w._seed()
    w._on_condition_picked("solo")
    assert w.current_config().layout_seed is None and not w.current_config().procedural


def test_a_seed_change_on_a_running_session_is_sent_live_not_rebuilt(window):
    w = window
    w._on_condition_picked("props")
    sent, started = [], []
    w.layout_requested.connect(sent.append)
    w.start_requested.connect(started.append)
    w._running_facts = {"procedural": True}
    w._prop_catalogue = _catalogue()
    w.state = P.STATE_RUNNING
    w.seed_box.setValue(77)
    w._on_seed_entered()
    assert sent == [{"seed": 77, "reset": True}] and not started


def test_the_opponent_bar_writes_the_table_the_session_races(window):
    w = window
    keys = [w.combo_opp_kind.itemData(i) for i in range(w.combo_opp_kind.count())]
    for want in ("forzaeth_pred", "forzaeth", "lane_switch", "interactive", "raceline", "ftg", "disparity", "self",
                 "mix:planners"):
        assert want in keys, want
    w._on_condition_picked("race")
    cfg = w.current_config()
    assert cfg.cars_per_race == 2 and cfg.opponent_slots == [{"kind": "forzaeth_pred", "speed_scale": [0.7, 1.0]}]
    w.combo_opp_kind.setCurrentIndex(keys.index("lane_switch"))
    w._on_opp_changed()
    w.spin_opp_lo.setValue(0.8)
    w.spin_opp_hi.setValue(0.9)
    w.spin_opp_n.setValue(2)
    cfg = w.current_config()
    assert cfg.cars_per_race == 3
    assert cfg.opponent_slots == [{"kind": "lane_switch", "speed_scale": [0.8, 0.9]}] * 2
    assert w.cb_opponents.isVisible()
    # a pick's own table shows in the bar
    w._apply_pick_settings(type("Pick", (), {"settings": {"cars_per_race": 2, "obstacle": "bare",
                                                          "opponent_slots": [{"kind": "ftg", "speed_scale": [0.6, 0.6]}]}})())
    assert w.combo_opp_kind.currentData() == "ftg" and w.spin_opp_lo.value() == pytest.approx(0.6)
    w._on_condition_picked("solo")
    assert not w.cb_opponents.isVisible() and w.current_config().opponent_slots is None


def test_editing_saving_and_loading_a_layout_reaches_the_config(window, tmp_path):
    w = window
    cat = _catalogue()
    w._on_condition_picked("props")
    w._layout_now = {"layout": [{"sid": cat[3]["id"], "x": 1.0, "y": 1.0, "yaw": 0.0}], "layout_seed": 9,
                     "prop_slots": 32}
    w._prop_catalogue = cat
    ed = w.layout_ed
    ed.begin(w._layout_now["layout"], cat, 32, seed=9)
    crate = LF.match_shape("wooden_crate", {"width": 0.30, "depth": 0.34, "height": 0.36}, cat)
    ed.set_shape(crate)                      # nothing picked: the shape for 놓기
    i = ed.add(4.0, 0.0)
    assert ed.pieces[i]["sid"] == crate and ed.selected == i
    ed.move(i, 5.0, 0.5)
    ed.rotate(90.0)                          # no centre line given: put down at 0, now a quarter turn
    assert (ed.pieces[i]["x"], ed.pieces[i]["y"]) == (5.0, 0.5) and ed.pieces[i]["yaw"] == pytest.approx(math.pi / 2)
    ed.select(0)
    drum = next(e["id"] for e in cat if e["style"] == "steel_drum")
    ed.set_shape(drum)                       # picked: that piece changes
    assert ed.pieces[0]["sid"] == drum
    ed.remove(0)
    assert len(ed.pieces) == 1
    assert ed.undo() and len(ed.pieces) == 2 and ed.pieces[0]["sid"] == drum
    ed.clear()
    assert not ed.pieces and ed.undo() and len(ed.pieces) == 2
    # hit test at a piece's centre, and not beside it
    p = ed.pieces[1]
    assert ed.hit(lambda z: (p["x"], p["y"])) == 1 and ed.hit(lambda z: (p["x"] + 3.0, p["y"])) is None
    w._editing = True
    path = w.save_layout_to(str(tmp_path / "edited.json"))
    w._editing = False
    data = LF.load(path)
    assert len(data["pieces"]) == 2 and data["seed_origin"] == 9
    n = w.load_layout_from(path)
    cfg = w.current_config()
    assert n == 2 and cfg.procedural and cfg.layout_seed is None
    assert [q["style"] for q in cfg.prop_layout] == [q["style"] for q in data["pieces"]]
    assert "직접" in w.layout_chip.text() or "edited.json" in w.layout_chip.text()
    w._on_seed_roll()                        # a seed again: the hand-made layout gives way
    assert w.current_config().prop_layout is None


def test_the_edit_view_maps_the_floor_to_the_screen_and_back(window):
    vp = window.viewport
    vp.resize(900, 600)
    vp._edit_cam = [3.0, -2.0, 20.0, 0.0, 1.0]
    for x, y in ((3.0, -2.0), (6.5, 1.0), (-1.0, -6.0)):
        sx, sy = vp.screen_point(x, y, 0.0)
        gx, gy = vp.floor_point(sx, sy, 0.0)
        assert (gx, gy) == pytest.approx((x, y), abs=1e-6)
    cx, cy = vp.screen_point(3.0, -2.0, 0.0)
    assert (cx, cy) == pytest.approx((vp.width() / 2, vp.height() / 2), abs=1e-6)
    # a crate's top is drawn further from the centre than its foot: picking tests its upper half
    sx0, _ = vp.screen_point(8.0, -2.0, 0.0)
    sx1, _ = vp.screen_point(8.0, -2.0, 0.3)
    assert sx1 != sx0


def test_the_controller_sends_set_layout_and_settles_its_ack(window):
    from f1sim.viewer.console.session import SessionController
    sent = []
    c = SessionController.__new__(SessionController)
    c.window = window
    c._active_gen = 4
    c.send = lambda kind, for_gen=None, **f: sent.append((kind, for_gen, f))
    SessionController.set_layout(c, {"seed": 5})
    SessionController.set_layout(c, {"pieces": [{"style": "x"}], "reset": False})
    assert sent[0] == (P.CMD_SET_LAYOUT, 4, {"reset": True, "seed": 5})
    assert sent[1] == (P.CMD_SET_LAYOUT, 4, {"reset": False, "pieces": [{"style": "x"}]})
    window.settle_layout({"layout": [{"sid": 0, "x": 0, "y": 0, "yaw": 0}], "layout_seed": 5, "layout_custom": False,
                          "prop_slots": 32, "prop_keep": 18})
    assert window._layout_now["layout_seed"] == 5 and len(window._layout_now["layout"]) == 1
    window.settle_layout({"error": "안 됨"})
    assert window._layout_now["layout_seed"] == 5 and "안 됨" in window.status_text.text()
