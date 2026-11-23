"""The console's training requests (11-24): the request format the page writes and the training PC's queue takes,
the presets built from a run's args.json, the queue's launch arguments and its reading of the exam files, and the
page's form turned into a request. No GPU, no training."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from f1sim.learn import train_requests as TR                                  # noqa: E402

TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "research", "plan-output-2026-11-01", "tools"))
ANALYSIS = os.path.normpath(os.path.join(TOOLS, "..", "out", "analysis"))


def _start(**kw):
    r = {"action": "start", "name": "scan31c", "world": "plan_world_korea26scan_qual.json", "init": "", "updates": 1500,
         "gpu": "both", "exam": "lab3", "flags": [["--steer-std-floor", "0.3"], ["--mem-keep-log", ""]]}
    r.update(kw)
    return r


# ------------------------------------------------------------------ validation
def test_good_requests_pass():
    TR.check_request(_start())
    TR.check_request(_start(init="scan31b/plan_u525.pt", updates=1500, gpu="0", exam="none", replace=["scan31b"]))
    TR.check_request({"action": "stop", "name": "scan31b"})


@pytest.mark.parametrize("bad, why", [
    ({"action": "run"}, "action"),
    ({"name": "Scan31c"}, "이름"),
    ({"name": "a"}, "이름"),
    ({"name": "x; rm -rf ~"}, "이름"),
    ({"world": "../plan_world_x.json"}, "세계"),
    ({"world": "plan_world_x.yaml"}, "세계"),
    ({"init": "/abs/plan_u1.pt"}, "출발"),
    ({"init": "scan31b/../../x/plan_u1.pt"}, "출발"),
    ({"updates": 0}, "끝 update"),
    ({"init": "scan31b/plan_u525.pt", "updates": 525}, "출발 u525"),
    ({"gpu": "2"}, "GPU"),
    ({"exam": "core"}, "시험"),
    ({"replace": ["$(x)"]}, "멈출"),
    ({"flags": [["-x", "1"]]}, "인자 이름"),
    ({"flags": [["--lr", "1;reboot"]]}, "쓸 수 없는"),
    ({"flags": [["--lr", "1 2"]]}, "쓸 수 없는"),
    ({"flags": [["--lr"]]}, "[이름, 값]"),
    ({"flags": [["--init", "x"]]}, "칸으로 정함"),
    ({"flags": [["--dist-world", "3"]]}, "칸으로 정함"),
    ({"flags": [["--name", "y"]]}, "칸으로 정함"),
    ({"flags": [["--check-args", ""]]}, "칸으로 정함"),
])
def test_bad_requests_refused(bad, why):
    with pytest.raises(ValueError, match=why if why != "[이름, 값]" else r"\[이름, 값\]"):
        TR.check_request(_start(**bad))


def test_seed_is_not_launcher_owned():
    TR.check_request(_start(flags=[["--seed", "7"]]))


def test_write_request_is_atomic_and_ordered(tmp_path):
    p1 = TR.write_request(_start(at=1000.0), str(tmp_path))
    p2 = TR.write_request({"action": "stop", "name": "scan31b", "at": 1000.5}, str(tmp_path))
    files = sorted(os.listdir(TR.req_dir(str(tmp_path))))
    assert [os.path.basename(p) for p in (p1, p2)] == files            # name order = time order, no .tmp left
    assert json.load(open(p1))["at"] == 1000.0
    with pytest.raises(ValueError):
        TR.write_request(_start(name="Bad"), str(tmp_path))
    assert len(os.listdir(TR.req_dir(str(tmp_path)))) == 2


def test_read_status_missing_is_empty(tmp_path):
    assert TR.read_status(str(tmp_path)) == {}


# ------------------------------------------------------------------ flags text
def test_flags_text_round_trip():
    fl = [["--horizon", "64"], ["--horizon-auto-down", ""], ["--lr", "-1"], ["--shaping-fade", "100,150"]]
    assert TR.parse_flags_text(TR.flags_text(fl)) == fl
    assert TR.parse_flags_text(TR.flags_text(fl, "\n")) == fl
    with pytest.raises(ValueError, match="앞에"):
        TR.parse_flags_text("64 --horizon")
    with pytest.raises(ValueError):
        TR.parse_flags_text("--lr 1 2")                                  # a second bare word
    with pytest.raises(ValueError, match="칸으로 정함"):
        TR.parse_flags_text("--world x.json")


def test_merge_flags_later_wins():
    assert TR.merge_flags([["--a", "1"], ["--b", ""]], [["--a", "2"], ["--c", "3"]]) == \
        [["--a", "2"], ["--b", ""], ["--c", "3"]]


# ------------------------------------------------------------------ presets
def _fake_run(runs, name, args, cks=()):
    d = runs / name
    d.mkdir(parents=True)
    (d / "args.json").write_text(json.dumps({"args": args}))
    for u in cks:
        (d / f"plan_u{u}.pt").write_bytes(b"")
    return d


def test_preset_only_what_differs_from_the_launcher(tmp_path):
    world = tmp_path / "plan_world_t.json"
    world.write_text(json.dumps({"contact_scale": 0.82}))
    base = {"horizon": 128, "minibatches": 16, "lr": 0.0003, "steer_std_floor": 0.0, "mem_keep_s": 0.0,
            "mem_keep_log": False, "noise_corr_ramp": "200,250", "contact_scale": None, "compile_leaves": True,
            "shaping_fade": ""}
    args = dict(base, name="r1a", world=str(world), updates=900, envs=608, seed=1, dist_world=2, init="",
                horizon=64, steer_std_floor=0.3, mem_keep_s=30.0, mem_keep_log=True, contact_scale=0.82,
                shaping_fade="100,150", gone_flag=5)
    _fake_run(tmp_path, "r1a", args, cks=(25, 50, 100))
    sc = TR.preset("r1a", "scratch", str(tmp_path), base)
    assert sc["world"] == "plan_world_t.json" and sc["updates"] == 900 and "init" not in sc
    assert dict(sc["flags"]) == {"--horizon": "64", "--steer-std-floor": "0.3", "--mem-keep-s": "30.0",
                                 "--mem-keep-log": "", "--shaping-fade": "100,150"}
    co = TR.preset("r1a", "continue", str(tmp_path), base)
    assert co["init"] == "r1a/plan_u100.pt" and co["offset"] == 100
    # without a base: every set value, launcher-owned keys and the world's own contact weight left out
    full = dict(TR.preset("r1a", "scratch", str(tmp_path)))["flags"]
    keys = {f for f, _ in full}
    assert "--gone-flag" in keys and "--contact-scale" not in keys
    assert not keys & {"--name", "--world", "--updates", "--envs", "--seed", "--dist-world", "--init"}
    assert TR.preset("nope", "scratch", str(tmp_path)) == {}


def test_contact_scale_kept_when_not_the_worlds(tmp_path):
    world = tmp_path / "plan_world_t.json"
    world.write_text(json.dumps({"contact_scale": 0.82}))
    _fake_run(tmp_path, "r2a", {"world": str(world), "contact_scale": 1.5, "updates": 10})
    assert ("--contact-scale", "1.5") in TR.preset("r2a", "scratch", str(tmp_path))["flags"]


def test_next_name(tmp_path):
    (tmp_path / "scan31c").mkdir()
    assert TR.next_name("scan31b", str(tmp_path)) == "scan31d"
    assert TR.next_name("scan31b", str(tmp_path), taken=["scan31d"]) == "scan31e"
    assert TR.next_name("scan31z", str(tmp_path)) == "scan32a"
    assert TR.next_name("qual", str(tmp_path)) == "qualx"


# ------------------------------------------------------------------ the queue
@pytest.fixture(scope="module")
def Q():
    sys.path.insert(0, TOOLS)
    import train_queue
    return train_queue


def test_launch_argv_positional_start(Q):
    sc = Q.launch_argv("x1a", "/w.json", 900, "", [("--mem-keep-s", "30"), ("--horizon-auto-down", "")])
    assert sc[2:7] == ["x1a", "/w.json", "900", "", "0"]           # launch_run.sh: checkpoint and update are positional
    assert sc[7:] == ["--mem-keep-s", "30", "--horizon-auto-down"]
    co = Q.launch_argv("x1b", "/w.json", 900, "x1a/plan_u325.pt", [("--mem-keep-s", "30"), ("--lr", "1e-4"),
                                                                   ("--noise-corr-ramp", "200,250"), ("--mem-keep-log", "")])
    assert co[4:7] == ["900", os.path.join(Q.RUNS, "x1a/plan_u325.pt"), "325"]
    assert co[7:] == ["--lr", "1e-4"]                                 # the from-scratch-only flags dropped


def test_parse_exam_text(Q):
    txt = ("[혼자 · 빈 트랙] 세션 64개\n  한 바퀴(세션 중앙값) 7.23 s [7.20-7.30]\n"
           "[혼자 · 상자] 세션 64개\n  세션당 세게 닿음: 벽 0.03 [0.00-0.08] · 상자 0.44 [0.30-0.60] · 차 0.00\n"
           "  한 바퀴(세션 중앙값) 7.57 s [7.5-7.6]\n")
    box = ("  0.5-0.9 s 전                  127번 | 세게 닿음   7.1 % [ 3.8-13.0]\n"
           "  첫째                           131번 | 세게 닿음   4.6 % [ 2.1- 9.6]\n"
           "  넷째 이후                        576번 | 세게 닿음   1.9 % [ 1.1- 3.4]\n")
    r = Q.parse_exam(txt, box)
    assert r["lap"] == 7.23 and r["lap_box"] == 7.57 and (r["box_walls"], r["box_boxes"]) == (0.03, 0.44)
    assert r["seen_05"] == [127, 7.1] and r["pass1"] == [131, 4.6] and r["pass4"] == [576, 1.9]
    assert "seen_09" not in r


@pytest.mark.skipif(not os.path.exists(os.path.join(ANALYSIS, "scan31b_u500_exam3.txt")),
                    reason="the scan31b exam files live on the training PC")
def test_exam_rows_on_real_files(Q):
    rows = {r["u"]: r for r in Q.exam_rows("scan31b", ANALYSIS)}
    r = rows[500]
    assert r["exam"] == "exam3" and r["lap"] == 7.23 and r["box_boxes"] == 0.06 and r["box_walls"] == 0.0
    assert r["seen_05"] == [60, 1.7] and r["seen_09"] == [694, 0.1] and r["seen_15"] == [187, 0.0]
    assert r["pass1"] == [116, 0.9] and r["pass2"] == [157, 0.0] and r["pass3"] == [157, 0.0] and r["pass4"] == [511, 0.2]
    assert sorted(rows) == [r["u"] for r in Q.exam_rows("scan31b", ANALYSIS)]


def test_plan_runs_ignores_wrappers_and_checks(Q, monkeypatch):
    ps = [(11, "bash -c cd /x && python3 -u -m f1sim.learn.plan_rl --name r9a --format v2"),
          (12, "python3 -u -m f1sim.learn.plan_rl --name r9a --format v2"),
          (13, "python3 -m f1sim.learn.plan_rl --name r9b --format v2 --check-args"),
          (14, "python3 -u -m f1sim.learn.plan_rl --name r9a --format v2 --dist-rank 1")]
    monkeypatch.setattr(Q, "procs", lambda: ps)
    assert Q.plan_runs() == {"r9a": [12, 14]}


def test_act_refuses_before_touching_anything(Q, monkeypatch, tmp_path):
    monkeypatch.setattr(Q, "procs", lambda: [(12, "python3 -u -m f1sim.learn.plan_rl --name busy1a --x 1")])
    monkeypatch.setattr(Q, "RUNS", str(tmp_path))
    monkeypatch.setattr(Q, "stop_run", lambda n: pytest.fail("must not stop anything"))
    monkeypatch.setattr(Q, "check_args", lambda cmd, gpu: (True, {}))
    monkeypatch.setattr(Q.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    assert Q.act({"action": "stop", "name": "idle1a"}, "t") == "돌고 있지 않음"
    assert Q.act(_start(name="new1a", flags=[]), "t").startswith("GPU 사용 중: busy1a")
    assert Q.act(_start(name="Bad"), "t").startswith("잘못된 요청")
    (tmp_path / "old1a").mkdir()
    assert "이미 있음" in Q.act(_start(name="old1a"), "t")
    monkeypatch.setattr(Q, "check_args", lambda cmd, gpu: (False, "unrecognized arguments: --foo 1"))
    assert Q.act(_start(name="new1a", replace=["busy1a"]), "t") == "인자 확인 실패: unrecognized arguments: --foo 1"


# ------------------------------------------------------------------ the page
@pytest.fixture(scope="module")
def qapp():
    _prev = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from f1sim.viewer.console import app as A
    yield QtWidgets.QApplication.instance() or A.create_app(["test"])
    if _prev is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev


def test_page_form_to_request(qapp, tmp_path):
    from f1sim.viewer.console.plan_train import PlanTrainPage
    st = {"at": 1.0, "host": "t", "busy": [], "taken": ["r1a", "r1b"], "next": "r1c", "order": ["r1b", "r1a"],
          "worlds": ["plan_world_a.json", "plan_world_b.json"], "knob_defaults": {"--lr": "0.0003"},
          "runs": {"r1b": {"alive": False, "checkpoints": [25, 50], "progress": {}, "exams": []},
                   "r1a": {"alive": False, "checkpoints": [100], "progress": {}, "exams": []}},
          "presets": [{"label": "p", "world": "plan_world_b.json", "updates": 700, "init": "r1b/plan_u50.pt",
                       "flags": [["--steer-std-floor", "0.3"], ["--horizon", "64"], ["--mem-chunk-rollout", ""]]}],
          "done": []}
    (tmp_path / "_train_status.json").write_text(json.dumps(st))
    page = PlanTrainPage(runs_dir=str(tmp_path))
    page.set_active(True)
    page.set_active(False)
    req = page.build_request()
    assert (req["name"], req["world"], req["init"], req["updates"], req["gpu"], req["exam"]) == \
        ("r1c", "plan_world_b.json", "r1b/plan_u50.pt", 700, "both", "lab3")
    assert dict((f, v) for f, v in req["flags"]) == {"--steer-std-floor": "0.3", "--horizon": "64", "--mem-chunk-rollout": ""}
    # a field beats the same flag in the text box; an empty field passes nothing
    page.ed_more.setPlainText("--horizon 64\n--lr 1e-3")
    page.knob_ed["--lr"].setText("2e-4")
    page.knob_ed["--steer-std-floor"].setText("")
    assert dict((f, v) for f, v in page.build_request()["flags"]) == {"--horizon": "64", "--lr": "2e-4"}
    page.ed_name.setText("r1a")
    with pytest.raises(ValueError, match="이미 있음"):
        page.build_request()
    page.ed_name.setText("r1c")
    page._send(page.build_request())
    names = os.listdir(TR.req_dir(str(tmp_path)))
    assert len(names) == 1 and names[0].endswith("_start_r1c.json")
    assert page.req_list.count() == 1                                   # shown as sent until the queue answers
    sent_at = json.load(open(os.path.join(TR.req_dir(str(tmp_path)), names[0])))["at"]
    st["done"] = [{"at": 2.0, "req_at": sent_at, "action": "start", "name": "r1c", "result": "GPU 사용 중: x"}]
    (tmp_path / "_train_status.json").write_text(json.dumps(st))
    page.refresh(force=True)
    assert page.req_list.count() == 1 and "GPU 사용 중" in page.req_list.item(0).text()


def test_window_menu_switches_pages(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_RUNS", str(tmp_path))
    from f1sim.viewer.console import window as W
    win = W.ConsoleWindow()
    try:
        assert win.centre_stack.count() == 2 and win.centre_stack.widget(1) is win.training
        texts = [a.text() for a in win.btn_menu.menu().actions()]
        assert texts == ["주행", "학습"]
        win.resize(1400, 900)
        win.show()
        win.btn_menu.menu().actions()[1].trigger()
        assert win.current_mode() == "train" and win.centre_stack.currentWidget() is win.training
        assert not win.pick_policy.isVisible() and not win._condbar.isVisible() and win.training._timer.isActive()
        win.set_mode("edit")                                  # the editor is not built: the driving page
        assert win.current_mode() == "drive" and win.pick_policy.isVisible() and win._condbar.isVisible()
        assert not win.training._timer.isActive()
        assert win._mode_actions["drive"].isChecked()
        win.set_mode("train")
        seen = []
        win._on_policy_picked = lambda n: seen.append(n)
        win.runs = [type("R", (), {"name": "watch_r1a"})()]
        win._watch_training_run("r1a")
        assert win.current_mode() == "drive" and seen == ["watch_r1a"]
        win._watch_training_run("r9z")                        # not in the list: says so, stays on driving
        assert seen == ["watch_r1a"] and "r9z" in win.status_text.text()
    finally:
        win.close()
        win.deleteLater()
