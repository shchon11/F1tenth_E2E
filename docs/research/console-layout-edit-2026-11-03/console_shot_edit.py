"""실제 콘솔(창 + SessionController + 워커 프로세스)을 화면 밖(offscreen)에서 띄워 11-03 기능을 찍는다:
장애물 시드 입력·다시 뽑기·같은 시드 재현, 장애물 편집(옮기기·놓기·돌리기·모양 바꾸기·지우기 → 이 배치로 달리기),
JSON 저장·불러오기, 혼자 조건에서 '장애물 놓기', ForzaETH 아닌 상대차와 1:1 경주.

인자: 출력 접두어 [정책 런 이름 (기본 pick_sst26h2)]
환경: QT_QPA_PLATFORM=offscreen, F1SIM_CONSOLE_PREFS 는 임시 파일로 (사용자의 ~/.f1sim/console.json 을 읽지도 쓰지도 않게).
"""
import json, math, os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("F1SIM_SRC", os.path.join(HERE, "..", "..", "..", "f1sim")))
from PyQt5 import QtCore, QtGui, QtWidgets
from f1sim.viewer.console.app import create_app
from f1sim.viewer.console.catalog import list_runs
from f1sim.viewer.console.session import SessionController
from f1sim.viewer.console.window import ConsoleWindow
from f1sim.viewer.console.protocol import STATE_PAUSED, STATE_RUNNING


def main():
    out = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else "pick_sst26h2"
    app = create_app(["shot"])
    w = ConsoleWindow(); c = SessionController(w)
    w.set_runs(list_runs()); w.set_maps(c._maps); w.resize(1600, 950); w.show()
    QtCore.QTimer.singleShot(0, c.start_worker)
    vp = w.viewport

    def step(t):
        end = time.time() + t
        while time.time() < end:
            app.processEvents(); time.sleep(0.01)

    def wait(pred, t=180.0, what=""):
        end = time.time() + t
        while time.time() < end:
            if pred():
                return True
            step(0.1)
        print("  !! 기다림 시간 초과:", what, "| 상태", w.state, flush=True)
        return False

    def shot(name):
        p = f"{out}_{name}.png"
        w.grab().save(p)
        print("  찍음", p, flush=True)

    def mouse(kind, x, y, button=QtCore.Qt.LeftButton, mods=QtCore.Qt.NoModifier):
        t = {"press": QtCore.QEvent.MouseButtonPress, "move": QtCore.QEvent.MouseMove,
             "release": QtCore.QEvent.MouseButtonRelease}[kind]
        b = QtCore.Qt.NoButton if kind == "move" else button
        ev = QtGui.QMouseEvent(t, QtCore.QPointF(x, y), b, button if kind != "release" else QtCore.Qt.NoButton, mods)
        QtWidgets.QApplication.sendEvent(vp, ev)
        step(0.05)

    def layout():
        return [(p["sid"], round(p["x"], 4), round(p["y"], 4)) for p in (w._layout_now.get("layout") or [])]

    wait(lambda: getattr(w.maps, "ready", False), 120, "맵 목록")
    step(1)
    w._on_policy_picked(want); step(1)
    w._on_condition_picked("props"); step(0.5)
    # -- 1. 시드를 직접 넣고 시작
    w.seed_box.setValue(4417); w._on_seed_entered(); step(0.3)
    cfg = w.current_config()
    print("1. 설정: 정책", want, "| 맵", cfg.map_name, "| 시드", cfg.layout_seed, "| 학습 장애물", cfg.procedural, flush=True)
    w._on_start()
    wait(lambda: w.state == STATE_RUNNING, 240, "주행 시작")
    step(6)
    lay_a = layout()
    print("   배치(시드 4417):", len(lay_a), "개 | 칩:", w.layout_chip.text(), flush=True)
    shot("1seed4417")
    # -- 2. 다시 뽑기 (주행 중, 세션을 다시 만들지 않고)
    gen0 = c._active_gen
    w._on_seed_roll()
    new_seed = w._seed()
    wait(lambda: w._layout_now.get("layout_seed") == new_seed, 30, "다시 뽑기 응답")
    step(5)
    lay_b = layout()
    print(f"2. 다시 뽑기 → 시드 {new_seed}: {len(lay_b)}개, 같은 세션 {c._active_gen == gen0}, 배치 달라짐 {lay_b != lay_a}", flush=True)
    shot("2reroll")
    # -- 3. 시드를 다시 4417 로 → 같은 배치
    w.seed_box.setValue(4417); w._on_seed_entered()
    wait(lambda: w._layout_now.get("layout_seed") == 4417, 30, "시드 4417 응답")
    step(3)
    print("3. 시드 4417 다시: 처음과 같은 배치", layout() == lay_a, flush=True)
    # -- 4. 장애물 편집
    w._on_edit_layout()
    wait(lambda: w.state == STATE_PAUSED, 20, "일시정지")
    step(1.5)
    ed = w.layout_ed
    print("4. 편집 시작: 조각", len(ed.pieces), "| 칸", ed.slots, "| 카메라", vp.camera,
          "| 일시정지 버튼 켜짐", w.btn_pause.isEnabled(), "| 시작 버튼 켜짐", w.btn_start.isEnabled(),
          "| 상태줄:", w.status_text.text()[:40], flush=True)
    shot("4edit_open")
    # 4a. 첫 장애물을 1 m 옆으로 끌기
    p0 = dict(ed.pieces[0])
    h0 = float(ed.shape(p0["sid"])["height"])
    sx, sy = vp.screen_point(p0["x"], p0["y"], 0.6 * h0)
    tx, ty = vp.screen_point(p0["x"] + 1.0, p0["y"], 0.0)
    gx, gy = vp.screen_point(p0["x"], p0["y"], 0.0)
    mouse("press", sx, sy); mouse("move", sx + (tx - gx) * 0.5, sy + (ty - gy) * 0.5); mouse("move", sx + (tx - gx), sy + (ty - gy))
    mouse("release", sx + (tx - gx), sy + (ty - gy))
    p0b = ed.pieces[0]
    print(f"   끌기: ({p0['x']:.2f},{p0['y']:.2f}) → ({p0b['x']:.2f},{p0b['y']:.2f}), 골라짐 {ed.selected}", flush=True)
    # 4b. 돌리기, 모양 바꾸기 (철제 드럼)
    w._edit_key("ccw"); w._edit_key("ccw")
    drum = next(e["id"] for e in ed.catalogue if e["style"] == "steel_drum" and not e["small"])
    i = w.layout_panel.combo_shape.findData(drum); w.layout_panel.combo_shape.setCurrentIndex(i); w.layout_panel._on_shape(i)
    print("   돌리기 30°, 모양 →", ed.label(ed.pieces[0]["sid"]), flush=True)
    # 4c. 두 번째 장애물 지우기
    n_before = len(ed.pieces)
    ed.select(1); w._edit_key("delete")
    print("   지우기:", n_before, "→", len(ed.pieces), flush=True)
    # 4d. 차 앞 2.5 m 에 나무 상자 놓기 (＋ 놓기 → 클릭)
    fr = c.buffer.latest
    x, y, yaw = float(fr["x"][0]), float(fr["y"][0]), float(fr["yaw"][0])
    ax, ay = x + 2.5 * math.cos(yaw), y + 2.5 * math.sin(yaw)
    crate = next(e["id"] for e in ed.catalogue if e["style"] == "wooden_crate")
    ed.select(None); ed.set_shape(crate)
    w.layout_panel.btn_add.click()
    px, py = vp.screen_point(ax, ay, 0.0)
    mouse("press", px, py); mouse("release", px, py)
    w._edit_key("escape")
    print(f"   놓기: 차 앞 ({ax:.2f},{ay:.2f}) → 조각 {len(ed.pieces)}개, 마지막 {ed.label(ed.pieces[-1]['sid'])} "
          f"({ed.pieces[-1]['x']:.2f},{ed.pieces[-1]['y']:.2f})", flush=True)
    ed.select(len(ed.pieces) - 1)
    step(0.8)
    shot("5edited")
    # 4e. 휠로 확대해서 한 장
    for _ in range(4):
        ev = QtGui.QWheelEvent(QtCore.QPointF(px, py), QtCore.QPointF(px, py), QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
                               QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
        QtWidgets.QApplication.sendEvent(vp, ev); step(0.05)
    step(0.6)
    shot("5edited_zoom")
    edited = ed.portable()
    saved = w.save_layout_to(f"{out}_layout.json")
    print("   저장:", saved, len(json.load(open(saved))["pieces"]), "개", flush=True)
    # -- 5. 이 배치로 달리기
    w._on_edit_run()
    wait(lambda: w._layout_now.get("layout_custom") and len(w._layout_now.get("layout") or []) == len(edited), 30,
         "편집 배치 적용")
    wait(lambda: w.state == STATE_RUNNING, 20, "재개")
    step(2.0)
    shot("6run_edited_2s")
    step(5)
    print("5. 편집 배치로 주행: 조각", len(w._layout_now.get("layout") or []), "| 직접 편집", w._layout_now.get("layout_custom"),
          "| 상태", w.state, "| 칩:", w.layout_chip.text(), flush=True)
    shot("6run_edited_7s")
    # -- 6. 불러오기 (다른 시드로 바꾼 뒤 저장한 파일을 다시)
    w.seed_box.setValue(1234); w._on_seed_entered()
    wait(lambda: w._layout_now.get("layout_seed") == 1234, 30, "시드 1234")
    n = w.load_layout_from(saved)
    wait(lambda: w._layout_now.get("layout_custom") and len(w._layout_now.get("layout") or []) == n, 30, "불러온 배치")
    step(2)
    print("6. 불러오기:", n, "개, 적용됨", w._layout_now.get("layout_custom"), flush=True)
    # -- 7. ForzaETH 아닌 상대차와 1:1 경주 (lane-switch, 0.8~0.9배)
    for kind, tag in (("lane_switch", "7race_lane"), ("ftg", "8race_ftg")):
        w._on_stop(); wait(lambda: w.state not in (STATE_RUNNING, STATE_PAUSED), 30, "정지")
        step(1)
        w._on_condition_picked("race")
        i = w.combo_opp_kind.findData(kind); w.combo_opp_kind.setCurrentIndex(i); w._on_opp_changed()
        w.spin_opp_lo.setValue(0.8); w.spin_opp_hi.setValue(0.9)
        cfg = w.current_config()
        print(f"7. 경주 설정: 차 {cfg.cars_per_race}대, 상대 {cfg.opponent_slots}", flush=True)
        w._on_start()
        ok = wait(lambda: w.state == STATE_RUNNING, 240, "경주 시작")
        step(8)
        f = c.last_facts
        print(f"   주행: {ok}, 상대 {f.get('opponent_mix')} | 표 {f.get('opponent_slots')} | 차 {f.get('total_cars')}", flush=True)
        fr = c.buffer.latest
        if fr is not None:
            print(f"   차 위치: " + ", ".join(f"#{i} ({float(fr['x'][i]):.1f},{float(fr['y'][i]):.1f}) {float(fr['vx'][i]):.1f} m/s"
                                       for i in range(int(fr['n']))), flush=True)
        shot(tag)
    # -- 8. 혼자 조건에서 '장애물 놓기' → 빈 배치로 다시 시작 → 편집 → 세 개 놓고 달리기
    w._on_stop(); wait(lambda: w.state not in (STATE_RUNNING, STATE_PAUSED), 30, "정지")
    w._on_condition_picked("solo"); step(0.5)
    w._on_edit_layout()
    wait(lambda: w._editing, 240, "빈 배치 편집 열림")
    step(1.5)
    ed = w.layout_ed
    print("8. 혼자 → 장애물 놓기: 조각", len(ed.pieces), "| 조건", w.pick_cond.current(), flush=True)
    fr = c.buffer.latest
    x, y, yaw = float(fr["x"][0]), float(fr["y"][0]), float(fr["yaw"][0])
    w.layout_panel.btn_add.click()
    for d, side in ((3.0, 0.25), (5.0, -0.25), (7.0, 0.0)):
        bx = x + d * math.cos(yaw) - side * math.sin(yaw); by = y + d * math.sin(yaw) + side * math.cos(yaw)
        px, py = vp.screen_point(bx, by, 0.0)
        mouse("press", px, py); mouse("release", px, py)
    w._edit_key("escape")
    step(0.5)
    shot("9solo_place")
    w._on_edit_run()
    wait(lambda: len(w._layout_now.get("layout") or []) == 3, 60, "세 개 적용")
    wait(lambda: w.state == STATE_RUNNING, 20, "재개")
    step(6)
    print("   세 개로 주행: 조각", len(w._layout_now.get("layout") or []), "| 상태", w.state, flush=True)
    shot("9solo_run")
    # -- 9. 칸보다 많이 놓기: 중심선을 따라 16개 더 → '이 배치로 달리기'가 칸을 늘려 세션을 다시 시작
    gen_before = c._active_gen
    w._on_edit_layout()
    wait(lambda: w._editing, 30, "편집 열림")
    step(1.0)
    ed = w.layout_ed
    cl = vp.geometry_data.centerline
    w.layout_panel.btn_add.click()
    for k in range(16):
        q = cl[int((k + 0.5) * len(cl) / 16) % len(cl)]
        px, py = vp.screen_point(float(q[0]), float(q[1]), 0.0)
        mouse("press", px, py); mouse("release", px, py)
    w._edit_key("escape")
    step(0.5)
    print(f"9. 칸 넘기기: 조각 {len(ed.pieces)}개, 칸 {ed.slots}, 안내: {w.layout_panel.note.text()[:50]}", flush=True)
    shot("10many_edit")
    w._on_edit_run()
    wait(lambda: c._active_gen != gen_before and w.state == STATE_RUNNING, 240, "다시 시작")
    step(6)
    print(f"   다시 시작: 세션 {gen_before} → {c._active_gen}, 조각 {len(w._layout_now.get('layout') or [])}, "
          f"칸 {w._layout_now.get('prop_slots')}", flush=True)
    shot("10many_run")
    done = []
    c.shutdown_finished.connect(lambda: done.append(1))
    c.begin_shutdown()
    for _ in range(200):
        step(0.1)
        if done:
            break
    w.allow_close()


if __name__ == "__main__":       # the worker is a spawned process: it re-imports this file
    main()
