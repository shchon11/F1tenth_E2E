"""실제 콘솔(창 + SessionController + 워커 프로세스)을 화면 밖(offscreen)에서 띄워 찍는다 (11-02): 대기, 첫 추천 정책 선택, 주행 중.
인자: 출력 접두어 [주행 초]"""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from PyQt5 import QtCore, QtWidgets
from f1sim.viewer.console.app import create_app
from f1sim.viewer.console.catalog import list_runs
from f1sim.viewer.console.session import SessionController
from f1sim.viewer.console.window import ConsoleWindow



def main():
    out = sys.argv[1]; drive_s = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    cond = sys.argv[3] if len(sys.argv) > 3 else ""          # 주행 조건 key (solo / props / race / race_props)
    app = create_app(["shot"])
    w = ConsoleWindow(); c = SessionController(w)
    w.set_runs(list_runs()); w.set_maps(c._maps); w.resize(1600, 950); w.show()
    QtCore.QTimer.singleShot(0, c.start_worker)


    def step(t):
        end = time.time() + t
        while time.time() < end:
            app.processEvents(); time.sleep(0.01)


    for _ in range(120):                       # wait for the worker's map catalogue
        step(0.5)
        if getattr(w.maps, "ready", False):
            break
    step(2)
    w.grab().save(out + "_1idle.png")
    first = next((r for r in w.runs if r.pinned), None)
    if first is not None:
        w.run_list.select(first.name); w._on_run_selected(first.name)
        step(2)
    w.grab().save(out + "_2picked.png")
    w.pick_policy.open_popup(); step(1.0)
    w.pick_policy.popup.grab().save(out + "_2popup_policy.png"); w.pick_policy.popup.hide(); step(0.3)
    w.pick_cond.open_popup(); step(0.6)
    w.pick_cond.popup.grab().save(out + "_2popup_cond.png"); w.pick_cond.popup.hide(); step(0.3)
    w.pick_map.open_popup(); step(0.6)
    w.pick_map.popup.grab().save(out + "_2popup_map.png"); w.pick_map.popup.hide(); step(0.3)
    if cond:
        w._on_condition_picked(cond); step(1)
    w._on_start()
    step(drive_s)
    w.grab().save(out + "_3running.png")
    print("찍음:", out, "| 상태", w.state, "| 첫 추천", first.name if first else None, flush=True)
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
