"""콘솔에서 pick 하나만 고르고(다른 설정은 손대지 않음) 시작해 화면을 찍음: pick이 '한 번 클릭'으로 맞는 세션을 여는지 확인.
인자: 출력 접두어 세션당 화면 시간[s] pick이름[:상대차를 봄] ..."""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from PyQt5 import QtCore
from f1sim.viewer.console.app import create_app
from f1sim.viewer.console.catalog import list_runs
from f1sim.viewer.console.session import SessionController
from f1sim.viewer.console.window import ConsoleWindow


def main():
    out, drive_s, picks = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    app = create_app(["shot"])
    w = ConsoleWindow(); c = SessionController(w)
    w.set_runs(list_runs()); w.set_maps(c._maps); w.resize(1600, 950); w.show()
    QtCore.QTimer.singleShot(0, c.start_worker)

    def step(t):
        end = time.time() + t
        while time.time() < end:
            app.processEvents(); time.sleep(0.01)

    for _ in range(120):
        step(0.5)
        if getattr(w.maps, "ready", False):
            break
    step(2)
    for n, item in enumerate(picks):
        run, _, opp = item.partition(":")
        w._on_policy_picked(run); step(2)
        cfg = w.current_config()
        print(run, "| 지도", getattr(w, "_selected_map", "?"), "| 내 차", cfg.ego_kind, "| 차 수", cfg.cars_per_race, "| 상대",
              getattr(cfg, "opponent_slots", "?"), "| 조건", w.pick_cond.current() if hasattr(w.pick_cond, "current") else "?", flush=True)
        if run.startswith("="):
            continue
        w._on_start(); step(10)
        if opp and w.combo_focus.count() > 1:
            w.combo_focus.setCurrentIndex(1); w._on_focus_combo(1)
        for k in range(2):
            step(drive_s / 2)
            w.grab().save(out + "_%d%s_%d.png" % (n, run, k))
        print(run, "상태:", getattr(w, "state", "?"), flush=True)
        w._on_stop(); step(6)
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
