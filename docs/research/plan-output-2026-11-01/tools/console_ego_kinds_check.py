"""콘솔과 같은 세션(plan policy 선택 + 지도 Korea 2025)에서 내 차를 다른 주행기로 바꿨을 때 차가 실제로 움직이는지.
주행기마다 세션을 만들어 걸음 수만큼 돌리고, 차에 들어간 속도 명령과 달린 거리·최고 속도를 찍음. 인자: 걸음 수 종류 [종류 ...]"""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")


class Conn:
    def send(self, *a, **k): pass
    def poll(self, *a, **k): return False
    def recv(self): raise EOFError
    def close(self): pass


def main():
    import torch
    from f1sim.viewer.console.app import create_app
    from f1sim.viewer.console.catalog import list_runs
    from f1sim.viewer.console.session import SessionController
    from f1sim.viewer.console.window import ConsoleWindow
    from f1sim.viewer import sim_worker as SW
    n_step = int(sys.argv[1]); kinds = sys.argv[2:]
    app = create_app(["chk"])
    w = ConsoleWindow(); c = SessionController(w)
    w.set_runs(list_runs()); w.set_maps(c._maps)

    def pump(t):
        end = time.time() + t
        while time.time() < end:
            app.processEvents(); time.sleep(0.01)

    pump(1)
    w._on_policy_picked("watch_plan25r"); pump(0.5)
    w._on_map_picked("real/iccas25"); pump(0.5)
    for kind in kinds:
        w.ego_panel.set_config({"ego_kind": kind}); pump(0.3)
        w._on_condition_picked("solo"); pump(0.5)
        cfg = w.current_config()
        wk = SW.SimWorker(Conn(), Conn()); wk.gen = 0
        try:
            ses = wk.build_session(cfg, 0)
        except Exception as e:                                        # a kind that needs a weights file, etc.
            print(f"{kind}: 세션을 만들지 못함 ({type(e).__name__}: {str(e)[:120]})", flush=True); continue
        env = ses["env"]
        rows = env.external_slot_rows.get(0)
        if rows is None:
            rows = env.learner_ids
        p0 = env.sim.state[rows][0, :2].clone(); dist = 0.0; vmax = 0.0; cmd_v = []
        for k in range(n_step):
            wk._step_once(ses)
            p = env.sim.state[rows][0, :2]
            d = float((p - p0).norm())
            if d < 1.5:
                dist += d
            p0 = p.clone()
            vmax = max(vmax, float(env.sim.state[rows][0, 3]))
            cmd_v.append(float(env.last_cmd[rows][0, 1]))
        print(f"{kind}: 운전자 {type(env.external_drivers.get(0)).__name__} | {n_step / 40:.0f} s 동안 달린 거리 {dist:.1f} m · 최고 속도 {vmax:.2f} m/s"
              f" · 차에 들어간 속도 명령 평균 {sum(cmd_v) / len(cmd_v):.2f} m/s (0인 걸음 {100 * sum(v == 0 for v in cmd_v) / len(cmd_v):.0f} %)"
              f" | 내 차 줄이 외부 명령에 묶임: {bool(env.ext_mask[rows].any())}", flush=True)
    print("끝", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
