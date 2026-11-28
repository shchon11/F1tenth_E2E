"""`python -m f1sim.server` : 시뮬 세션 하나를 전송 하나로 내보낸다(스파이크).

예) python -m f1sim.server --track real/korea26 --transport tcp --port 47800 --autostart
    python -m f1sim.server --record out/run.tenthrec      # 보낸 hello·track·state·event를 그대로 기록(리플레이)

준비가 끝나면 표준 출력에 `TENTH_SERVER_READY {json}` 한 줄을 쓴다(실행기가 이 줄을 기다린다).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

from . import protocol as P
from . import transports, wire
from .session import SimSession


class Recorder:
    """리플레이 기록: 서버가 보낸 메시지를 u32 길이 접두 틀 그대로 이어 쓴다. 첫 메시지는 hello."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "wb")

    def write(self, payload: bytes) -> None:
        self.f.write(wire.frame(payload))

    def close(self) -> None:
        self.f.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m f1sim.server", description=__doc__)
    ap.add_argument("--track", default="real/korea26")
    ap.add_argument("--cars", type=int, default=1)
    ap.add_argument("--transport", default="tcp", choices=("tcp", "ws", "udp", "shm"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47800)
    ap.add_argument("--autostart", action="store_true", help="앱의 start 명령 없이 바로 달린다")
    ap.add_argument("--ftg-speed", type=float, default=6.0, help="Follow-the-Gap max_speed(최고 속도는 0.6배)")
    ap.add_argument("--compile", action="store_true", help="물리 루프를 inductor CPU로 컴파일(첫 기동 수십 초)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--duration", type=float, default=0.0, help="이 시간(초, 벽시계) 뒤 스스로 끝남. 0 = 무한")
    ap.add_argument("--record", default="")
    ap.add_argument("--stats-every", type=float, default=5.0)
    a = ap.parse_args(argv)

    t_boot = time.perf_counter()
    sess = SimSession(a.track, n_cars=a.cars, ftg_max_speed=a.ftg_speed, compile_cpu=a.compile, threads=a.threads)
    if a.compile:   # 컴파일을 기동 단계에서 끝낸다(첫 스텝)
        sess.step(); sess.reset()
    t0 = time.perf_counter()
    track_payload = wire.packb(sess.track_message())
    geom_ms = (time.perf_counter() - t0) * 1e3
    tr = transports.make(a.transport, a.host, a.port)
    tr.start()
    rec = Recorder(a.record) if a.record else None
    hello = wire.packb(P.hello("tenth-sim", 1.0 / sess.control_dt, a.transport, track=a.track, cars=a.cars,
                               pid=os.getpid()))
    if rec:
        rec.write(hello); rec.write(track_payload)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    print("TENTH_SERVER_READY " + json.dumps({
        "transport": a.transport, "host": a.host, "port": a.port, "pid": os.getpid(),
        "boot_s": round(time.perf_counter() - t_boot, 2), "track_bytes": len(track_payload),
        "geometry_ms": round(geom_ms, 1)}), flush=True)

    status = "running" if a.autostart else "ready"
    seq = 0
    dt = sess.control_dt
    t_next = time.perf_counter()
    t_end = time.perf_counter() + a.duration if a.duration > 0 else None
    win_wall, win_sim, step_ms = time.perf_counter(), sess.t, []
    rt = 1.0
    t_stats = time.perf_counter()
    while not stop["flag"]:
        now = time.perf_counter()
        if t_end is not None and now >= t_end:
            break
        for cid in tr.joined():
            tr.send_reliable(hello, cid)
            tr.send_reliable(track_payload, cid)
        for cid, m in tr.poll():
            err = P.check_client(m)
            if err:
                tr.send_reliable(wire.packb(P.msg(P.K_ERROR, note=err)), cid)
                continue
            if m["k"] == P.K_CMD:
                op = m["op"]
                if op == "start" or op == "resume":
                    status = "running"
                elif op == "pause":
                    status = "paused"
                elif op == "stop":
                    status = "stopped"
                elif op == "reset":
                    sess.reset()
                    status = "ready"
                tr.send_reliable(wire.packb(P.ack(m.get("seq", 0), op)), cid)
        if status == "running":
            a0 = time.perf_counter()
            sess.step()
            step_ms.append((time.perf_counter() - a0) * 1e3)
            for ev in sess.events:
                p = wire.packb(P.event(ev.pop("type"), gen=sess.gen, **ev))
                tr.send_reliable(p)
                if rec:
                    rec.write(p)
            sess.events.clear()
        if now - win_wall >= 1.0:
            rt = (sess.t - win_sim) / (now - win_wall) if status == "running" else 0.0
            win_wall, win_sim = now, sess.t
        seq += 1
        frame = wire.packb(P.state_frame(gen=sess.gen, seq=seq, t=sess.t, rt=rt, status=status,
                                         cars=sess.cars(), scans=sess.scans([sess.focus])))
        tr.send_state(frame)
        if rec:
            rec.write(frame)
        if a.stats_every > 0 and now - t_stats >= a.stats_every:
            t_stats = now
            sm = sorted(step_ms) if step_ms else [0.0]
            print("TENTH_STATS " + json.dumps({
                "t_sim": round(sess.t, 2), "rt": round(rt, 3), "status": status,
                "step_ms_med": round(sm[len(sm) // 2], 2), "frame_bytes": len(frame),
                "frames_sent": tr.frames_sent, "dropped": tr.dropped(), "clients": len(tr.clients),
                "laps": [c["lap"] for c in sess.cars()], "best": [c["best_lap"] for c in sess.cars()]}), flush=True)
            step_ms.clear()
        # 실시간 맞추기: 늦으면 따라잡지 않고(폭주 방지) 지금부터 다시 센다
        t_next += dt
        sl = t_next - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
        else:
            t_next = time.perf_counter()
    tr.close()
    if rec:
        rec.close()
    print("TENTH_SERVER_BYE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
