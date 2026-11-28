"""전송 비교용 합성 서버: 차 N대(기본 2) × LiDAR 1081점 × 40 Hz를 실제 상태 프레임과 같은 모양으로 보낸다.

물리를 돌리지 않으므로(torch import 없음) 전송·직렬화·앱 쪽 비용만 잰다.
python -m f1sim.server.bench --transport tcp --port 47810 --cars 2 --rate 40 --duration 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

from . import protocol as P
from . import transports, wire


def synth_frame(seq: int, t: float, n_cars: int, n_beams: int, rng: np.random.Generator) -> dict:
    cars, scans = [], []
    for i in range(n_cars):
        a = 0.35 * t + i * 0.4
        x, y = 6.0 * math.cos(a), 3.0 * math.sin(a)
        yaw = math.atan2(3.0 * math.cos(a), -6.0 * math.sin(a))
        cars.append({"id": i, "x": x, "y": y, "yaw": yaw, "v": 4.0 + math.sin(t), "vy": 0.0, "wz": 0.3,
                     "steer": 0.1 * math.sin(2 * t), "roll": 0.0, "pitch": 0.0, "cmd_steer": 0.0, "cmd_v": 4.0,
                     "lap": int(t // 9), "s": (t % 9) * 5.0, "lap_t": t % 9, "last_lap": 9.0, "best_lap": 8.9,
                     "clean_streak": 1, "best_streak": 1, "contacts": 0, "contact": False, "driver": "bench"})
        ang = np.linspace(-0.75 * math.pi, 0.75 * math.pi, n_beams, dtype=np.float32)
        r = (2.0 + 1.2 * np.abs(np.sin(ang * 2.0 + t)) + rng.normal(0, 0.01, n_beams)).astype(np.float32)
        scans.append({"car": i, "pose": [x, y, yaw], "ranges": r, "hit": np.ones(n_beams, np.uint8)})
    return P.state_frame(gen=1, seq=seq, t=t, rt=1.0, status="running", cars=cars, scans=scans)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m f1sim.server.bench")
    ap.add_argument("--transport", default="tcp", choices=("tcp", "ws", "udp", "shm"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47810)
    ap.add_argument("--cars", type=int, default=2)
    ap.add_argument("--beams", type=int, default=1081)
    ap.add_argument("--rate", type=float, default=40.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    tr = transports.make(a.transport, a.host, a.port)
    tr.start()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    rng = np.random.default_rng(0)
    hello = wire.packb(P.hello("tenth-bench", a.rate, a.transport, cars=a.cars, pid=os.getpid()))
    print("TENTH_SERVER_READY " + json.dumps({"transport": a.transport, "port": a.port, "pid": os.getpid()}), flush=True)
    dt = 1.0 / a.rate
    t0 = time.perf_counter()
    t_next = t0
    seq = 0
    enc_ms = []
    sizes = []
    pings = 0
    started = None
    while not stop["flag"]:
        now = time.perf_counter()
        if started is not None and now - started >= a.duration:
            break
        for cid in tr.joined():
            tr.send_reliable(hello, cid)
        for cid, m in tr.poll():
            pass                     # ping은 전송 계층이 바로 답한다
        if tr.clients and started is None:
            started = now
        seq += 1
        e0 = time.perf_counter()
        b = wire.packb(synth_frame(seq, now - t0, a.cars, a.beams, rng))
        enc_ms.append((time.perf_counter() - e0) * 1e3)
        sizes.append(len(b))
        tr.send_state(b)
        t_next += dt
        sl = t_next - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
        else:
            t_next = time.perf_counter()
    res = {"transport": a.transport, "frames": seq, "frame_bytes": int(np.median(sizes)) if sizes else 0,
           "encode_ms_med": round(float(np.median(enc_ms)), 3) if enc_ms else None,
           "sent": tr.frames_sent, "dropped": tr.dropped(), "pings": pings}
    print("TENTH_BENCH " + json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)
    tr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
