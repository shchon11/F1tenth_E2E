"""전송만의 지연 측정용 파이썬 수신기(화면 프레임에 묶이지 않음): 받은 즉시 해독하고 (받은 시각 - 서버 ts)를 잰다.
Godot 앱은 _process(화면 주기)마다 소켓을 훑으므로 앱 쪽 지연에는 0~1프레임 대기가 섞인다. 이 수신기는 그것을 뺀 값을 준다.
python -m f1sim.server.probe --transport tcp --port 47810 --seconds 10
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import socket
import struct
import sys
import time

import numpy as np

from . import SCHEMA_VERSION, wire


def run(transport: str, host: str, port: int, seconds: float) -> dict:
    lat, dec, rtt = [], [], []
    t_end = time.time() + seconds
    hello = wire.packb({"k": "hello", "v": SCHEMA_VERSION, "client": "probe"})

    def on(payload: bytes):
        a = time.time()
        m = wire.unpackb(payload)
        dec.append((time.time() - a) * 1e3)
        if m.get("k") == "state":
            lat.append((a - m["ts"]) * 1e3)
        elif m.get("k") == "pong":
            rtt.append((a - m["t"]) * 1e3)

    def ping():
        return wire.packb({"k": "ping", "v": SCHEMA_VERSION, "t": time.time()})

    t_ping = 0.0
    if transport == "tcp":
        s = socket.create_connection((host, port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(0.2)
        sp = wire.FrameSplitter()
        s.sendall(wire.frame(hello))
        while time.time() < t_end:
            if time.time() - t_ping > 0.25:
                t_ping = time.time(); s.sendall(wire.frame(ping()))
            try:
                data = s.recv(1 << 20)
            except socket.timeout:
                continue
            for p in sp.feed(data):
                on(p)
        s.close()
    elif transport == "ws":
        from websockets.sync.client import connect
        with connect(f"ws://{host}:{port}", compression=None, max_size=2 ** 26) as ws:
            ws.send(hello)
            while time.time() < t_end:
                if time.time() - t_ping > 0.25:
                    t_ping = time.time(); ws.send(ping())
                try:
                    p = ws.recv(timeout=0.2)
                except TimeoutError:
                    continue
                on(p)
    elif transport == "udp":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.sendto(hello, (host, port))
        while time.time() < t_end:
            if time.time() - t_ping > 0.25:
                t_ping = time.time(); s.sendto(ping(), (host, port))
            try:
                p, _ = s.recvfrom(1 << 16)
            except socket.timeout:
                continue
            on(p)
    elif transport == "shm":
        path = f"/dev/shm/tenth_{port}"
        while not os.path.exists(path) and time.time() < t_end:
            time.sleep(0.01)
        fd = os.open(path, os.O_RDONLY)
        mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        last = -1
        while time.time() < t_end:
            s1 = struct.unpack_from("<Q", mm, 8)[0]
            if s1 != last and not (s1 & 1):
                n = struct.unpack_from("<I", mm, 16)[0]
                p = bytes(mm[24:24 + n])
                if struct.unpack_from("<Q", mm, 8)[0] == s1:
                    last = s1
                    on(p)
            time.sleep(0.0002)     # 0.2 ms 간격 폴링(바쁜 대기 대신)
        mm.close(); os.close(fd)
    q = lambda a, x: round(float(np.percentile(a, x)), 3) if a else None
    return {"transport": transport, "frames": len(lat), "lat_ms_p50": q(lat, 50), "lat_ms_p95": q(lat, 95),
            "lat_ms_p99": q(lat, 99), "decode_ms_p50": q(dec, 50), "rtt_ms_p50": q(rtt, 50), "rtt_ms_p95": q(rtt, 95)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m f1sim.server.probe")
    ap.add_argument("--transport", default="tcp", choices=("tcp", "ws", "udp", "shm"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47810)
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args(argv)
    print("TENTH_PROBE " + json.dumps(run(a.transport, a.host, a.port, a.seconds)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
