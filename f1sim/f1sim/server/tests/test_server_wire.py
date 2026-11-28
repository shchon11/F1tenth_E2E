"""시뮬 서버 다리 시험: 직렬화 규격, 틀 자르기, 프로토콜 검사, TCP 전송 왕복, 세션 한 바퀴 일부.
실행: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q f1sim/f1sim/server/tests
"""
import socket
import struct
import time

import numpy as np
import pytest

from f1sim.server import SCHEMA_VERSION, protocol as P, transports, wire
from f1sim.server.bench import synth_frame


@pytest.mark.parametrize("v", [None, True, False, 0, 127, 128, 255, 65536, 2 ** 40, -1, -32, -33, -200, -40000,
                               -2 ** 40, 0.5, -3.25e-9, "", "가나다", "x" * 300, b"\x00\x01", [1, [2, 3]],
                               {"a": 1, "b": {"c": [None]}}, list(range(40))])
def test_roundtrip(v):
    assert wire.unpackb(wire.packb(v)) == v


def test_ndarray_is_little_endian_raw():
    a = np.arange(5, dtype=np.float32) * 1.5
    d = wire.unpackb(wire.packb({"r": a}))["r"]
    assert d["$nd"] == "f4" and d["shape"] == [5]
    assert d["data"] == a.astype("<f4").tobytes()
    np.testing.assert_array_equal(wire.from_nd(d), a)


def test_known_bytes():
    # 표준 MessagePack과 같은 바이트(다른 언어 구현과의 계약)
    assert wire.packb({"k": "cmd"}) == b"\x81\xa1k\xa3cmd"
    assert wire.packb(1.0) == b"\xcb" + struct.pack(">d", 1.0)
    assert wire.packb(-1) == b"\xff"


def test_frame_splitter_partial():
    a, b = wire.packb({"x": 1}), wire.packb({"y": "z" * 1000})
    stream = wire.frame(a) + wire.frame(b)
    sp = wire.FrameSplitter()
    got = []
    for i in range(0, len(stream), 7):
        got += sp.feed(stream[i:i + 7])
    assert got == [a, b]


def test_check_client():
    assert P.check_client({"k": "cmd", "v": SCHEMA_VERSION, "op": "start"}) is None
    assert "버전" in P.check_client({"k": "cmd", "v": 99, "op": "start"})
    assert "명령" in P.check_client({"k": "cmd", "v": SCHEMA_VERSION, "op": "fly"})


def test_tcp_transport_roundtrip():
    tr = transports.TcpTransport("127.0.0.1", 0)
    tr.port = 0
    tr.srv = socket.create_server(("127.0.0.1", 0))
    tr.srv.settimeout(0.5)
    port = tr.srv.getsockname()[1]
    import threading
    threading.Thread(target=tr._accept, daemon=True).start()
    c = socket.create_connection(("127.0.0.1", port))
    c.settimeout(3)
    t_end = time.time() + 3
    while not tr.clients and time.time() < t_end:
        time.sleep(0.01)
    assert tr.joined()
    for i in range(5):                     # 최신 칸: 앞의 것은 버려질 수 있다
        tr.send_state(wire.packb(synth_frame(i, 0.1 * i, 2, 1081, np.random.default_rng(i))))
    tr.send_reliable(wire.packb(P.pong(1.0)))
    c.sendall(wire.frame(wire.packb({"k": "ping", "v": SCHEMA_VERSION, "t": 2.0})))
    sp = wire.FrameSplitter()
    msgs = []
    while time.time() < t_end and not any(m["k"] == "pong" for m in msgs):
        msgs += [wire.unpackb(p) for p in sp.feed(c.recv(65536))]
    kinds = [m["k"] for m in msgs]
    assert "pong" in kinds and "state" in kinds
    st = [m for m in msgs if m["k"] == "state"][-1]
    assert len(st["cars"]) == 2 and st["scans"][0]["ranges"]["$nd"] == "f4"
    c.sendall(wire.frame(wire.packb({"k": "cmd", "v": SCHEMA_VERSION, "op": "start", "seq": 1})))
    t_end = time.time() + 2
    got = []
    while time.time() < t_end and not got:
        got = tr.poll()
        time.sleep(0.01)
    assert got and got[0][1]["k"] == "cmd"     # ping은 전송 계층이 답하고 수신함에 넣지 않는다
    c.close()
    tr.close()


def test_session_laps_quick():
    """기존 코어 그대로: Korea 2026, FTG, CPU. 4초 시뮬 동안 접촉 없이 앞으로 간다."""
    from f1sim.server.session import SimSession
    s = SimSession("real/korea26", n_cars=1)
    s0 = float(s.last.s[0])
    for _ in range(160):
        s.step()
    car = s.cars()[0]
    assert car["v"] > 1.0
    assert car["contacts"] == 0
    assert car["lap_t"] is not None            # 출발선 1 m 뒤에서 시작 -> 첫 통과로 계시 시작
    sc = s.scans()[0]
    assert sc["ranges"].dtype == np.float32 and sc["ranges"].shape == (1081,)
    tm = s.track_message()
    assert tm["meshes"]["ducts"]["pos"].dtype == np.float32
    assert 40.0 < tm["length_m"] < 70.0
    del s0
