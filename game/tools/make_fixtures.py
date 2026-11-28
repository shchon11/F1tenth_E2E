#!/usr/bin/env python3
"""GDScript 시험용 고정 자료: 파이썬 서버의 직렬화(f1sim.server.wire)로 만든 실제 상태 프레임 + 기대값 JSON.
python 쪽 규격이 바뀌면 이 스크립트를 다시 돌리고 game/tests/test_msgpack.gd가 통과하는지 본다.
사용: .venv/bin/python game/tools/make_fixtures.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "f1sim"))

from f1sim.server import wire  # noqa: E402
from f1sim.server.bench import synth_frame  # noqa: E402

f = synth_frame(4242, 3.25, 2, 1081, np.random.default_rng(7))
f["cars"][0]["last_lap"] = None
b = wire.packb(f)
out = os.path.join(os.path.dirname(HERE), "tests", "fixtures")
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "state_frame.msgpack"), "wb").write(b)
r = f["scans"][0]["ranges"]
json.dump({"v": f["v"], "seq": f["seq"], "n_cars": len(f["cars"]), "car1_x": f["cars"][1]["x"],
           "n_beams": int(r.size), "r0_first": float(r[0]), "r0_last": float(r[-1]),
           "car0_last_lap": None, "bytes": len(b)},
          open(os.path.join(out, "state_frame.json"), "w"), indent=1)
print(f"state_frame.msgpack {len(b)} bytes")
