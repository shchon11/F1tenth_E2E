"""The live tap's named reward parts (11-24): the parts sum to the reward the training used, a late term (plan_rl
--plan-car-cost, priced after the rollout) lands on its own decisions, the sums cover settled decisions only, the prediction
head's points go to the world. Env-free: a minimal stand-in for the env."""
import json
import math
from types import SimpleNamespace

import numpy as np
import torch

from f1sim.learn.live_tap import LiveTap


def fake_env():
    sim = SimpleNamespace(state=torch.zeros(2, 8), absent=torch.zeros(2, dtype=torch.bool), absent_any=False,
                          contact_counts=torch.zeros(2, 6), tid=torch.zeros(2, dtype=torch.long),
                          other_idx=torch.tensor([[1], [0]]), lidar=SimpleNamespace(angles=torch.linspace(-2.3, 2.3, 16)),
                          P={"mount_x": torch.full((2,), 0.12)}, track=None)
    return SimpleNamespace(sim=sim, learner_ids=torch.tensor([0]), M=2, procedural=None,
                           cfg=SimpleNamespace(vehicle=SimpleNamespace(length=0.58, width=0.31)),
                           last_result=SimpleNamespace(scan=torch.full((2, 16), 3.0)))


def rec(tap, k, lap=0.0, car=0.0, pred=None, pose=None):
    z = torch.zeros(10, 2)
    rew = lap + car
    tap.record(1, k, z, z, torch.zeros(10), torch.zeros(()), torch.tensor(rew), torch.tensor(lap), torch.tensor(car),
               torch.zeros(()), torch.zeros(()), torch.zeros(()), torch.zeros(()),
               parts=torch.tensor([lap, 0.0, 0.0, 0.0, car]), pred=pred, pred_pose=pose)
    return json.load(open(tap.path))["now"]


def test_parts_settle_and_sum(tmp_path):
    tap = LiveTap(fake_env(), str(tmp_path), row=0, name="tap_t.json")
    tap.set_parts(["바퀴 완주", "전진 shaping", "벽 접촉", "상자 접촉", "차 접촉(내 잘못)"], late="차 겹침 plan")
    now = None
    for k in range(5):                                       # a rollout of 5 decisions: a lap at 2, a car contact at 3
        now = rec(tap, k, lap=1.0 if k == 2 else 0.0, car=-0.2 if k == 3 else 0.0)
    assert now["parts_lag_s"] is None and now["parts"][-1]["n"] == "합" and now["parts"][-1]["ep"] == 0.0   # nothing settled
    tap.settle([0.0, -0.05, 0.0, -0.03, 0.0])                # the rollout's plan-car cost, per decision
    now = rec(tap, 5)                                        # the next rollout's first decision
    names = [p["n"] for p in now["parts"]]
    assert names == ["바퀴 완주", "전진 shaping", "벽 접촉", "상자 접촉", "차 접촉(내 잘못)", "차 겹침 plan", "합"]
    ep = {p["n"]: p["ep"] for p in now["parts"]}
    assert math.isclose(ep["바퀴 완주"], 1.0) and math.isclose(ep["차 접촉(내 잘못)"], -0.2)
    assert math.isclose(ep["차 겹침 plan"], -0.08) and math.isclose(ep["합"], 1.0 - 0.2 - 0.08)
    assert now["parts_lag_s"] == 0.1 and abs(now["parts_resid"]) < 1e-9
    assert now["car_dims"] == [0.58, 0.31] and now["opp_dims"] == [0.58, 0.31]
    assert now["pred"] is None and now["opp_plan"] is None


def test_pred_in_world(tmp_path):
    tap = LiveTap(fake_env(), str(tmp_path), row=0, name="tap_p.json")
    q = torch.zeros(21)
    q[0], q[1] = 0.2, 0.0                                    # point 1: 1 m ahead (x PRED_SCALE 5) in the car's frame
    now = rec(tap, 0, pred=q, pose=torch.tensor([2.0, 3.0, math.pi / 2]))
    assert np.allclose(now["pred"]["pts"][0], [2.0, 4.0], atol=1e-3) and abs(now["pred"]["p"] - 0.5) < 1e-6
