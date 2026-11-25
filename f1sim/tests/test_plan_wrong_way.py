"""plan_rl --wrong-way-cost (`learn.plan_reward`): the distance field's downhill direction (across the finish line, beside a
wall), the cost's geometry (facing backwards and driving pays per metre and, past 120 deg, per decision; reversing with the
nose along the track is free; standing, waivers and restarts), the reward's line crossings, the car-push waiver through
`control_step`, the flag's round trip through the arguments a checkpoint keeps, and the per-update numbers over two processes
(--dist-world: one all_reduce on every process). Env-free (CPU)."""
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

from f1sim.track import TrackTensors
from f1sim.learn import plan_reward as R

LAP = 40.0
RES = 0.05
X0, X1, Y0, Y1 = -5.0, 5.0, -1.0, 1.0           # a straight strip, driven towards +x; the finish line at x = 0
WALL_Y = 0.82                                    # |y| > this is wall (cells at +-0.85 and out)
K = 0.2
WORLD = os.path.join(os.path.dirname(__file__), "..", "f1sim", "learn", "configs", "plan_world_korea26scan_qual.json")


def strip_goal():
    """a GoalField on the strip: distance left = (LAP - x) mod LAP (x just past the line: ~LAP; just before it: ~0), NaN in
    the walls -- read through the real `TrackTensors.sample_edt` (nearest cell)"""
    W, H = int(round((X1 - X0) / RES)) + 1, int(round((Y1 - Y0) / RES)) + 1
    jj, ii = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    x, y = X0 + jj * RES, Y0 + ii * RES
    D = torch.remainder(LAP - x, LAP)
    D = torch.where(y.abs() > WALL_Y, torch.full_like(D, float("nan")), D)
    tr = SimpleNamespace(edt=torch.zeros(H * W), t_res=torch.tensor([RES]), t_origin=torch.tensor([[X0, Y0]]),
                         t_H=torch.tensor([H]), t_W=torch.tensor([W]), t_off=torch.tensor([0]))
    tr.sample_edt = lambda xy, tid, field=None, with_inside=False: TrackTensors.sample_edt(tr, xy, tid, field, with_inside)
    g = R.GoalField.__new__(R.GoalField)
    g.track, g.field, g.lap = tr, D.reshape(-1).float(), torch.tensor([LAP])
    return g


# ---------------------------------------------------------------------------------------------------------- the direction
def test_downhill_points_on_through_the_line_and_beside_a_wall():
    g = strip_goal()
    xy = torch.tensor([[2.0, 0.0], [-2.0, 0.3], [0.1, 0.0], [-0.1, 0.0], [0.0, 0.0], [0.05, -0.2],   # across the line too
                       [1.0, 0.75], [1.0, -0.78],                                                    # a wall within h: one-sided
                       [1.0, 0.95]])                                                                 # in the wall
    d, known = g.downhill(xy, torch.zeros(len(xy), dtype=torch.long))
    assert known[:-1].all() and not bool(known[-1])
    assert torch.allclose(d[:-1], torch.tensor([[1.0, 0.0]]).expand(len(xy) - 1, 2), atol=1e-5)
    assert torch.equal(d[-1], torch.zeros(2))


# ------------------------------------------------------------------------------------------------------------- the cost
def cost(back, ang_deg, vx, known=True, ok=True):
    t = lambda v: torch.tensor([float(v)])
    return R.wrong_way_cost(t(back), torch.cos(torch.tensor([math.radians(ang_deg)])), torch.tensor([bool(known)]), t(vx), K,
                            torch.tensor([bool(ok)]))


@pytest.mark.parametrize("back, ang, vx, known, ok, want_m, want_head", [
    (-0.2, 0, 2.0, True, True, 0.0, False),       # driving on: nothing
    (0.2, 180, 2.0, True, True, 0.2, True),        # turned round and driving: the metres and the heading term
    (0.15, 130, 1.5, True, True, 0.15, True),
    (0.05, 100, 1.0, True, True, 0.05, False),     # 90-120 deg (a wide hairpin line points up to ~110-136 deg off): metres only
    (-0.03, 100, 2.5, True, True, 0.0, False),     # ... and nothing while the distance left still drops
    (0.15, 0, -1.5, True, True, 0.0, False),       # reversing with the nose along the track: recovery, free
    (-0.15, 180, -1.5, True, True, 0.0, False),    # facing backwards and reversing (moving on along the track): free
    (0.0, 180, 0.1, True, True, 0.0, False),       # facing backwards and standing: free
    (0.02, 180, 0.19, True, True, 0.0, False),     # rolling below 0.2 m/s: free
    (0.2, 180, 2.0, True, False, 0.0, False),      # waived (pushed by a car) / restarted
    (0.2, 0, 2.0, False, True, 0.2, False),        # no direction (a wall within h): the motion decides, metres only
    (-0.2, 180, 2.0, False, True, 0.0, False),
])
def test_cost_geometry(back, ang, vx, known, ok, want_m, want_head):
    w = cost(back, ang, vx, known, ok)
    assert float(w["metres"]) == pytest.approx(want_m, abs=1e-6)
    assert bool(w["head"]) == want_head
    assert float(w["cost"]) == pytest.approx(K * (want_m + R.WW_HEAD_M * want_head), abs=1e-6)


def test_size_beats_the_lap_value_of_forward_travel():
    # 1 m backwards costs more than 5x what 1 m forwards earns (a lap ~39-42 m = +1): the user's "말도안되는 벌점"
    assert float(cost(1.0, 100, 1.0)["cost"]) > 5.0 / 39.0
    # one second turned round at 2 m/s: 2 m + ten decisions' heading terms, against ~5 wall contacts at 2 m/s
    one_s = sum(float(cost(0.2, 170, 2.0)["cost"]) for _ in range(10))
    wall = 0.82 * (R.CONTACT_S + R.CONTACT_S_PER_MPS * 2.0) / R.T_REF
    assert one_s > 4 * wall


# ----------------------------------------------------------------------------------------- the reward, through PlanReward
def strip_reward(monkeypatch, n):
    g = strip_goal()
    monkeypatch.setattr(R, "GoalField", lambda track, device: g)
    state = torch.zeros(n, 8)
    sim = SimpleNamespace(state=state, tid=torch.zeros(n, dtype=torch.long), track=None, control_dt=0.025,
                          corners=torch.tensor([[0.29, 0.155], [0.29, -0.155], [-0.29, -0.155], [-0.29, 0.155]]))
    env = SimpleNamespace(sim=sim, learner_ids=torch.arange(n), last_result=None, procedural=None)
    pr = R.PlanReward(env, 0.995, "cpu")
    pr.ww_k = K
    return pr, state


def place(state, x, yaw, vx):
    state[:, 0] = torch.tensor(x); state[:, 2] = torch.tensor(yaw); state[:, 3] = torch.tensor(vx)


def test_decision_charges_turned_round_driving_only(monkeypatch):
    pr, st = strip_reward(monkeypatch, 4)
    #  0 drives on, 1 turned round and drives (backwards along the track), 2 reverses nose-on (recovery), 3 turned round, reversing
    place(st, [1.0, 1.0, 1.0, 1.0], [0.0, math.pi, 0.0, math.pi], [2.0, 2.0, -2.0, -2.0])
    pr.begin(torch.arange(4))
    place(st, [1.2, 0.8, 0.8, 1.2], [0.0, math.pi, 0.0, math.pi], [2.0, 2.0, -2.0, -2.0])
    out = pr.decision(torch.zeros(4, 10, 2), restarted=torch.zeros(4, dtype=torch.bool))
    want = torch.tensor([0.0, K * (0.2 + R.WW_HEAD_M), 0.0, 0.0])
    assert torch.allclose(-out["wrong_way"], want, atol=1e-5)
    assert torch.allclose(out["ww_m"], torch.tensor([0.0, 0.2, 0.0, 0.0]), atol=1e-5)
    assert out["ww_hit"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert torch.allclose(out["reward"], out["lap"] + out["shaping"] + out["contact"] + out["plan_hit"] + out["wrong_way"])


def test_line_crossings_count_only_the_metres(monkeypatch):
    pr, st = strip_reward(monkeypatch, 3)
    # 0 crosses forwards, 1 turned round crosses backwards, 2 restarts (put elsewhere, facing backwards)
    place(st, [-0.1, 0.1, 2.0], [0.0, math.pi, math.pi], [2.0, 2.0, 2.0])
    pr.begin(torch.arange(3))
    place(st, [0.1, -0.1, -3.0], [0.0, math.pi, math.pi], [2.0, 2.0, 2.0])
    rs = torch.tensor([False, False, True])
    out = pr.decision(torch.zeros(3, 10, 2), restarted=rs)
    assert out["lap"].tolist() == [1.0, -1.0, 0.0]
    assert torch.allclose(out["progress"], torch.tensor([0.2, -0.2, 0.0]), atol=1e-5)
    assert torch.allclose(-out["wrong_way"], torch.tensor([0.0, K * (0.2 + R.WW_HEAD_M), 0.0]), atol=1e-5)


def test_push_waiver_lasts_one_second(monkeypatch):
    pr, st = strip_reward(monkeypatch, 3)
    place(st, [1.0, 1.0, 1.0], [math.pi] * 3, [2.0] * 3)
    pr.begin(torch.arange(3))
    # a car contact at this control step: rows 0 and 1 not responsible (pushed), row 2 responsible (rammed it)
    on = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    pr.env.last_result = SimpleNamespace(contact_onsets=on, contact_impacts=torch.zeros(3, 3))
    pr.at_fault = lambda: torch.tensor([False, False, True])
    x = 1.0
    charged = []
    for dec in range(14):                          # 1.4 s, the contact in the first control step
        for j in range(4):
            pr.control_step(torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))
            pr.env.last_result = SimpleNamespace(contact_onsets=torch.zeros(3, 3), contact_impacts=torch.zeros(3, 3))
        x -= 0.2
        place(st, [x] * 3, [math.pi] * 3, [2.0] * 3)
        out = pr.decision(torch.zeros(3, 10, 2), restarted=torch.zeros(3, dtype=torch.bool))
        charged.append((out["ww_hit"] > 0).tolist())
    # the push at the first control step's end: decision k ends 0.075 + 0.1 k s after it -> waived for k = 0..9, charged on
    assert [c[0] for c in charged] == [False] * 10 + [True] * 4
    assert [c[1] for c in charged] == [False] * 10 + [True] * 4
    assert all(c[2] for c in charged)              # our own ram: no waiver
    # a restart clears the waiver
    pr.ww_since[:] = 0.0
    pr.begin(torch.tensor([0]))
    assert float(pr.ww_since[0]) >= 1e8 and float(pr.ww_since[1]) == 0.0


def test_off_by_default_and_costs_nothing(monkeypatch):
    pr, st = strip_reward(monkeypatch, 1)
    pr.ww_k = 0.0
    place(st, [1.0], [math.pi], [2.0])
    pr.begin(torch.arange(1))
    place(st, [0.8], [math.pi], [2.0])
    out = pr.decision(torch.zeros(1, 10, 2), restarted=torch.zeros(1, dtype=torch.bool))
    assert float(out["wrong_way"]) == 0.0 and "ww_hit" not in out


# ------------------------------------------------------------------------------------------ the flag in the checkpoint
def check_args(capsys, *extra):
    from f1sim.learn import plan_rl
    with pytest.raises(SystemExit) as e:
        plan_rl.main(["--name", "t", "--world", WORLD, "--check-args", *extra])
    return e.value.code, capsys.readouterr().out


def test_flag_round_trip_through_the_checkpoint_args(capsys, tmp_path):
    code, out = check_args(capsys, "--wrong-way-cost", "0.2")
    assert code == 0
    args = json.loads(out)["args"]
    assert args["wrong_way_cost"] == 0.2
    torch.save({"args": args}, tmp_path / "ck.pt")                        # what plan_rl saves: vars(args)
    assert torch.load(tmp_path / "ck.pt", weights_only=False)["args"]["wrong_way_cost"] == 0.2
    code, out = check_args(capsys)
    assert code == 0 and json.loads(out)["args"]["wrong_way_cost"] == 0.0
    code, _ = check_args(capsys, "--wrong-way-cost", "-0.1")
    assert "--wrong-way-cost" in str(code)


# --------------------------------------------------------------------------------------------- --dist-world collectives
def _rank(rank, world, init, out_dir):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    try:
        # rank 0: cost 1.0 laps over 10 min, 3 of 1000 decisions, 2 m; rank 1: nothing charged, 600 decisions over 6 min
        loc = [(1.0, 3.0, 2.0, 1000.0, 10.0), (0.0, 0.0, 0.0, 600.0, 6.0)][rank]
        got = R.ww_totals(*loc, dist=dist)
        json.dump(got, open(os.path.join(out_dir, f"r{rank}.json"), "w"))
    finally:
        dist.destroy_process_group()


def test_totals_over_two_processes(tmp_path):
    import torch.multiprocessing as mp
    init = f"file://{tmp_path / 'store'}"
    mp.spawn(_rank, args=(2, init, str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (json.load(open(tmp_path / f"r{k}.json")) for k in (0, 1))
    assert r0 == r1                                                     # every process logs the same numbers
    assert r0["wrong_way_cost_per_min"] == pytest.approx(1.0 / 16.0)
    assert r0["wrong_way_share"] == pytest.approx(3.0 / 1600.0)
    assert r0["wrong_way_m_per_min"] == pytest.approx(2.0 / 16.0)
    alone = R.ww_totals(1.0, 3.0, 2.0, 1000.0, 10.0)                   # one process: its own
    assert alone["wrong_way_cost_per_min"] == pytest.approx(0.1) and alone["wrong_way_share"] == pytest.approx(0.003)
