"""(11-18) --line3-boxes: every draw is three bank pieces on the policy's own driven line (`set_driven_line` builds the 0.5 m
stretch tables, `_redraw_bank` draws one piece in each third of the lap), weighted toward the line; a track whose line is not
measured yet draws as before. Checked on a stand-in that carries only what those read."""
import types

import torch

from f1sim.procedural_obstacles import ProceduralObstacles

L_M = 12.0
LINE = torch.stack([torch.arange(0, L_M, 0.25), torch.zeros(int(L_M / 0.25))], -1)[None]      # (1, 48, 2) along y = 0


def _po():
    po = object.__new__(ProceduralObstacles)
    po.device = torch.device("cpu")
    po.place = "bank"
    po.gen = torch.Generator().manual_seed(0)
    po.tr = types.SimpleNamespace(length=torch.tensor([L_M]))
    # two layouts of 30 pieces along the lap, 0.2 m apart, at y = 0 / 0.2 / 1.0 in turn (two in three within 0.4 m)
    x = torch.arange(60, dtype=torch.float32) * 0.2
    y = torch.tensor([0.0, 0.2, 1.0]).repeat(20)
    pose = torch.stack([x, y, torch.zeros(60)], -1).view(2, 30, 3)
    po.b_sid = torch.zeros(1, 2, 30, dtype=torch.long)
    po.b_pose = pose[None]
    po.b_n = torch.tensor([2])
    po.C, po.P, po.B = 30, 30, 256
    po.count, po.keep, po.fraction = "fixed", 3, 1.0
    po.opp_corridor, po.opp_keep, po.has_raceline = False, 0, False
    po.online_p, po.b_online = 0.0, None
    po.line3, po.line3_sig = True, 0.15
    po.l3_idx = po.l3_w = po.l3_nw = None
    po.last_live = torch.zeros(po.B, po.P, dtype=torch.bool)
    po.kept = []
    po._write = lambda dst, src, xy, yaw, sid, live, P, Q: po.kept.append((xy, live))
    return po


def _draw(po, n=200):
    t = torch.zeros(n, dtype=torch.long)
    po._redraw_bank(t, torch.arange(n), torch.arange(n))
    return po.kept[-1]


def test_three_on_the_line_one_in_each_third():
    po = _po()
    po.set_driven_line(LINE, torch.ones(1, 48, dtype=torch.bool), near=0.40, bin_m=0.25)
    assert int(po.l3_nw[0]) == 24                                        # 48 bins of 0.25 m -> 24 stretches of 0.5 m
    xy, live = _draw(po)
    assert bool((live.sum(1) == 3).all())                                # three a draw
    assert bool((xy[..., 1].abs()[live] <= 0.4).all())                   # all within 0.4 m of the line
    xs = torch.sort(xy[..., 0][live].view(-1, 3), 1).values
    gaps = torch.cat([xs[:, 1:] - xs[:, :-1], (xs[:, :1] + L_M) - xs[:, 2:]], 1)
    assert float(gaps.min()) >= 0.5                                      # spread over the lap: one in each third
    near = (xy[..., 1].abs()[live] < 0.1).float().mean()
    assert float(near) > 0.6                                             # weighted toward the line (y = 0 over y = 0.2)


def test_unmeasured_line_draws_as_before():
    po = _po()
    ok = torch.zeros(1, 48, dtype=torch.bool)
    ok[0, :10] = True                                                    # 10 of 48 bins: not ready (< 80 % of a lap)
    po.set_driven_line(LINE, ok, near=0.40, bin_m=0.25)
    assert int(po.l3_nw[0]) == 0
    xy, live = _draw(po)
    assert bool((live.sum(1) == 3).all())                                # keep 3 of one layout, wherever they are
    assert bool((xy[..., 1].abs()[live] > 0.4).any())                    # some off the line: the layout's own pieces
