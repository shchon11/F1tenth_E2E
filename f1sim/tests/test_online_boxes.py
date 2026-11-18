"""(11-18) --online-boxes: bank pieces near the policy's own driven line are marked (`set_driven_line`) and, in the share
`online_p` of the draws, kept first (`_redraw_bank`). Checked on a stand-in that carries only what those two read."""
import types

import torch

from f1sim.procedural_obstacles import ProceduralObstacles


def _po(online_p=0.0):
    po = object.__new__(ProceduralObstacles)
    po.device = torch.device("cpu")
    po.place = "bank"
    po.gen = torch.Generator().manual_seed(0)
    # one track, one layout of 4 pieces: two on the line y = 0 (x = 1, 3), two 1 m off it (x = 2, 4)
    po.b_sid = torch.tensor([[[0, 0, 0, 0]]])
    po.b_pose = torch.tensor([[[[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.1, 0.0], [4.0, -1.0, 0.0]]]])
    po.b_n = torch.tensor([1])
    po.C, po.P, po.B = 4, 4, 64
    po.count, po.keep, po.fraction = "uniform", 1, 1.0       # one piece kept per draw
    po.opp_corridor, po.opp_keep, po.has_raceline = False, 0, False
    po.online_p, po.b_online = online_p, None
    po.line3, po.l3_nw = False, None
    po.last_live = torch.zeros(po.B, po.P, dtype=torch.bool)
    po.kept = []
    po._write = lambda dst, src, xy, yaw, sid, live, P, Q: po.kept.append((xy, live))
    return po


LINE = torch.stack([torch.linspace(0, 5, 21), torch.zeros(21)], -1)[None]      # (1, 21, 2) along y = 0


def test_marks_the_pieces_on_the_line():
    po = _po()
    share = po.set_driven_line(LINE, torch.ones(1, 21, dtype=torch.bool))
    assert po.b_online.tolist() == [[[True, False, True, False]]] and abs(share - 0.5) < 1e-9


def test_unmeasured_line_marks_nothing():
    po = _po()
    po.set_driven_line(LINE, torch.zeros(1, 21, dtype=torch.bool))
    assert not bool(po.b_online.any())


def _kept_on_line(online_p, n=64):
    po = _po(online_p)
    po.set_driven_line(LINE, torch.ones(1, 21, dtype=torch.bool))
    t = torch.zeros(n, dtype=torch.long)
    po._redraw_bank(t, torch.arange(n), torch.arange(n))
    xy, live = po.kept[-1]
    on = (xy[..., 1].abs() < 0.4) & live                              # (n, 4) a kept piece on the line
    return float(on.any(1).float().mean())


def test_online_share_keeps_line_pieces_first():
    assert _kept_on_line(1.0) == 1.0                                  # always the line's pieces first
    base = _kept_on_line(0.0)                                         # off: one of four at random, half of them on the line
    assert 0.25 < base < 0.75
