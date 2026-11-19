"""(11-19) --mem-aux-seen: the memory's side task lists only the boxes the car has already passed in this race
(`boxes_passed`)."""
import torch

from f1sim.learn.plan_rl import boxes_passed

NAN = float("nan")


def _step(seen, prev_ds, prev_pos, ds, live, pos):
    return boxes_passed(seen, prev_ds, prev_pos, torch.tensor(ds), torch.tensor(live), torch.tensor(pos))


def test_a_box_counts_once_the_car_went_by_it():
    pos = [[[1.0, 2.0], [5.0, 5.0]]]
    seen, pds, ppos = torch.zeros(1, 2, dtype=torch.bool), torch.full((1, 2), NAN), torch.zeros(1, 2, 2)
    seen, pds, ppos = _step(seen, pds, ppos, [[2.5, 8.0]], [[True, True]], pos)       # first look: nothing to compare with
    assert not seen.any()
    seen, pds, ppos = _step(seen, pds, ppos, [[0.4, 6.0]], [[True, True]], pos)       # still ahead
    assert not seen.any()
    seen, pds, ppos = _step(seen, pds, ppos, [[-0.3, 5.3]], [[True, True]], pos)      # beside / just behind: passed
    assert seen.tolist() == [[True, False]]
    seen, pds, ppos = _step(seen, pds, ppos, [[-20.0, 1.0]], [[True, True]], pos)     # far behind: still known
    assert seen.tolist() == [[True, False]]
    assert torch.isfinite(pds).all() and torch.equal(ppos, torch.tensor(pos))


def test_moved_or_removed_boxes_are_forgotten():
    pos = torch.tensor([[[1.0, 2.0], [5.0, 5.0]]])
    seen = torch.ones(1, 2, dtype=torch.bool)
    pds = torch.tensor([[-5.0, -6.0]])
    moved = pos.clone(); moved[0, 0] += 0.5                                           # slot 0 drawn elsewhere
    seen2, pds2, _ = boxes_passed(seen, pds, pos, torch.tensor([[4.0, -6.5]]), torch.tensor([[True, True]]), moved)
    assert seen2.tolist() == [[False, True]]
    seen3, pds3, _ = boxes_passed(seen, pds, pos, torch.tensor([[-5.5, -6.5]]), torch.tensor([[True, False]]), pos)
    assert seen3.tolist() == [[True, False]] and torch.isnan(pds3[0, 1])             # slot 1 removed


def test_no_pass_from_a_jump_or_driving_backwards():
    pos = [[[1.0, 2.0]]]
    seen, pds, ppos = torch.zeros(1, 1, dtype=torch.bool), torch.tensor([[NAN]]), torch.tensor(pos)
    seen, pds, ppos = _step(seen, pds, ppos, [[-0.5]], [[True]], pos)                # a restart beside the box (NaN before)
    assert not seen.any()
    seen, pds, ppos = _step(seen, pds, ppos, [[0.5]], [[True]], pos)                 # backwards past it
    assert not seen.any()
    seen, pds, ppos = _step(seen, torch.tensor([[6.0]]), ppos, [[-0.5]], [[True]], pos)   # 6.5 m in one decision: a teleport
    assert not seen.any()
