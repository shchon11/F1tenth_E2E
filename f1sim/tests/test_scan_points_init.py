"""`--scan-points` (11-06): an --init without it moves onto the new columns -- the pre-scan columns as they were, each new
beam the parent's nearest beam's weight over the beams sharing it, the prop / car point rows at zero, the parent's visible
prop slots dropped, the rest as it was (`state_rl.grow_to_scan_points`)."""
import torch

from f1sim.gym_env import PRIV_V2_PROP_DIM
from f1sim.learn import state_rl


def test_the_parent_columns_land_where_the_new_layout_reads_them():
    scan0, b0, b1, mid, tail, h = 5, 4, 10, 3, 2, 6
    d0 = scan0 + b0 + mid + PRIV_V2_PROP_DIM + tail
    vis0 = scan0 + 3 * b1 + mid
    D = vis0 + tail
    w = torch.arange(d0, dtype=torch.float32)[None].expand(h, d0).clone()   # column i holds i
    sd = {"pi.0.weight": w.clone(), "v.0.weight": w.clone()}
    nm = {"mean": torch.arange(d0, dtype=torch.float32), "var": torch.ones(d0), "n": 1e6}
    sd, nm = state_rl.grow_to_scan_points(sd, nm, scan0, b0, b1, vis0, D, "cpu")
    W = sd["pi.0.weight"][0]
    assert W.shape[0] == D and nm["mean"].shape[0] == D
    assert torch.equal(W[:scan0], torch.arange(scan0, dtype=torch.float32))
    near = torch.round(torch.arange(b1) * (b0 - 1) / (b1 - 1)).long()
    share = torch.bincount(near, minlength=b0).float()
    assert torch.allclose(W[scan0:scan0 + b1], (scan0 + near).float() / share[near])
    # summed over the new beams, each parent beam's weight comes back whole
    assert torch.allclose(torch.zeros(b0).index_add_(0, near, W[scan0:scan0 + b1]), (scan0 + torch.arange(b0)).float())
    assert torch.equal(nm["mean"][scan0:scan0 + b1], (scan0 + near).float())
    assert float(W[scan0 + b1:scan0 + 3 * b1].abs().max()) == 0.0
    assert float(nm["mean"][scan0 + b1:scan0 + 3 * b1].min()) == 1.0
    assert torch.equal(W[scan0 + 3 * b1:vis0], torch.arange(scan0 + b0, scan0 + b0 + mid, dtype=torch.float32))
    assert torch.equal(W[vis0:], torch.arange(d0 - tail, d0, dtype=torch.float32))      # the prop slots skipped
