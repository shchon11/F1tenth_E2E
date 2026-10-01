"""The maneuver head: K candidate plans, the teacher's choice among them, and each one's outcome."""
import torch

from f1sim.learn.maneuver import ManeuverHead, maneuver_loss
from f1sim.learn.model import ActorCritic, load_checkpoint, save_checkpoint

SMALL = dict(n_stack=2, n_beams=64, proprio_dim=6, priv_dim=4, act_dim=8, scan_stem="plain")


def test_the_action_is_the_chosen_candidates_plan_and_survives_a_save(tmp_path):
    torch.manual_seed(0)
    m = ActorCritic(**SMALL, maneuver={"K": 5})
    scan, pro = torch.rand(3, 2, 64), torch.rand(3, 6)
    out = m.actor.maneuver_forward(scan, pro)
    k = out["score"].argmax(1)
    want = out["plans"][torch.arange(3), k]
    assert torch.allclose(m.actor.forward(scan, pro), want)
    p = tmp_path / "m.pt"; save_checkpoint(str(p), m, {})
    m2, _ = load_checkpoint(str(p))
    assert m2.meta["maneuver"]["K"] == 5
    assert torch.allclose(m2.actor.forward(scan, pro), want)


def test_choosing_never_averages_two_good_options():
    # two equally good moves, left and right: the head learns both plans and picks one, the plain
    # head's regression would sit in the middle
    torch.manual_seed(1)
    head = ManeuverHead(4, 1, K=2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    feat = torch.zeros(64, 4)
    plans_t = torch.tensor([[-0.8], [0.8]]).expand(64, 2, 1)
    choice = torch.tensor([0, 1] * 32)
    for _ in range(300):
        l = maneuver_loss(head(feat), plans_t, choice, torch.zeros(64, 2), torch.zeros(64, 2), torch.zeros(64))
        opt.zero_grad(); (l["plan"] + l["choice"]).backward(); opt.step()
    a = ManeuverHead.select(head(feat[:1]))
    assert abs(float(a)) > 0.6, a                              # one of the two, not the middle
