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
    k = ManeuverHead.rank(out).argmax(1)
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


def test_the_ppo_policy_is_the_candidate_mixture_and_its_log_prob_is_exact():
    import math
    from f1sim.learn.maneuver import ManeuverDist
    torch.manual_seed(2)
    out = {"score": torch.tensor([[2.0, 0.0]]), "plans": torch.tensor([[[0.5], [-0.5]]]),
           "risk": torch.full((1, 2), -30.0)}                    # no predicted contact: the rank is the score
    d = ManeuverDist(out, torch.tensor([0.2]))
    a = torch.tensor([[0.1]])
    p = torch.softmax(out["score"], 1)[0]
    n = lambda mu: math.exp(-0.5 * ((0.1 - mu) / 0.2) ** 2) / (0.2 * math.sqrt(2 * math.pi))
    want = math.log(float(p[0]) * n(0.5) + float(p[1]) * n(-0.5))
    assert abs(float(d.log_prob(a).sum(1)) - want) < 1e-5
    assert torch.allclose(d.mean, torch.tensor([[0.5]]))


def test_acting_and_ppo_re_evaluation_agree_on_the_log_prob():
    torch.manual_seed(3)
    m = ActorCritic(**SMALL, maneuver={"K": 4})
    scan, pro, priv = torch.rand(5, 2, 64), torch.rand(5, 6), torch.rand(5, 4)
    a, lp, _ = m.act(scan, pro)
    lp2 = m.evaluate(scan, pro, priv, a)[0]
    inside = (a.abs() < 1).all(1)                  # act scores the sample before its clamp, as the Normal head does
    assert inside.any() and torch.allclose(lp[inside], lp2[inside], atol=1e-5)
    assert a.shape == (5, 8)


def test_a_candidate_predicted_to_hit_is_vetoed_even_if_the_teacher_logit_prefers_it():
    out = {"score": torch.tensor([[3.0, 2.0]]), "plans": torch.tensor([[[0.5], [-0.5]]]),
           "risk": torch.tensor([[6.0, -6.0]])}                  # candidate 0: sure contact
    assert float(ManeuverHead.select(out)) == -0.5


def _transform_model():
    spec = {"K": 3, "mode": "transform", "cand_off": [0.0, 0.4, -0.4], "cand_spd": [1.0, 1.0, 0.6], "i_line": 0,
            "v_max": 10.0, "k_lim": 1.2}
    return ActorCritic(**SMALL, maneuver=spec)


def test_a_transform_head_executes_its_one_plan_shifted_to_the_chosen_candidate(tmp_path):
    torch.manual_seed(4)
    m = _transform_model()
    scan, pro = torch.rand(4, 2, 64), torch.rand(4, 6)
    pro[:, 0] = 0.5                                              # 5 m/s
    with torch.no_grad():
        m.actor.maneuver.score.bias.copy_(torch.tensor([5.0, 0.0, 0.0])); m.actor.maneuver.score.weight.zero_()
        m.actor.maneuver.risk.bias.fill_(-10.0); m.actor.maneuver.risk.weight.zero_()
    base = m.actor.maneuver_forward(scan, pro)["base"]
    assert torch.allclose(m.actor.forward(scan, pro), base, atol=1e-6)    # the plain line: the plan itself
    with torch.no_grad():
        m.actor.maneuver.score.bias.copy_(torch.tensor([0.0, 0.0, 5.0]))
    a = m.actor.forward(scan, pro)
    assert (a[:, :6] - base[:, :6]).abs().max() > 1e-3          # shifted path
    assert torch.allclose(a[:, 6:], ((base[:, 6:] + 1) * 0.6 - 1).clamp(-1, 1), atol=1e-5)   # slower
    a.sum().backward()
    assert m.actor.mu.weight.grad.abs().sum() > 0               # the gradient reaches the plain plan
    p = tmp_path / "t.pt"; save_checkpoint(str(p), m, {})
    m2, _ = load_checkpoint(str(p))
    assert torch.allclose(m2.actor.forward(scan, pro), a.detach(), atol=1e-5)


def test_a_recurrent_transform_actor_re_evaluates_a_sequence():
    from f1sim.learn.memory import memory_spec
    torch.manual_seed(5)
    spec = {"K": 3, "mode": "transform", "cand_off": [0.0, 0.4, -0.4], "cand_spd": [1.0, 1.0, 0.6], "i_line": 0,
            "v_max": 10.0, "k_lim": 1.2}
    m = ActorCritic(**SMALL, maneuver=spec, memory=memory_spec(hidden_size=16))
    T, B = 3, 2
    scan, pro, priv = torch.rand(T, B, 2, 64), torch.rand(T, B, 6), torch.rand(T, B, 4)
    act = torch.zeros(T, B, 8)
    out = m.evaluate_sequence(scan, pro, priv, act, None, m.initial_hidden(B), torch.ones(T, B))
    assert out[0].shape == (T * B,)
