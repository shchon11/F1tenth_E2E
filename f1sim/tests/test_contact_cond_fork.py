"""(11-26) plan_rl --contact-cond LO,HI --init <a checkpoint without it>: the fork. log(c) is the state's last input; in the
fork every weight that reads it starts at zero (`plan_rl.widen_state`: the trunk's first layer, the GRUs' inputs with
mem_front, the prediction head's first layer; `widen_critic_state`: the critic's first layer) and its statistics are mean 0 /
variance 1. So the fork's outputs do not depend on c at all until those weights move: they are the parent's for EVERY c, not
only at the range's geometric mean. Exact in the arithmetic: log(c)'s normalised value is finite (clamped to +-10), so its
term is 0 * x = +-0 and adds nothing. Against the parent the numbers agree to the sum's rounding only (a wider matrix product
groups its terms differently: ~1 ulp); across different c the fork's numbers are bit-identical (same shapes, same order, the
one differing term is a zero)."""
import pytest
import torch

from f1sim.learn import plan_intent as IN
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import Actor2, Critic, N_SCANS, STATE_DIM2

BEAMS = 90
MEM = 16
PRIV = 20
LOG_C = (-1.2, 0.0, 0.55, 2.3, -40.0, 40.0)        # the range 0.3..10, its middle, and far outside (clamped by the norm)

CASES = {
    "scan31b2": dict(memory=MEM, mem_front=True, mem_value=True),       # the parent of the tm39 arms: GRU before the trunk
    "plain": dict(),
    "mem_after_trunk": dict(memory=MEM, mem_value=True),                # its GRU reads the trunk's features, not the state
    "box_memory": dict(memory=MEM, mem_front=True, box_memory=8, aux_dim=12),
    "pred_opp": dict(memory=MEM, mem_front=True, mem_value=True, pred_opp=True),
}
WIDENED = {
    "scan31b2": ["trunk.0.weight", "gru.weight_ih"],
    "plain": ["trunk.0.weight"],
    "mem_after_trunk": ["trunk.0.weight"],
    "box_memory": ["trunk.0.weight", "gru.weight_ih", "gru_box.weight_ih"],
    "pred_opp": ["trunk.0.weight", "gru.weight_ih", "pred.mlp.0.weight"],
}


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCANS, BEAMS, generator=g), torch.randn(n, STATE_DIM2, generator=g)


def _parent(kw, seed=0):
    """a 'trained' parent: every weight and the input statistics moved off their start"""
    torch.manual_seed(seed)
    a = Actor2(BEAMS, **kw)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.1 * torch.randn_like(p))
        a.norm.mean.normal_(); a.norm.var.uniform_(0.5, 2.0); a.norm.count.fill_(1e8)
    return a.eval()


def _with_c(st, log_c):
    return torch.cat([st, torch.full((st.shape[0], 1), float(log_c))], 1)


@pytest.mark.parametrize("case", list(CASES))
def test_the_fork_drives_as_its_parent_for_every_c(case):
    kw = CASES[case]
    base = _parent(kw)
    fork = Actor2(BEAMS, state_dim=STATE_DIM2 + 1, **kw)
    sd = dict(base.state_dict())
    assert PR.widen_state(sd, fork, STATE_DIM2) == WIDENED[case]
    for k_ in WIDENED[case]:                     # one zero column right after the parent's state, the rest as saved
        at = PR.ENC_OUT + STATE_DIM2
        w0, w1 = base.state_dict()[k_], sd[k_]
        assert torch.equal(w1[:, :at], w0[:, :at]) and torch.equal(w1[:, at + 1:], w0[:, at:]) and bool((w1[:, at] == 0).all())
    assert PR.init_actor(fork, base.state_dict()) == ""
    fork.eval()
    assert float(fork.norm.mean[-1]) == 0.0 and float(fork.norm.var[-1]) == 1.0
    assert torch.equal(fork.norm.mean[:-1], base.norm.mean) and torch.equal(fork.norm.count, base.norm.count)
    mem = fork.memory
    n = 5
    hb = torch.zeros(n, mem) if mem else None
    hf = {lc: (torch.zeros(n, mem) if mem else None) for lc in LOG_C}
    for k in range(6):                                      # a few decisions, the memory carried
        sc, st = _inputs(n, k)
        with torch.no_grad():
            mb, lb, vb = base(sc, st, hb)
            hb = base.h_out
            outs = {}
            for lc in LOG_C:
                mf, lf, vf = fork(sc, _with_c(st, lc), hf[lc])
                hf[lc] = fork.h_out
                outs[lc] = (mf, lf, vf, hf[lc], fork.pred_out)
        ref = outs[LOG_C[0]]
        for lc in LOG_C:
            mf, lf, vf, h_, po = outs[lc]
            # the parent's, to the sum's rounding
            assert torch.allclose(mf, mb, rtol=0, atol=2e-6) and torch.equal(lf, lb)
            assert torch.allclose(vf, vb, rtol=0, atol=1e-5)
            if mem:
                assert torch.allclose(h_, hb, rtol=0, atol=2e-6)
                if hasattr(fork, "v_mem"):
                    assert torch.allclose(fork.vmem_read(h_), base.vmem_read(hb), rtol=0, atol=2e-6)
            if po is not None:
                assert torch.allclose(po, base.pred_out, rtol=0, atol=2e-6)
            # every c: bit-identical (c does not enter until its weights move)
            assert all(torch.equal(x, y) for x, y in zip(outs[lc], ref) if x is not None)
    # wired in: once the trunk's log(c) column moves, c changes the plan
    with torch.no_grad():
        fork.trunk[0].weight[:, PR.ENC_OUT + STATE_DIM2].normal_(0.0, 0.5)
    sc, st = _inputs(n, 9)
    h0 = torch.zeros(n, mem) if mem else None
    with torch.no_grad():
        m_lo, _, _ = fork(sc, _with_c(st, -1.2), h0)
        m_hi, _, _ = fork(sc, _with_c(st, 2.3), h0)
    assert (m_lo - m_hi).abs().max() > 1e-4


def test_the_fork_critic_values_as_its_parent_for_every_c():
    torch.manual_seed(5)
    old = Critic(BEAMS, PRIV, state_dim=STATE_DIM2)
    with torch.no_grad():
        for p in old.parameters():
            p.add_(0.1 * torch.randn_like(p))
        old.norm.mean.normal_(); old.norm.var.uniform_(0.5, 2.0)
    new = Critic(BEAMS, PRIV, state_dim=STATE_DIM2 + 1)
    c_sd = PR.widen_critic_state(old.state_dict(), STATE_DIM2, 1)
    new.load_state_dict(c_sd)
    assert float(new.norm.mean[STATE_DIM2]) == 0.0 and float(new.norm.var[STATE_DIM2]) == 1.0
    assert torch.equal(new.norm.mean[STATE_DIM2 + 1:], old.norm.mean[STATE_DIM2:])        # the privileged part kept
    assert bool((new.mlp[0].weight[:, PR.ENC_OUT + STATE_DIM2] == 0).all())
    sc, st = _inputs(6, 1)
    pv = torch.randn(6, PRIV)
    with torch.no_grad():
        v0 = old(sc, st, pv)
        vs = [new(sc, _with_c(st, lc), pv) for lc in LOG_C]
    for v in vs:
        assert torch.allclose(v, v0, rtol=0, atol=1e-5) and torch.equal(v, vs[0])
    # with --intent too: log(c) in the state's end first, then the intent's inputs at the privileged part's end
    both = Critic(BEAMS, PRIV + 5, state_dim=STATE_DIM2 + 1)
    sd2 = IN.widen_critic(c_sd, both)
    assert sd2 is not None
    both.load_state_dict(sd2)
    with torch.no_grad():
        vb = both(sc, _with_c(st, 1.7), torch.cat([pv, torch.rand(6, 5)], 1))
    assert torch.allclose(vb, v0, rtol=0, atol=1e-5)


def test_init_refusals(tmp_path):
    a = Actor2(BEAMS)
    ck_plain = tmp_path / "plain.pt"
    torch.save({"actor": a.state_dict(), "critic": {}, "args": {}, "beams": BEAMS}, ck_plain)
    b = Actor2(BEAMS, state_dim=STATE_DIM2 + 1)
    ck_cc = tmp_path / "cc.pt"
    torch.save({"actor": b.state_dict(), "critic": {}, "args": {"contact_cond": "0.3,10"}, "beams": BEAMS}, ck_cc)
    # a checkpoint without log(c): the fork is allowed (was refused before 11-26)
    with pytest.raises(SystemExit) as e:
        PR.main(["--name", "x", "--contact-cond", "0.3,10", "--init", str(ck_plain), "--check-args"])
    assert e.value.code == 0
    # with log(c): the same range only
    with pytest.raises(SystemExit) as e:
        PR.main(["--name", "x", "--contact-cond", "0.3,10", "--init", str(ck_cc), "--check-args"])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        PR.main(["--name", "x", "--contact-cond", "0.5,5", "--init", str(ck_cc), "--check-args"])
    assert "same range" in str(e.value.code)
    # a log(c) checkpoint without --contact-cond: refused (its state is one wider)
    with pytest.raises(SystemExit) as e:
        PR.main(["--name", "x", "--init", str(ck_cc), "--check-args"])
    assert "--contact-cond" in str(e.value.code)
    with pytest.raises(ValueError):
        PR.init_actor(Actor2(BEAMS), b.state_dict())
