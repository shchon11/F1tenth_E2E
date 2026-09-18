"""Invalid teacher fallbacks remain history but never become supervised targets."""
from types import SimpleNamespace

import pytest
import torch

from f1sim.learn import dagger


def buffer(valid=None, invalid_target=float('nan')):
    buf = dagger.StepBuffer(k=2)
    for t in range(4):
        ok = True if valid is None else valid[t]
        label = torch.full((1, dagger.ACT_DIM), 0.2 if ok else invalid_target)
        buf.add(torch.full((1, 5), float(t)), torch.tensor([[float(t)]]), label,
                torch.tensor([t in (0, 2)]), gap=torch.tensor([1.0 if ok else float('nan')]),
                valid=None if valid is None else torch.tensor([ok]))
    return buf.finalize()


@pytest.mark.parametrize('hard_frac', [0.0, 0.5, 1.0])
def test_flat_and_hard_sampling_exclude_invalid(hard_frac):
    buf = buffer([False, True, False, True])
    scan, pro, lab = buf.sample(100, 'cpu', hard_frac)
    assert set(pro[:, 0].tolist()) == {1.0, 3.0}
    assert torch.isfinite(lab).all()
    # Previous invalid observations still appear in scan history.
    torch.testing.assert_close(scan[:, 1, 0], pro[:, 0] - 1)


def test_chunks_keep_invalid_frames_and_episode_boundaries():
    buf = buffer([True, False, False, True])
    scan, pro, lab, keep, valid = buf.sample_chunks(1, 4, 'cpu', return_valid=True)
    assert pro[:, 0, 0].tolist() == [0, 1, 2, 3]
    assert keep[:, 0].tolist() == [False, True, False, True]
    assert valid[:, 0].tolist() == [True, False, False, True]
    assert len(buf) == 4 and buf.valid_count == 2
    assert len(buf.sample_chunks(1, 4, 'cpu')) == 4


class Actor(torch.nn.Module):
    has_memory = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.05))
        self.frames = 0
        self.history = []

    def embed(self, scan, proprio):
        self.frames += len(proprio)
        return proprio, None

    def motion_input(self, scan):
        return None

    def head(self, x, unused, h, rows=None):
        h = x[None] * self.weight if h is None else h + x[None] * self.weight
        self.history.append(h.detach().clone())
        return h[0], h, None

    def mu(self, feats):
        return feats.expand(-1, dagger.ACT_DIM)


def train(buf):
    actor = Actor()
    opt = torch.optim.SGD(actor.parameters(), lr=0.1)
    rows = []
    loss = dagger.train_epochs(SimpleNamespace(actor=actor), [buf], 1, 4, 'cpu', opt,
                               rows.append, chunk=4)
    return actor, loss, rows, opt


def test_invalid_targets_have_zero_influence_but_frames_advance_state():
    a, loss_a, _, _ = train(buffer([False, False, False, True], float('nan')))
    b, loss_b, _, _ = train(buffer([False, False, False, True], 1e9))
    torch.testing.assert_close(a.weight, b.weight)
    assert loss_a == loss_b and torch.isfinite(torch.tensor(loss_a))
    assert a.frames == 4
    # Episode boundary at 2 resets history; invalid frame 2 still contributes to valid 3.
    assert a.history[-1].item() == pytest.approx(0.25)


def test_all_invalid_skips_optimizer_and_is_explicit():
    buf = buffer([False] * 4)
    with pytest.raises(ValueError, match='no valid teacher labels'):
        buf.sample(1, 'cpu')
    actor, loss, rows, opt = train(buf)
    assert loss == 0 and actor.frames == 0 and actor.weight.item() == pytest.approx(0.05)
    assert not opt.state and actor.weight.grad is None
    assert rows[-1] == {'dagger/skipped_no_valid_labels': 1, 'dagger/optimizer_updates': 0}


def test_all_valid_default_preserves_chunk_api_and_training():
    buf = buffer()
    assert buf.V.all() and buf.valid_count == len(buf)
    a, loss_a, _, _ = train(buf)
    b, loss_b, _, _ = train(buffer([True] * 4))
    assert loss_a == loss_b
    torch.testing.assert_close(a.weight, b.weight)


def test_collection_copies_teacher_validity_for_primary_rows(monkeypatch):
    teacher = SimpleNamespace(last_label_valid=None)

    class Env:
        B, M = 4, 2

        def reset(self):
            return None, {}

        def teacher_label(self, obj):
            obj.last_label_valid = torch.tensor([True, False, False, True])
            return torch.zeros(4, dagger.ACT_DIM)

        def step(self, action):
            # The teacher may reuse/mutate its diagnostic tensor on the next step.
            teacher.last_label_valid.fill_(True)
            return None, None, None, None, {}

    monkeypatch.setattr(dagger, 'flatten_obs', lambda obs: (torch.zeros(4, 1, 5), torch.zeros(4, 1)))
    runtime = SimpleNamespace(scan=None, observe=lambda scan, pro: scan, reset=lambda done: None)
    monkeypatch.setattr(dagger, 'runtime_for', lambda *args: runtime)
    buf = dagger.collect(Env(), object(), teacher, 1, 1.0, 'cpu', dagger.StepBuffer(1)).finalize()
    assert buf.V.tolist() == [[True, False]]
    assert buf.N.tolist() == [[True, True]]
