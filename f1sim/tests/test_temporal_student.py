import torch

from f1sim.learn.dagger import StepBuffer
from f1sim.learn.model import ActorCritic, ScanStem, load_checkpoint, save_checkpoint


def test_scan_stem_adds_newest_to_older_differences() -> None:
    stem = ScanStem(3, 8, scan_deltas=True)
    scan = torch.tensor([[[3.] * 8, [2.] * 8, [0.] * 8]])
    features = stem.scan_features(scan)
    assert features.shape == (1, 5, 8)
    torch.testing.assert_close(features[:, :3], scan)
    torch.testing.assert_close(features[:, 3:], torch.tensor([[[1.] * 8, [2.] * 8]]))


def test_gru_scan_stem_encodes_fixed_window_without_external_state() -> None:
    stem = ScanStem(6, 64, temporal_encoder="gru")
    scan = torch.rand(3, 6, 64)
    first = stem(scan)
    second = stem(scan)
    assert first.shape == (3, 256)
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, stem(scan.flip(1)))


def test_temporal_checkpoint_roundtrip_preserves_architecture(tmp_path) -> None:
    model = ActorCritic(6, 64, 16, 8, act_dim=6, scan_deltas=True)
    path = tmp_path / "temporal.pt"
    save_checkpoint(str(path), model)
    loaded, _ = load_checkpoint(str(path))
    assert loaded.meta["scan_deltas"] is True
    scan = torch.rand(2, 6, 64)
    proprio = torch.rand(2, 16)
    torch.testing.assert_close(loaded.actor(scan, proprio), model.actor(scan, proprio))


def test_non_residual_experiment_metadata_remains_loadable(tmp_path) -> None:
    model = ActorCritic(2, 32, 8, 4)
    path = tmp_path / "legacy.pt"
    save_checkpoint(str(path), model)
    checkpoint = torch.load(path)
    checkpoint["meta"]["residual_plan"] = False
    torch.save(checkpoint, path)
    loaded, _ = load_checkpoint(str(path))
    assert loaded.meta["act_dim"] == 2


def test_dagger_buffer_reconstructs_strided_scans_without_crossing_reset() -> None:
    buffer = StepBuffer(k=3, stride=2)
    for step in range(7):
        scan = torch.tensor([[float(step)]])
        reset = torch.tensor([step == 4])
        buffer.add(scan, torch.zeros(1, 1), torch.zeros(1, 1), reset)
    buffer.finalize()
    sample = torch.tensor([3, 6])
    scan, _, _ = buffer.samples_at(sample, torch.zeros(2, dtype=torch.long), "cpu")
    torch.testing.assert_close(scan[:, :, 0], torch.tensor([[3., 1., 0.], [6., 4., 4.]]))
