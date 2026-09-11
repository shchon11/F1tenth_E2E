"""The resnet scan stem, and the properties it exists to provide.

Measured failure mode it targets: on convoluted maps 69 % of collisions happen with the car already
outside the lane at ~2.2 m/s, i.e. it takes a wrong branch rather than losing grip. That is a
perception failure, and the plain stem's receptive field before its single flatten Linear is ~79
beams (~20 deg of a 270 deg scan).
"""
import pytest
import torch

from f1sim.learn.model import ActorCritic, ScanStem, load_checkpoint, save_checkpoint

BEAMS, STACK = 1080, 6


def _receptive_field(stem: ScanStem, trunk) -> int:
    """Beams whose gradient reaches one central trunk output."""
    x = torch.zeros(1, STACK, BEAMS, requires_grad=True)
    h = trunk(stem.scan_features(x) if stem.scan_stem == "plain" else stem._augment(stem.scan_features(x)))
    h[0, :, h.shape[2] // 2].sum().backward()
    nonzero = (x.grad.abs().sum((0, 1)) > 0).nonzero().flatten()
    return int(nonzero[-1] - nonzero[0] + 1)


def test_resnet_stem_sees_far_more_of_the_scan_than_the_plain_stem():
    plain = ScanStem(STACK, BEAMS, scan_deltas=True, scan_stem="plain")
    resnet = ScanStem(STACK, BEAMS, scan_deltas=True, scan_stem="resnet")
    assert _receptive_field(plain, plain.conv) < 150            # ~79 beams, ~20 deg
    assert _receptive_field(resnet, resnet.trunk) > 600         # whole-scan context


def test_sector_profile_carries_absolute_range_past_the_normalisation():
    # GroupNorm standardises over the beam axis, so a uniformly nearer scan would otherwise look
    # identical to a uniformly farther one. The raw per-sector nearest return must not.
    stem = ScanStem(STACK, BEAMS, scan_deltas=True, scan_stem="resnet").eval()
    near, far = torch.full((1, STACK, BEAMS), 0.2), torch.full((1, STACK, BEAMS), 0.9)
    assert not torch.allclose(stem._sector_profile(near), stem._sector_profile(far))
    with torch.no_grad():
        assert not torch.allclose(stem(near), stem(far))


def test_sector_profile_reports_the_nearest_return_not_the_average():
    # A single close return inside an otherwise open sector is what decides a collision.
    stem = ScanStem(STACK, BEAMS, scan_stem="resnet")
    scan = torch.full((1, STACK, BEAMS), 0.9)
    scan[0, 0, 5] = 0.05
    profile = stem._sector_profile(scan)
    assert profile[0, 0].item() == pytest.approx(0.05)          # first sector takes the minimum
    assert profile[0, 1].item() == pytest.approx(0.9)


def test_beam_angle_channel_breaks_translation_symmetry():
    # 270 deg of FOV: an obstacle to the left and the same obstacle to the right are different
    # situations, so the stem must not be translation invariant along the beam axis.
    stem = ScanStem(STACK, BEAMS, scan_deltas=True, scan_stem="resnet").eval()
    left, right = torch.full((1, STACK, BEAMS), 0.9), torch.full((1, STACK, BEAMS), 0.9)
    left[0, :, 100:160] = 0.1
    right[0, :, BEAMS - 160:BEAMS - 100] = 0.1
    with torch.no_grad():
        assert not torch.allclose(stem(left), stem(right), atol=1e-6)


def test_both_stems_run_under_both_temporal_encoders():
    for stem in ("plain", "resnet"):
        for encoder, deltas in (("cnn", True), ("gru", False)):
            model = ActorCritic(STACK, BEAMS, 322, 17, act_dim=6, scan_deltas=deltas,
                                temporal_encoder=encoder, scan_stem=stem)
            scan, proprio, priv = torch.rand(3, STACK, BEAMS), torch.rand(3, 322), torch.rand(3, 17)
            action, logp = model.act(scan, proprio)
            assert action.shape == (3, 6) and logp.shape == (3,)
            assert model.critic(scan, proprio, priv).shape == (3,)


def test_checkpoints_without_scan_stem_load_as_the_original_architecture(tmp_path):
    # Every existing run predates the field; none of them may silently change shape.
    original = ActorCritic(STACK, BEAMS, 322, 17, act_dim=6, scan_deltas=True)
    meta = dict(original.meta)
    meta.pop("scan_stem")
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": original.state_dict(), "meta": meta, "extra": {}}, path)

    restored, extra = load_checkpoint(path)
    assert extra["skipped"] == []
    assert restored.meta["scan_stem"] == "plain"
    scan, proprio = torch.rand(2, STACK, BEAMS), torch.rand(2, 322)
    torch.testing.assert_close(restored.actor(scan, proprio), original.actor(scan, proprio))


def test_resnet_checkpoints_round_trip(tmp_path):
    model = ActorCritic(STACK, BEAMS, 322, 17, act_dim=6, scan_deltas=True, scan_stem="resnet")
    path = tmp_path / "resnet.pt"
    save_checkpoint(path, model)
    restored, _ = load_checkpoint(path)
    assert restored.meta["scan_stem"] == "resnet"
    scan, proprio = torch.rand(2, STACK, BEAMS), torch.rand(2, 322)
    torch.testing.assert_close(restored.actor(scan, proprio), model.actor(scan, proprio))
