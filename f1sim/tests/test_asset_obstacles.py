"""Asset-only random modes: geometry, seeded replay, density and historical compatibility."""
from dataclasses import replace
import numpy as np
import pytest

from f1sim import tracks, props
from f1sim.track import Track, StaticProp
from f1sim.asset_obstacles import scaled_prop, with_asset_obstacles


@pytest.fixture
def circuit():
    res = .05
    y, x = np.mgrid[:400, :400] * res
    r = np.hypot(x - 10, y - 10)
    occ = (r < 5) | (r > 9)
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    line = np.stack((10 + 7 * np.cos(theta), 10 + 7 * np.sin(theta)), axis=1)
    return Track.from_occupancy(occ, res, centerline=line, name="asset-test")


@pytest.mark.parametrize("style", props.STYLES)
def test_scaling_uses_one_shared_geometry(style):
    a = StaticProp(style, 1, 2, seed=6)
    b = scaled_prop(a, 1.7)
    aa, bb = a.build(), b.build()
    np.testing.assert_allclose(bb.envelope.footprint, aa.envelope.footprint * 1.7, atol=1e-8)
    assert bb.envelope.height == pytest.approx(aa.envelope.height * 1.7)
    # Builders may use unscaled decorative bevels. The footprint/height enclosing their
    # visual mesh stays authoritative for contacts, LiDAR and teacher projection.
    for part in bb.parts:
        assert part.pos[:, 2].max() <= bb.envelope.height + 1e-6


@pytest.mark.parametrize("family", ("edge", "line", "hard"))
def test_assets_only_and_seeded_dimensions(circuit, family):
    a = with_asset_obstacles(circuit, family, 44)
    b = with_asset_obstacles(circuit, family, 44)
    c = with_asset_obstacles(circuit, family, 45)
    assert a.props == b.props and a.props != c.props
    assert all(p.style in props.STYLES and p.dims for p in a.props)
    np.testing.assert_array_equal(a.occupancy, circuit.occupancy)
    np.testing.assert_array_equal(a.tall, circuit.tall)
    assert len({(p.style, p.dims) for p in a.props}) > 1 or family == "edge"


def test_density_increases_and_high_uses_existing_patterns(circuit):
    low, medium, high = [with_asset_obstacles(circuit, f, 44) for f in ("edge", "line", "hard")]
    assert len(low.props) < len(medium.props) < len(high.props)
    from f1sim.hard_obstacles import with_hard_obstacles
    recipe = with_hard_obstacles(circuit, 44)
    assert high.hard_patterns == recipe.hard_patterns
    assert len(high.props) == len(recipe.hard_boxes)
    for p, (x, y, yaw, sx, sy) in zip(high.props, recipe.hard_boxes):
        assert (p.x, p.y, p.yaw) == (x, y, yaw)
        assert np.all(np.ptp(p.build().envelope.footprint, axis=0) <= np.array([sx, sy]) + 1e-8)


def test_authored_default_and_bare_are_unmodified(circuit):
    p = StaticProp("wooden_crate", 17, 10)
    circuit.props = (p,)
    assert with_asset_obstacles(circuit, "", 1).props == (p,)
    assert tracks.asset_scenario("scene/demo") == "scene/demo"
    assert tracks.asset_scenario("scene/demo#bare") == "scene/demo#bare"


@pytest.mark.parametrize("name", ("real/icra22@mir+rev#line:*", "rt/monza#edge:8", "scene/demo#bare+hard:7"))
def test_scenario_marker_roundtrip(name):
    spec = tracks.asset_scenario(name)
    sc = tracks.parse(spec)
    assert sc.asset == "mixed" and sc.scale == 1
    assert tracks.parse(sc.with_seed(8).legacy()) == sc.with_seed(8)
    assert all("!assets=mixed:1" in n for n in tracks.expand(spec, 2))


def test_old_scenario_unchanged():
    assert tracks.resolve("real/icra22#line:7") == "real:icra2022+rlobs7"
    assert tracks.parse("real:icra2022+rlobs7").asset == ""


@pytest.mark.parametrize("suffix", ("mixed:nan", "mixed:-1", "unknown:1", "mixed:0"))
def test_invalid_asset_contract_rejected(suffix):
    with pytest.raises(ValueError):
        tracks.parse("real/icra22#line:7!assets=" + suffix)


def test_actual_loader_keeps_analytic_assets(circuit, monkeypatch):
    from f1sim import maps
    monkeypatch.setattr(maps, "_load_base", lambda name, **kw: circuit)
    monkeypatch.setattr(maps, "_BASE_CACHE", {})
    t = maps.load("real/icra22@rev#line:44!assets=mixed:1")
    assert t.props
    np.testing.assert_array_equal(t.occupancy, circuit.occupancy)
    assert t.centerline.shape == circuit.centerline.shape


def test_high_patterns_account_for_authored_geometry(circuit, monkeypatch):
    import f1sim.hard_obstacles as hard
    circuit.props = (StaticProp("cardboard_box", 17., 10.),)
    original = hard.with_hard_obstacles
    seen = []
    def inspect(track, *args, **kwargs):
        seen.append(int((track.occupancy & ~circuit.occupancy).sum()))
        return original(track, *args, **kwargs)
    monkeypatch.setattr(hard, "with_hard_obstacles", inspect)
    result = with_asset_obstacles(circuit, "hard", 44, n=2)
    assert seen[0] > 0
    assert result.props[0] == circuit.props[0]
    np.testing.assert_array_equal(result.occupancy, circuit.occupancy)
