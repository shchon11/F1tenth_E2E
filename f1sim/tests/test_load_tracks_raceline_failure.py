"""A track whose raceline cannot be built is dropped loudly, not allowed to abort the whole load.

2026-09-18: one `icra2022_assets_mixed_1` whose minimum-time solve did not converge killed a console
DAgger launch after the other 5511 tracks had loaded. There is deliberately no fallback to an
unconverged line, so the only safe behaviours are: leave the track out (default), or raise.
"""
import pytest

from f1sim.learn import common


class _RL:
    pass


def _patch(monkeypatch, bad):
    monkeypatch.setattr(common.maps, "load", lambda n: n)

    def build(track, **kw):
        if track in bad:
            raise ValueError(f"{track}: minimum-time solve did not converge")
        return _RL()

    monkeypatch.setattr(common.Raceline, "build_cached", staticmethod(build))
    monkeypatch.setattr(common, "raceline_clearance", lambda t, rl: 1.0)


def test_one_failed_raceline_drops_that_track_only(monkeypatch, capsys):
    _patch(monkeypatch, {"b"})
    tracks, rls = common.load_tracks(["a", "b", "c"], racelines=True)
    assert tracks == ["a", "c"] and len(rls) == 2
    out = capsys.readouterr().out
    assert "dropping 1 track(s) whose raceline could not be built" in out and "b" in out


def test_strict_mode_still_raises(monkeypatch):
    _patch(monkeypatch, {"b"})
    with pytest.raises(ValueError, match="did not converge"):
        common.load_tracks(["a", "b"], racelines=True, drop_infeasible=False)


def test_every_track_failing_is_an_error_not_an_empty_set(monkeypatch):
    _patch(monkeypatch, {"a", "b"})
    with pytest.raises(ValueError, match="no track's raceline could be built"):
        common.load_tracks(["a", "b"], racelines=True)
