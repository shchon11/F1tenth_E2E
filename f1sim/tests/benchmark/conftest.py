"""Fixtures for the benchmark tests.

The package is a normal subpackage of `f1sim` here, so there is no path attachment to do -- the
`bench` fixture stays only because every test takes it, and it now just proves the import works.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def bench():
    import f1sim.learn.benchmark as b
    return b


@pytest.fixture(scope="session")
def vehicle_length():
    """The real value from the repo. The pass thresholds derive from it, so a drift shows up here."""
    from f1sim.params import VehicleParams
    return float(VehicleParams().length)


def pytest_configure(config):
    """Register the marker locally rather than editing the repo-wide pytest configuration."""
    config.addinivalue_line("markers",
                            "slow: builds a real simulator; CPU only, seconds not milliseconds")
