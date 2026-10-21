"""No test may read or write the running user's console settings (`~/.f1sim/console.json`).

2026-10-21: the module-scoped `live` window of `test_sim_worker_console_integration` was built before the
per-test sandbox was in place and closed after it was gone, so it restored the user's real file and, on
close, wrote back what the tests had set -- "충돌해도 계속 주행" off and a pytest tmp dir as the recording
folder. The user's console then ended the episode at the first wall contact."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.viewer.console import prefs   # noqa: E402


@pytest.fixture(scope="session")
def session_level_path():
    return prefs.path()


@pytest.fixture(scope="module")
def module_level_path():
    return prefs.path()


def test_a_session_scoped_fixture_never_sees_the_users_file(session_level_path):
    assert session_level_path != prefs.PREFS_PATH


def test_a_module_scoped_fixture_never_sees_the_users_file(module_level_path):
    assert module_level_path != prefs.PREFS_PATH


def test_a_test_never_sees_the_users_file():
    assert prefs.path() != prefs.PREFS_PATH
