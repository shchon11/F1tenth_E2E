"""`dagger_v2.explicit_flags`: the dests a v2 command line set (ppo / dagger `--arch v2`)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn.dagger_v2 import explicit_flags   # noqa: E402


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rand-range", action="append", default=[])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    return ap


def test_an_append_option_is_given_rather_than_a_crash():
    """2026-10-21: `--arch v2 ... --rand-range A --rand-range B` died in argparse on the object() sentinel."""
    argv = ["--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
    assert explicit_flags(parser(), argv) == {"rand_range"}


def test_unset_options_are_not_given_and_set_ones_are_even_at_their_default():
    assert explicit_flags(parser(), []) == set()
    assert explicit_flags(parser(), ["--lr", "1e-4", "--amp"]) == {"lr", "amp"}
