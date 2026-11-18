"""(11-18, the user: "잡음 크기에 따라서 update당 주행 rollout수를 유동적으로 하라는거지. 무작정 2배 하지 말고") `plan_rl.horizon_k`:
the rollout grows with the policy gradient's noise scale at once and, with --horizon-auto-down, comes back down one step after
the noise has asked for less long enough."""
from f1sim.learn.plan_rl import horizon_k

PER_K = 64 * 1024                       # --horizon 64 over 1024 cars: one K unit = 65,536 decisions


def test_grows_at_once_to_cover_the_noise():
    assert horizon_k(2, 0, 200_000, PER_K, 4, 2, 1.0, True, 8) == (4, 0)      # 200k needs 4 units
    assert horizon_k(2, 0, 150_000, PER_K, 4, 2, 1.0, True, 8) == (3, 0)


def test_stays_inside_the_bounds():
    assert horizon_k(2, 0, 5_000_000, PER_K, 4, 2, 1.0, True, 8) == (4, 0)    # never above KMAX
    assert horizon_k(2, 0, 1_000, PER_K, 4, 2, 1.0, True, 8) == (2, 0)        # never below KMIN


def test_comes_down_one_step_after_patience():
    k, low = 4, 0
    for _ in range(7):                                                           # 7 updates asking for 2: not yet
        k, low = horizon_k(k, low, 50_000, PER_K, 4, 2, 1.0, True, 8)
        assert k == 4
    k, low = horizon_k(k, low, 50_000, PER_K, 4, 2, 1.0, True, 8)              # the 8th: one step, not straight to 2
    assert (k, low) == (3, 0)


def test_a_higher_reading_resets_the_wait():
    k, low = horizon_k(4, 6, 300_000, PER_K, 4, 2, 1.0, True, 8)
    assert (k, low) == (4, 0)


def test_without_down_it_only_grows():
    k, low = 4, 0
    for _ in range(20):
        k, low = horizon_k(k, low, 10_000, PER_K, 4, 2, 1.0, False, 8)
    assert k == 4


def test_mult_scales_the_need():
    assert horizon_k(2, 0, 100_000, PER_K, 4, 2, 2.0, True, 8) == (4, 0)      # 2 x 100k = 200k -> 4 units
