"""`learn.scene_replay.SceneReplay` (11-13): a race restarted from a stored moment. The store, which races take part,
the moment coming back exactly (car, boxes, and the same drive from it), the rule that a scene starts on a decision, and
the episode's length."""
import pytest
import torch

from f1sim.learn.scene_replay import SceneReplay

SUB = 4


def _env(n=6):
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    from f1sim.params import Config
    trs, rls = common.load_tracks(["gen:competition:0"], racelines=False)
    cfg = Config(); cfg.sim.compile = False
    env = common.make_env(trs, n, "cpu", EnvConfig(action_mode="direct", race_size=1, procedural_obstacles=1.0,
                                                    procedural_density=1.5, procedural_max_props=10,
                                                    procedural_raceline_corridor="off", spawn_runway=3.0), cfg=cfg, seed=0)
    env.reset(seed=5)
    return env


def _act(k, n):
    a = torch.zeros(n, 2)
    a[:, 0] = 0.3 * torch.sin(torch.tensor(0.4 * k)); a[:, 1] = 0.5
    return a


def test_the_store_is_a_ring_and_its_entries_are_fixed():
    env = _env(2)
    sc = SceneReplay(env, share=0.5, size=3, episode_steps=40)
    assert sc.races.tolist() == [True, False]
    mk = lambda v: {"a": torch.full((1, 2), float(v)), "b": torch.tensor([v])}
    for v in range(5):
        sc.add(mk(v))
    assert sc.n == 3 and sorted(sc.data["b"].tolist()) == [2, 3, 4] and sc.n_added == 5
    with pytest.raises(KeyError):
        sc.add({"a": torch.zeros(1, 2)})
    sc.add({"a": torch.zeros(0, 2), "b": torch.zeros(0, dtype=torch.long)})        # nothing: nothing changes
    assert sc.n == 3


def test_keep_unsolved_gives_up_the_scene_driven_clean_most_and_keeps_the_failing_ones():
    """(11-17) `keep_unsolved`: a full store gives up the scene with the longest clean streak (oldest among equals), never a
    slot a race is replaying now; a failed replay sets the streak back to 0"""
    env = _env(2)
    sc = SceneReplay(env, share=0.5, size=3, episode_steps=40)
    sc.keep_unsolved = True
    mk = lambda v: {"a": torch.full((1, 2), float(v)), "b": torch.tensor([v])}
    for v in range(3):
        sc.add(mk(v))                                            # slots 0, 1, 2 hold 0, 1, 2
    # scene 1 driven clean twice, scene 2 once, scene 0 failed
    for slot, hit in ((1, False), (1, False), (2, False), (0, True)):
        sc.cur[0] = slot
        sc.report(torch.tensor([0]), torch.tensor([hit]))
    assert sc.clean.tolist() == [0, 2, 1] and sc.tries.tolist() == [1, 2, 1] and int(sc.cur[0]) == -1
    sc.add(mk(3))                                                # the cleanest (scene 1) makes room
    assert sorted(sc.data["b"].tolist()) == [0, 2, 3]
    sc.cur[0] = 2                                                # scene 2 is being replayed: not given up now
    sc.add(mk(4))
    assert sorted(sc.data["b"].tolist()) == [2, 3, 4]            # 0 (oldest, never clean) went, 2 (busy) stayed
    sc.report(torch.tensor([0]), torch.tensor([True]))           # scene 2's replay failed: its streak is 0 again
    assert int(sc.clean[(sc.data["b"] == 2).nonzero()[0, 0]]) == 0


def test_a_stored_moment_comes_back_exactly_and_drives_the_same():
    env = _env()
    n = env.B
    sc = SceneReplay(env, share=0.5, size=8, episode_steps=40, seed=3)
    env.scene_replay = sc
    for k in range(30):
        sc.to_boundary = SUB - 1 - k % SUB
        env.step(_act(k, n))
    assert not sc.in_replay.any() and sc.n_loaded == 0                             # an empty store changes nothing
    snap = sc.capture()
    sc.add({k: v[[1]] for k, v in snap.items()})                                   # race 1, an ordinary one, is the scene
    ref = []
    for k in range(12):                                                            # what race 1 does from that moment
        sc.to_boundary = 1                                                         # (off a decision: nothing may load)
        env.step(_act(k, n)); ref.append(env.sim.state[1].clone())
    sc.to_boundary = 0
    env._reset_envs(torch.tensor([0, 1]))
    g, idx = sc.take_loaded()
    assert g.tolist() == [0] and idx.tolist() == [0]                               # race 0 takes scenes, race 1 never
    assert sc.in_replay.tolist()[:2] == [True, False]
    assert int(env.ep_step[0]) == 0 and int(env.ep_limit[0]) == 40
    assert torch.equal(env.sim.state[0], snap["sim_state"][1, 0])
    assert int(env.sim.tid[0]) == int(snap["tid"][1])
    assert torch.equal(env.procedural.p_poses[0], snap["box_p_poses"][1])
    assert torch.equal(env.procedural.p_sid[0], snap["box_p_sid"][1])
    assert torch.equal(env.sim.params.buf[:, 0], snap["params"][1, 0])
    for k in range(12):
        sc.to_boundary = 1
        env.step(_act(k, n))
        assert torch.allclose(env.sim.state[0], ref[k], atol=1e-5), k


def test_a_scene_starts_on_a_decision_and_its_episode_has_its_own_length():
    env = _env()
    n = env.B
    sc = SceneReplay(env, share=0.5, size=8, episode_steps=8, seed=3)
    env.scene_replay = sc
    for k in range(8):
        sc.to_boundary = SUB - 1 - k % SUB
        env.step(_act(k, n))
    sc.add({k: v[[1, 3]] for k, v in sc.capture().items()})
    env.ep_limit[0] = env.ep_step[0] + 2                                           # ends in the middle of a decision
    loaded = []
    for k in range(8, 8 + 16):
        sc.to_boundary = SUB - 1 - k % SUB
        env.step(_act(k, n))
        loaded.append(bool((sc.take_loaded()[0] == 0).any()))                      # race 0 (other scene races end as they do)
        if k == 9:                                                                 # the fresh grid lasts to the boundary
            assert not sc.in_replay[0] and int(env.ep_limit[0]) == 2
    # steps 8..23: the cut at step 9, the scene on the boundary (step 11), then every 8 steps (step 19)
    assert [k for k, v in zip(range(8, 24), loaded) if v] == [11, 19]
    assert bool(sc.in_replay[0]) and not sc.in_replay[[1, 3, 5]].any()             # the odd races never take a scene


def test_scene_replay_and_level_replay_together_are_refused():
    env = _env(2)
    env.scene_replay = SceneReplay(env, share=1.0, size=2, episode_steps=8)
    env.level_replay = object()
    with pytest.raises(RuntimeError):
        env._reset_envs(torch.tensor([0]))
