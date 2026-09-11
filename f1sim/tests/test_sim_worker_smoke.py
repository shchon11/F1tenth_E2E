"""End-to-end smoke: a real spawned worker process, a real checkpoint, a real environment.

The lifecycle tests stub the simulator so they can pin races and acks; this file does the opposite
and stubs nothing, because "the protocol is consistent" and "the thing actually drives a policy"
are different claims and only one of them is worth a slow test.

It is skipped, not failed, when there is no checkpoint under `~/f1sim_runs` -- the repository does
not carry one. It never trains, never writes to a run directory, and only ever signals the process
it started itself.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.viewer.console import protocol as P

#: Procedural, so it costs no map file and builds in about two seconds.
SMOKE_MAP = "gen:competition:2"
READY_TIMEOUT = 300.0                      # a cold CUDA graph capture is 15-20 s; CPU is quicker


class WorkerProc:
    """The console's half: spawn the worker, talk to it, and take it down again.

    Only ever touches the process it started (`self.proc`), which is the same rule the console
    follows -- no pattern matching over the process table.
    """

    def __init__(self, tmp_path):
        import f1sim.viewer.sim_worker as SW
        ctx = multiprocessing.get_context("spawn")
        self.ctl, ctl_theirs = ctx.Pipe(duplex=True)
        self.frames, frames_theirs = ctx.Pipe(duplex=False)      # (readable, writable)
        assert self.frames.readable and frames_theirs.writable
        self.proc = ctx.Process(target=SW.main, args=(ctl_theirs, frames_theirs),
                                kwargs={"log_path": str(tmp_path / "worker.log")},
                                daemon=False, name="f1sim-sim-worker-test")
        self.proc.start()
        self.seq = 0
        self.messages = []

    def send(self, kind, **fields):
        self.seq += 1
        self.ctl.send(P.message(kind, seq=self.seq, **fields))
        return self.seq

    def wait_for(self, kind, timeout=60.0, **match):
        deadline = time.monotonic() + timeout
        idx = 0
        while True:
            while idx < len(self.messages):
                m = self.messages[idx]
                idx += 1
                if m.get("kind") == kind and all(m.get(k) == v for k, v in match.items()):
                    return m
            left = deadline - time.monotonic()
            if left <= 0:
                seen = ", ".join(f"{m.get('kind')}" for m in self.messages[-12:])
                errs = [m for m in self.messages if m.get("kind") == P.MSG_ERROR]
                raise AssertionError(f"{kind} {match} 가 {timeout}s 안에 오지 않았습니다. 최근: {seen}"
                                     + (f"\n첫 오류: {errs[0].get('message')}\n{errs[0].get('detail','')[:2000]}"
                                        if errs else ""))
            if self.ctl.poll(min(0.2, left)):
                self.messages.append(self.ctl.recv())

    def collect_frames(self, seconds):
        out, end = [], time.monotonic() + seconds
        while time.monotonic() < end:
            if self.frames.poll(max(0.0, end - time.monotonic())):
                out.append(self.frames.recv())
        return out

    def close(self):
        try:
            if self.proc.is_alive():
                self.send(P.CMD_SHUTDOWN)
                self.proc.join(timeout=15)
        except (BrokenPipeError, OSError):
            pass
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        for c in (self.ctl, self.frames):
            try:
                c.close()
            except OSError:
                pass


@pytest.fixture(scope="module")
def worker(tmp_path_factory, tmp_legacy_run):
    # No `~/f1sim_runs` dependency: every session below drives the generated legacy checkpoint, so
    # this runs on a machine that has never trained anything.
    w = WorkerProc(tmp_path_factory.mktemp("worker"))
    yield w
    w.close()


@pytest.fixture(scope="module")
def cpu_session(worker, tmp_legacy_run):
    cfg = P.SessionConfig(run=tmp_legacy_run, map_name=SMOKE_MAP, races=1, cars_per_race=1,
                          device="cpu", compile=False)
    worker.send(P.CMD_START, gen=100, config=cfg.to_dict())
    ready = worker.wait_for(P.MSG_READY, timeout=READY_TIMEOUT, gen=100)
    return ready["facts"]


# ==================================================================== handshake / catalogue
def test_hello_reports_this_process(worker):
    seq = worker.send(P.CMD_HELLO)
    hello = worker.wait_for(P.MSG_HELLO, seq=seq)
    assert hello["pid"] == worker.proc.pid, "hello must identify the process the console started"
    assert hello["protocol"] == P.PROTOCOL_VERSION
    assert isinstance(hello["cuda"], bool) and hello["torch"]
    assert os.path.isdir(hello["runs_dir"])
    assert hello["monotonic"] <= time.monotonic(), "the worker's clock must be the same monotonic"


def test_map_catalog_uses_the_group_names_the_console_expects(worker):
    from f1sim.viewer.console.catalog import GROUP_ORDER
    groups = worker.wait_for(P.MSG_MAPS, seq=worker.send(P.CMD_LIST_MAPS))
    assert not groups.get("error")
    assert list(groups["groups"]) == list(GROUP_ORDER)
    assert all(len(v) > 0 for v in groups["groups"].values())


def test_describe_a_missing_run_is_an_answer_not_a_crash(worker):
    seq = worker.send(P.CMD_DESCRIBE, run="definitely_not_a_run_xyz")
    d = worker.wait_for(P.MSG_DESCRIBED, seq=seq)
    assert d["ok"] is False and "런 디렉터리가 없습니다" in d["error"]
    assert worker.proc.is_alive()


def test_describe_the_latest_run_reads_the_real_file(worker):
    """The one check here that genuinely reads the machine's own runs: `describe` answers about the
    real `latest_run()`, whatever its arm, because describing a checkpoint does not load it."""
    from f1sim.learn.watch import latest_run
    if not latest_run():
        pytest.skip("no run under ~/f1sim_runs to describe")
    d = worker.wait_for(P.MSG_DESCRIBED, seq=worker.send(P.CMD_DESCRIBE, run="latest"), timeout=90)
    assert d["ok"] is True
    info = d["info"]
    assert os.path.exists(info["path"]) and info["path"].startswith(latest_run())
    assert info["file"].endswith(".pt") and info["age_s"] >= 0
    assert info["progress"]


# ==================================================================== a real session
def test_an_unknown_map_fails_readably_and_the_worker_survives(worker, tmp_legacy_run):
    cfg = P.SessionConfig(run=tmp_legacy_run, map_name="not_a_map_at_all", device="cpu", compile=False)
    worker.send(P.CMD_START, gen=50, config=cfg.to_dict())
    err = worker.wait_for(P.MSG_ERROR, timeout=120, gen=50)
    assert err["retryable"] is True and err["detail"]
    assert "not_a_map_at_all" in err["message"]
    assert worker.proc.is_alive()


def test_ready_facts_describe_what_was_built(cpu_session):
    f = cpu_session
    assert f["gen"] == 100 and f["map"] == SMOKE_MAP
    assert f["device"] == "cpu" and f["compile"] is False, "compile must be off without CUDA"
    assert f["total_cars"] == 1 and f["car_ids"] == [0]
    assert f["action_mode"] in ("plan", "direct")
    assert f["speed_cap"] > 0 and f["v_max_policy"] >= f["speed_cap"]
    assert f["lidar"]["n_beams"] > 0 and f["lidar"]["range_max"] > 0
    assert f["info_line"], "the console shows this as the checkpoint identity"
    assert os.path.basename(f["checkpoint"]) in ("ppo_latest.pt", "student_latest.pt")


def test_geometry_arrives_ready_to_upload(worker, cpu_session):
    geom = worker.wait_for(P.MSG_GEOMETRY, gen=100)["geometry"]
    assert geom["build_ms"] > 0
    assert len(geom["bounds"]) == 4 and geom["bounds"][2] > geom["bounds"][0]
    for part in ("floor", "ducts", "walls"):
        arrays = geom[part]
        if arrays is None:
            continue
        pos, nrm, col, idx = arrays
        assert pos.dtype == np.float32 and pos.ndim == 2 and pos.shape[1] == 3
        assert nrm.shape == pos.shape and col.shape == (len(pos), 4)
        assert idx.dtype == np.int32 and idx.ndim == 1 and len(idx) % 3 == 0
        assert idx.max() < len(pos), f"{part}: index out of range for its own vertices"


def test_frames_carry_real_finite_state(worker, cpu_session):
    frames = worker.collect_frames(4.0)
    assert len(frames) >= 3, f"only {len(frames)} frames in 4 s"
    for f in frames:
        assert all(k in f for k in ("t", "n", "seq", "x", "y", "yaw")), "console rejects these"
        assert f["gen"] == 100
    seqs = [f["seq"] for f in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    ts = [f["t"] for f in frames]
    assert ts == sorted(ts), "simulation time must not go backwards"

    f = frames[-1]
    n = f["n"]
    assert n == cpu_session["total_cars"]
    for key in ("x", "y", "yaw", "vx", "steer", "roll", "pitch", "lap", "coll", "s", "wall", "len"):
        assert f[key].shape == (n,), f"{key} has shape {f[key].shape}, expected {(n,)}"
        assert np.isfinite(f[key]).all(), f"{key} has non-finite values"
    assert f["rear"].shape == (n, 4)
    assert f["scan"].shape == (cpu_session["lidar"]["n_beams"],)
    assert f["scan_type"].dtype == np.int32 and set(np.unique(f["scan_type"])) <= {0, 1, 2, 3, 4}
    assert f["ids"].dtype == np.int32 and f["ids"].tolist() == list(range(n))
    assert 0 <= f["focus"] < n and f["focus"] == f["focus_env"]
    assert set(f["P"]) == {"mount_x", "mount_y", "mount_z", "mount_yaw", "mount_roll", "mount_pitch"}
    assert len(f["dash"]) == 7 and all(np.isfinite(f["dash"]))
    assert 0.0 < f["mu"] < 3.0, "mu is the simulator's true friction, not an estimate"
    assert f["control_dt"] > 0 and f["worker_dropped"] >= 0
    assert f["info_line"] == cpu_session["info_line"]
    if cpu_session["action_mode"] == "plan":
        assert f["plan"] is not None and f["plan"].shape[1] == 3


def test_sim_rate_is_measured_against_the_wall_clock(worker, cpu_session):
    """`sim_rate` is averaged over a fixed wall-clock window, not a fixed number of steps.

    A step-count window gives every window equal weight however long it took, so slow windows are
    under-counted and the reported rate sits above the one the session is achieving -- measured at
    0.34 reported against 0.26 observed with four cars before this changed. The check below is
    tight enough to catch that bias returning.
    """
    frames = [f for f in worker.collect_frames(4.0) if f["sim_rate"] > 0]
    assert len(frames) >= 5
    sim_s = frames[-1]["t"] - frames[0]["t"]
    wall_s = frames[-1]["created_monotonic"] - frames[0]["created_monotonic"]
    observed = sim_s / max(1e-6, wall_s)
    reported = float(np.median([f["sim_rate"] for f in frames]))
    assert reported == pytest.approx(observed, rel=0.3), \
        f"reported sim_rate {reported:.3f} does not match the observed {observed:.3f}"
    assert reported <= 1.5, "the pacer must never run the simulation faster than real time"


def test_imu_is_averaged_over_however_many_samples_the_step_produced(worker, cpu_session):
    """The IMU sampler is driven by `imu_rate` against the substep clock, so the number of samples
    in a control step varies and is sometimes zero. A frame reports the mean of the samples that
    existed, and says how many there were, rather than indexing a fixed K."""
    frames = worker.collect_frames(3.0)
    with_imu = [f for f in frames if "imu" in f]
    assert with_imu, "the smoke checkpoint's env has the IMU on; no frame carried it"
    counts = {f["imu_samples"] for f in with_imu}
    assert all(c >= 1 for c in counts)
    for f in with_imu:
        assert f["imu"].shape == (6,) and np.isfinite(f["imu"]).all()
    assert all("imu_samples" in f for f in with_imu)


def test_frames_are_snapshots_not_views_of_live_tensors(worker, cpu_session):
    """Two frames apart in time must not share memory: on a CPU device `.cpu().numpy()` hands out a
    view of the tensor's own storage, and a queued frame would then be rewritten before it is
    drawn."""
    frames = worker.collect_frames(2.0)
    assert len(frames) >= 3
    moved = [abs(float(b["x"][0]) - float(a["x"][0])) for a, b in zip(frames, frames[1:])]
    assert max(moved) > 0, "every frame has the same position: they are aliasing one buffer"


def test_pause_ack_means_the_simulation_stopped(worker, cpu_session):
    seq = worker.send(P.CMD_PAUSE, paused=True)
    ack = worker.wait_for(P.MSG_ACK, seq=seq, timeout=30)
    assert ack["state"]["paused"] is True
    last_seq, t_at_ack = ack["state"]["last_seq"], ack["state"]["t"]

    after = worker.collect_frames(1.0)
    fresh = [f for f in after if f["seq"] > last_seq]
    assert not fresh, f"frames produced after the pause ack: {[f['seq'] for f in fresh]}"
    assert all(f["t"] <= t_at_ack + 1e-9 for f in after)

    ack = worker.wait_for(P.MSG_ACK, seq=worker.send(P.CMD_PAUSE, paused=False), timeout=30)
    assert ack["state"]["paused"] is False
    resumed = worker.collect_frames(2.0)
    assert resumed and resumed[-1]["t"] > t_at_ack, "resume did not advance simulation time"


def test_reset_restarts_the_episode_without_rewinding_the_clock(worker, cpu_session):
    before = worker.collect_frames(1.0)
    assert before
    ack = worker.wait_for(P.MSG_ACK, seq=worker.send(P.CMD_RESET), timeout=30)
    assert ack["command"] == "reset"
    assert ack["state"]["t"] >= before[-1]["t"]
    after = worker.collect_frames(2.0)
    assert after and after[-1]["seq"] > ack["state"]["last_seq"]
    assert after[-1]["t"] >= ack["state"]["t"]


def test_a_second_generation_replaces_the_first(worker, cpu_session, tmp_legacy_run):
    """The guarantee is about *production*, not arrival.

    Frames the sender already wrote into the pipe cannot be unsent, so a short tail of the previous
    generation can still be read after `ready` -- which is why every frame carries `gen` and the
    console discards the ones it is no longer showing. What must be true is that the worker stopped
    *producing* the old generation before it announced the new one, and that is what
    `created_monotonic` against the ready message's own timestamp checks.
    """
    cfg = P.SessionConfig(run=tmp_legacy_run, map_name=SMOKE_MAP, races=1, cars_per_race=1,
                          device="cpu", compile=False, saliency=True)
    worker.send(P.CMD_START, gen=101, config=cfg.to_dict())
    ready = worker.wait_for(P.MSG_READY, timeout=READY_TIMEOUT, gen=101)
    facts = ready["facts"]
    assert facts["gen"] == 101
    frames = worker.collect_frames(2.0)
    assert frames

    stale = [f for f in frames if f["gen"] != 101]
    assert all(f["created_monotonic"] <= ready["monotonic"] for f in stale), \
        "the old generation was still being simulated after the new one was announced"
    assert [f["gen"] for f in frames] == sorted(f["gen"] for f in frames), \
        "generations interleaved: the swap is not a clean cut"
    new = [f for f in frames if f["gen"] == 101]
    assert new, "the new generation produced no frames"
    assert new[0]["seq"] == 0, "sequence restarts with the generation"
    sal = [f for f in new if "saliency" in f]
    assert sal, "saliency was requested in the config and never arrived"
    assert sal[-1]["saliency"].shape == (facts["lidar"]["n_beams"],)
    assert sal[-1]["saliency_age"] >= 0


def test_stop_then_start_again(worker, tmp_legacy_run):
    worker.wait_for(P.MSG_ACK, seq=worker.send(P.CMD_STOP), timeout=30)
    worker.wait_for(P.MSG_STOPPED, timeout=10)
    assert worker.proc.is_alive()
    cfg = P.SessionConfig(run=tmp_legacy_run, map_name=SMOKE_MAP, device="cpu", compile=False)
    worker.send(P.CMD_START, gen=102, config=cfg.to_dict())
    assert worker.wait_for(P.MSG_READY, timeout=READY_TIMEOUT, gen=102)["facts"]["gen"] == 102


def test_shutdown_leaves_no_process_behind(tmp_path):
    w = WorkerProc(tmp_path)
    try:
        w.wait_for(P.MSG_HELLO, seq=w.send(P.CMD_HELLO), timeout=90)
        pid = w.proc.pid
        w.send(P.CMD_SHUTDOWN)
        w.wait_for(P.MSG_BYE, timeout=20)
        w.proc.join(timeout=8)
        assert not w.proc.is_alive(), f"worker {pid} outlived its shutdown"
        assert w.proc.exitcode == 0
    finally:
        w.close()


@pytest.mark.skipif(not os.environ.get("F1SIM_WORKER_CUDA"),
                    reason="CUDA smoke runs only under a GPU lease (set F1SIM_WORKER_CUDA=1)")
def test_cuda_session_reports_the_device_it_actually_used(tmp_path, tmp_legacy_run):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    w = WorkerProc(tmp_path)
    try:
        cfg = P.SessionConfig(run=tmp_legacy_run, map_name=SMOKE_MAP, races=1, cars_per_race=1,
                             device="cuda", compile=False)
        w.send(P.CMD_START, gen=1, config=cfg.to_dict())
        facts = w.wait_for(P.MSG_READY, timeout=READY_TIMEOUT, gen=1)["facts"]
        assert facts["device"].startswith("cuda")
        frames = w.collect_frames(4.0)
        assert len(frames) >= 5
        assert np.isfinite(frames[-1]["x"]).all() and np.isfinite(frames[-1]["scan"]).all()
        assert frames[-1]["sim_rate"] > 0
    finally:
        w.close()
