"""`viewer/recorder.py`: the one encoder the console and the headless CLI share.

Recording used to be three copies of a `subprocess.Popen(["ffmpeg", ...])` inside `learn.watch`,
reachable only headless. It is one module now, so the claims worth pinning are:

1. **The CPU command did not change.** Every clip in this repository was made with that exact argv,
   and a re-encode that differs is a clip nobody can compare against the old ones. It is frozen here
   as a literal rather than rebuilt from the function under test.
2. **The GPU encoder is chosen by trying it, not by asking.** `ffmpeg -encoders` lists `h264_nvenc`
   on machines with no device; the choice has to be a measurement, it has to be overridable, and a
   named encoder must never cause a probe.
3. **Submitting never blocks.** The console calls `submit` from the thread that paints the window.
   A recorder that can make the window wait is worse than no recorder, so a full queue drops and
   counts.
4. **What comes out is a playable file** of the size, rate and length it says it is.
"""
import json
import os
import subprocess
import time

import pytest

from f1sim.viewer import recorder as R

#: What `learn.watch` ran before this module existed, written out. Not built from `ffmpeg_argv`:
#: the point is to catch a change to that function, and a test that derives its expectation from the
#: thing it is testing catches nothing.
FROZEN_LIBX264 = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                  "-s", "1600x900", "-r", "30", "-i", "-", "-c:v", "libx264",
                  "-pix_fmt", "yuv420p", "-crf", "20", "/tmp/out.mp4"]

needs_ffmpeg = pytest.mark.skipif(not R.ffmpeg_available(), reason="no ffmpeg on this machine")


# ================================================================ 1. the command
def test_the_cpu_command_is_the_one_every_existing_clip_was_made_with():
    assert R.ffmpeg_argv("/tmp/out.mp4", 1600, 900, 30) == FROZEN_LIBX264
    assert R.ffmpeg_argv("/tmp/out.mp4", 1600, 900, 30, encoder="libx264") == FROZEN_LIBX264


def test_the_gpu_command_is_the_same_pipeline_with_the_card_compressing():
    argv = R.ffmpeg_argv("/tmp/out.mp4", 1280, 720, 30, encoder="h264_nvenc")
    assert argv[:argv.index("-c:v")] == R.ffmpeg_argv("/tmp/out.mp4", 1280, 720, 30)[:argv.index("-c:v")]
    assert argv[argv.index("-c:v") + 1] == "h264_nvenc"
    # nvenc has no -crf; the constant-quality knob is -cq at the same number
    assert "-crf" not in argv and argv[argv.index("-cq") + 1] == "20"
    assert argv[-1] == "/tmp/out.mp4" and "yuv420p" in argv


def test_the_headless_cli_builds_its_encoder_through_this_module():
    """`learn.watch` must not grow a second copy of the command."""
    import inspect

    from f1sim.learn import watch
    src = inspect.getsource(watch)
    assert "_recorder.FfmpegEncoder(" in src
    # No second copy of the *command*: the flags below are what an ffmpeg invocation is made of, and
    # `--video-encoder`'s help text naming libx264 is documentation, not a command.
    for piece in ('"ffmpeg"', "'ffmpeg'", '"-c:v"', '"rawvideo"', '"-pix_fmt"', '"-crf"'):
        assert piece not in src, f"watch.py spells the ffmpeg command itself again ({piece})"


# ================================================================ 2. which encoder
def test_a_named_encoder_is_taken_at_its_word_and_probes_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(R, "nvenc_works", lambda *a, **k: called.append(1) or True)
    monkeypatch.delenv(R.ENCODER_ENV, raising=False)
    assert R.resolve_encoder("libx264") == "libx264"
    assert R.resolve_encoder("h264_nvenc") == "h264_nvenc"
    assert not called, "naming an encoder must not touch the GPU"


def test_auto_asks_and_falls_back(monkeypatch):
    monkeypatch.delenv(R.ENCODER_ENV, raising=False)
    monkeypatch.setattr(R, "nvenc_works", lambda *a, **k: True)
    assert R.resolve_encoder("auto") == "h264_nvenc"
    monkeypatch.setattr(R, "nvenc_works", lambda *a, **k: False)
    assert R.resolve_encoder("auto") == "libx264"


def test_the_environment_pins_the_choice(monkeypatch):
    """How a machine, or a test suite that must not touch a shared card, says which one to use."""
    monkeypatch.setattr(R, "nvenc_works", lambda *a, **k: True)
    monkeypatch.setenv(R.ENCODER_ENV, "libx264")
    assert R.resolve_encoder("auto") == "libx264"
    assert R.resolve_encoder("h264_nvenc") == "libx264", "the environment wins over the argument"


def test_an_unknown_name_falls_back_to_auto_rather_than_to_a_broken_command(monkeypatch):
    monkeypatch.delenv(R.ENCODER_ENV, raising=False)
    monkeypatch.setattr(R, "nvenc_works", lambda *a, **k: False)
    assert R.resolve_encoder("h265_magic") == "libx264"


def test_the_probe_needs_a_card_before_it_spends_anything(monkeypatch):
    monkeypatch.setattr(R, "_gpu_present", lambda: False)
    monkeypatch.setattr(R, "_NVENC_OK", None, raising=False)
    ran = []
    monkeypatch.setattr(R.subprocess, "Popen", lambda *a, **k: ran.append(1))
    assert R.nvenc_works(refresh=True) is False
    assert not ran, "the probe ran ffmpeg on a machine with no GPU"


def test_the_probe_frame_is_big_enough_for_nvenc_to_accept():
    """NVENC refuses a frame below its minimum dimension, so a tiny probe reports a working card as
    broken. Measured: 64x64 fails with 'Frame Dimension less than the minimum supported value'."""
    w, h = R.PROBE_SIZE
    assert w >= 256 and h >= 144 and w % 2 == 0 and h % 2 == 0


# ================================================================ 3. the spec
@pytest.mark.parametrize("spec,window,want", [
    (R.RecordSpec(width=1920, height=1080), (800, 600), (1920, 1080)),
    (R.RecordSpec(width=0, height=0), (1601, 901), (1600, 900)),      # the window, made even
    (R.RecordSpec(width=1281, height=721), (800, 600), (1280, 720)),
])
def test_the_spec_resolves_to_something_the_encoder_accepts(spec, window, want):
    """yuv420p subsamples by two, so an odd side is refused -- and a window is whatever size it was
    dragged to."""
    r = spec.resolved(window)
    assert (r.width, r.height) == want


def test_the_default_path_names_the_map_and_the_time():
    p = R.default_video_path("real/bb22-1@rev#line:44")
    assert p.startswith(R.VIDEO_DIR) and p.endswith(".mp4")
    assert "real-bb22-1-rev-line-44_" in os.path.basename(p)
    assert "/" not in os.path.basename(p) and "#" not in os.path.basename(p)


# ================================================================ 4. the recorder
def _frame(w, h):
    return bytes(bytearray((i * 7) % 256 for i in range(w * h * 3)))


@needs_ffmpeg
def test_a_recording_is_a_playable_file_of_the_size_and_length_it_claims(tmp_path, monkeypatch):
    monkeypatch.setenv(R.ENCODER_ENV, "libx264")
    spec = R.RecordSpec(path=str(tmp_path / "clip.mp4"), width=320, height=180, fps=30)
    rec = R.VideoRecorder(spec)
    assert rec.ok
    for _ in range(60):
        rec.submit(_frame(320, 180))
    rec.stop()
    assert rec.error is None and rec.dropped == 0 and rec.written == 60
    assert rec.duration == pytest.approx(2.0, abs=1e-6)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                          spec.path], capture_output=True, text=True, timeout=60)
    j = json.loads(out.stdout)
    st = next(s for s in j["streams"] if s.get("codec_type") == "video")
    assert (st["width"], st["height"]) == (320, 180)
    assert st["avg_frame_rate"] == "30/1" and st["codec_name"] == "h264"
    assert abs(float(j["format"]["duration"]) - 2.0) <= 0.2
    assert "libx264" in rec.summary() and "320×180" in rec.summary()


@needs_ffmpeg
def test_submitting_never_blocks_and_drops_are_counted(tmp_path, monkeypatch):
    """The contract with the thread that paints the window. A queue of two and four hundred frames
    is the encoder losing by a mile; what must not happen is a wait."""
    monkeypatch.setenv(R.ENCODER_ENV, "libx264")
    spec = R.RecordSpec(path=str(tmp_path / "drop.mp4"), width=320, height=180, fps=30)
    rec = R.VideoRecorder(spec, queue_frames=2)
    frame = _frame(320, 180)
    t0 = time.perf_counter()
    for _ in range(400):
        rec.submit(frame)
    elapsed = time.perf_counter() - t0
    rec.stop()
    assert elapsed < 1.0, f"400 submits took {elapsed:.2f}s -- the render thread was made to wait"
    assert rec.submitted == 400
    assert rec.dropped > 300, rec.dropped
    assert rec.written + rec.dropped == rec.submitted
    assert f"버린 프레임 {rec.dropped}" in rec.summary()


@needs_ffmpeg
def test_a_finished_clip_says_which_encoder_wrote_it(tmp_path, monkeypatch):
    monkeypatch.setenv(R.ENCODER_ENV, "libx264")
    spec = R.RecordSpec(path=str(tmp_path / "named.mp4"), width=320, height=180, fps=30,
                        encoder="auto")
    rec = R.VideoRecorder(spec)
    for _ in range(4):
        rec.submit(_frame(320, 180))
    rec.stop()
    assert rec.codec == "libx264" and rec.codec in rec.summary()


def test_no_ffmpeg_is_an_answer_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(R.subprocess, "Popen", _raise_missing)
    rec = R.VideoRecorder(R.RecordSpec(path=str(tmp_path / "x.mp4"), width=320, height=180, fps=30,
                                       encoder="libx264"))
    assert not rec.ok and "ffmpeg" in (rec.error or "")
    assert rec.submit(_frame(320, 180)) is False        # and submitting is still safe
    rec.stop()
    assert "녹화 실패" in rec.summary()


def _raise_missing(*_a, **_k):
    raise FileNotFoundError("ffmpeg")


@needs_ffmpeg
def test_a_gpu_encoder_that_dies_before_the_first_frame_falls_back(tmp_path, monkeypatch):
    """A card can pass the probe and still refuse the real stream -- another process takes the last
    session slot in between. Nothing has been written yet at that point, so the clip is saved by
    switching to the CPU rather than lost."""
    monkeypatch.delenv(R.ENCODER_ENV, raising=False)
    monkeypatch.setattr(R, "nvenc_works", lambda *a, **k: True)
    real = R.ffmpeg_argv

    def broken(path, w, h, fps, crf=20, encoder="libx264"):
        argv = real(path, w, h, fps, crf)                # the working CPU command ...
        if encoder == "h264_nvenc":                      # ... with an encoder ffmpeg will reject
            argv[argv.index("libx264")] = "nope_nvenc"
        return argv

    monkeypatch.setattr(R, "ffmpeg_argv", broken)
    enc = R.FfmpegEncoder(str(tmp_path / "fb.mp4"), 320, 180, 30, encoder="auto")
    assert enc.encoder == "h264_nvenc"
    for _ in range(4):
        enc.write(_frame(320, 180))
    enc.close()
    assert enc.encoder == "libx264", "the fallback did not happen"
    assert enc.frames >= 1 and os.path.getsize(tmp_path / "fb.mp4") > 0
