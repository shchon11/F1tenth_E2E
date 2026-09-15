"""Writing what the viewer is drawing to an mp4 -- one implementation, two callers.

Recording existed in exactly one place: `python -m f1sim.learn.watch --record out.mp4`, three copies
of the same `subprocess.Popen(["ffmpeg", ...])` inside `watch.main`, reachable only headless. The
user asked for it in the console instead -- *"비디오 렌더러 스크립트같은거 있는것 같은데 이거 좀
비쥬얼라이져 기본 기능에 넣어놔라"* -- and two encoders would be two answers to "what did this
produce", so the headless path moved here rather than being copied.

Two things use this module and they need different shapes:

* **Headless** (`watch.py`) renders one frame, writes it, renders the next. Nothing is concurrent,
  nothing can fall behind, and blocking on ffmpeg is correct -- the loop has nowhere else to be.
  `FfmpegEncoder` is that: a pipe with the argv the CLI has always used.
* **The console** renders on the GUI thread, at whatever rate the display and the keepalive produce.
  Blocking that thread on a pipe would stall the window and, through it, the frame the user is
  watching. `VideoRecorder` therefore hands frames to an encoder thread through a bounded queue and
  **drops** when the queue is full, counting every drop. A recording with a number of dropped frames
  beside it is a recording you can trust; one that silently stalled the UI is not.

The argv is deliberately one function. `crf 20`, `libx264`, `yuv420p`, `-loglevel error` and the
`rawvideo` input are what every clip in this repository was made with, and a re-encode that differs
is a clip nobody can compare against the old ones.

**Which encoder.** `auto` prefers `h264_nvenc` where the card can actually do it and falls back to
`libx264`, and the finished clip says which one ran. "Can actually do it" is decided by *encoding two
frames with the real argv*, not by grepping `ffmpeg -encoders`: nvenc is listed on machines with no
device, with a driver the runtime does not match, and with every session slot already taken, and each
of those fails at the first frame rather than at startup. The probe is lazy, cached for the process,
and never runs when the caller names an encoder -- so nothing touches the GPU unless a recording
actually starts on `auto`. Rendering is unaffected either way: the frames come from whatever GL
context the session already has, which is the GPU on a desktop and llvmpipe under Xvfb.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

#: Where the console puts a clip when nobody says otherwise.
VIDEO_DIR = os.path.join(os.path.expanduser("~"), "f1sim_videos")

#: The resolutions the console offers. `0` height means "whatever the window is", which is the only
#: one that depends on how the window happens to be sized.
RESOLUTIONS: Tuple[Tuple[str, int, int], ...] = (
    ("720p (1280×720)", 1280, 720),
    ("1080p (1920×1080)", 1920, 1080),
    ("현재 창 크기", 0, 0),
)

#: Frame rates offered. 30 is the default the headless recorder has always used.
FPS_CHOICES: Tuple[int, ...] = (24, 30, 60)

#: How many rendered frames may wait for the encoder before one is dropped. Two seconds at 30 fps:
#: enough to ride out a garbage collection or a slow x264 block, short enough that a recording which
#: is genuinely falling behind says so within a couple of seconds instead of growing a backlog that
#: turns into memory.
QUEUE_FRAMES = 60


#: The video encoders a recording may ask for. `auto` is "the GPU one if it works here".
ENCODERS: Tuple[str, ...] = ("auto", "h264_nvenc", "libx264")
ENCODER_LABEL = {"auto": "자동 (GPU 우선)", "h264_nvenc": "h264_nvenc (GPU)", "libx264": "libx264 (CPU)"}

#: Overridden by `$F1SIM_VIDEO_ENCODER` (`auto` / `h264_nvenc` / `libx264`), which is how a machine
#: or a test pins the choice without touching the card.
ENCODER_ENV = "F1SIM_VIDEO_ENCODER"

_NVENC_OK: Optional[bool] = None

#: Frame size the nvenc probe encodes. NVENC has a minimum frame dimension and refuses anything
#: under it, so the probe cannot be arbitrarily small without reporting a working card as broken.
PROBE_SIZE: Tuple[int, int] = (256, 144)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffmpeg_argv(path: str, width: int, height: int, fps: int, crf: int = 20,
                encoder: str = "libx264") -> List[str]:
    """The encoder command. One function so the console and the CLI cannot drift apart.

    The `libx264` form is byte for byte what `learn.watch` ran before this module existed; changing
    it re-encodes every future clip differently from every past one. `h264_nvenc` is the same
    pipeline with the card doing the compression: it has no `-crf`, so the constant-quality knob is
    `-cq` at the same number, under `vbr` rate control.
    """
    head = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{int(width)}x{int(height)}", "-r", str(int(fps)), "-i", "-"]
    if encoder == "h264_nvenc":
        return head + ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                       "-cq", str(int(crf)), "-pix_fmt", "yuv420p", str(path)]
    return head + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(int(crf)), str(path)]


def _gpu_present() -> bool:
    """Is there a card at all? Asked before anything is spent probing the encoder on it."""
    if any(os.path.exists(f"/dev/nvidia{i}") for i in range(4)):
        return True
    return shutil.which("nvidia-smi") is not None and _run_quiet(["nvidia-smi", "-L"]) == 0


def _run_quiet(argv: List[str], timeout: float = 10.0) -> int:
    try:
        return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=timeout).returncode
    except (OSError, subprocess.SubprocessError):
        return -1


def nvenc_works(refresh: bool = False) -> bool:
    """Whether this machine can actually encode with `h264_nvenc`, measured once per process.

    Measured by encoding two frames with the real argv and throwing the file away. `ffmpeg
    -encoders` lists nvenc on machines with no device, with a mismatched driver, and with every
    session slot taken; all three fail at the first frame, which is exactly when a recording must not
    discover them.

    The probe frame is `PROBE_SIZE` and not something tiny: NVENC refuses a frame below its minimum
    dimension outright ("Frame Dimension less than the minimum supported value"), so a 64x64 probe
    reports a working card as broken. It is the smallest 16:9 size comfortably above that bound.
    """
    global _NVENC_OK
    if _NVENC_OK is not None and not refresh:
        return _NVENC_OK
    _NVENC_OK = False
    if not ffmpeg_available() or not _gpu_present():
        return _NVENC_OK
    import tempfile
    with tempfile.TemporaryDirectory(prefix="f1sim_nvenc_") as d:
        out = os.path.join(d, "probe.mp4")
        w, h = PROBE_SIZE
        argv = ffmpeg_argv(out, w, h, 30, encoder="h264_nvenc")
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            proc.communicate(b"\x00" * (w * h * 3) * 2, timeout=30.0)
            _NVENC_OK = proc.returncode == 0 and os.path.getsize(out) > 0
        except (OSError, subprocess.SubprocessError):
            _NVENC_OK = False
    return _NVENC_OK


def resolve_encoder(want: str = "auto") -> str:
    """`auto` -> the encoder that will actually run here. A named one is taken at its word."""
    want = str(os.environ.get(ENCODER_ENV) or want or "auto").strip()
    if want not in ENCODERS:
        want = "auto"
    if want != "auto":
        return want
    return "h264_nvenc" if nvenc_works() else "libx264"


class FfmpegEncoder:
    """A raw-RGB pipe into ffmpeg. Blocking, by design -- see the module docstring.

    `write` swallows a broken pipe rather than raising: ffmpeg dying half way through a long
    headless render should cost the clip, not the run that was producing it. `error` says what
    happened, and `close` returns the exit status so a caller can report it.
    """

    def __init__(self, path: str, width: int, height: int, fps: int, crf: int = 20,
                 encoder: str = "libx264"):
        self.path, self.width, self.height, self.fps = str(path), int(width), int(height), int(fps)
        self.crf = int(crf)
        self.frames = 0
        self.error: Optional[str] = None
        #: The encoder that is actually running, which is what a finished clip is labelled with.
        self.encoder = resolve_encoder(encoder)
        #: True while the GPU encoder was *chosen for us*. A card can pass the probe and still
        #: refuse the real stream -- another process takes the last session slot between the two --
        #: so a failure before the first frame falls back to the CPU rather than losing the clip.
        self._may_fall_back = self.encoder == "h264_nvenc" and str(encoder) == "auto"
        self.proc = None
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._open()

    def _open(self) -> None:
        try:
            self.proc = subprocess.Popen(
                ffmpeg_argv(self.path, self.width, self.height, self.fps, self.crf, self.encoder),
                stdin=subprocess.PIPE)
        except FileNotFoundError:
            self.proc = None
            self.error = ("ffmpeg 을 찾을 수 없습니다. 녹화에는 ffmpeg 이 필요합니다 "
                          "(apt install ffmpeg).")

    @property
    def ok(self) -> bool:
        return self.proc is not None and self.error is None

    def write(self, rgb: bytes) -> bool:
        if self.proc is None or self.proc.stdin is None:
            return False
        try:
            self.proc.stdin.write(rgb)
            self.frames += 1
            return True
        except (BrokenPipeError, ValueError, OSError) as exc:
            if self._may_fall_back and not self._produced_anything():
                # The GPU encoder was picked for us and died without producing a file. The frames
                # counted so far went into a pipe nobody read, so nothing is lost by switching to
                # the CPU encoder and carrying on -- and the alternative is handing back an empty
                # mp4 on a machine whose card passed the probe a moment earlier.
                #
                # `frames == 0` is not the test: a write into the OS pipe buffer succeeds for a
                # while after ffmpeg has already exited, so the count says nothing about what was
                # encoded. The file does.
                self._may_fall_back = False
                try:
                    if self.proc.stdin is not None:
                        self.proc.stdin.close()
                    self.proc.wait(timeout=10)
                except (OSError, subprocess.SubprocessError):
                    pass
                self.encoder = "libx264"
                self.frames = 0
                self._open()
                return self.write(rgb)
            if self.error is None:
                self.error = f"ffmpeg 파이프가 끊겼습니다: {exc}"
            return False

    def _produced_anything(self) -> bool:
        try:
            return os.path.getsize(self.path) > 0
        except OSError:
            return False

    def close(self) -> int:
        """Flush, wait, and return ffmpeg's exit code (-1 when it never started)."""
        if self.proc is None:
            return -1
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        code = self.proc.wait()
        if code != 0 and self.error is None:
            self.error = f"ffmpeg 이 코드 {code} 로 끝났습니다."
        self.proc = None
        return code


@dataclass(frozen=True)
class RecordSpec:
    """What a console recording is: where it goes, how big, how fast, and what is in it.

    Carried on `SessionConfig` so a session remembers its recording settings, and excluded from the
    comparison that decides whether a change needs a rebuild -- changing the frame rate is not a
    different simulation.
    """
    path: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 30
    #: `""` = whatever the viewport is showing. Otherwise a camera key the viewport understands, so
    #: a clip can be filmed from the chase camera while the window stays on the overview.
    camera: str = ""
    #: LiDAR dots, the raceline, car labels. Off makes a clean clip of the driving itself.
    overlays: bool = True
    #: Stop after this many seconds. 0 = run until stopped.
    seconds: float = 0.0
    #: `auto` (the GPU encoder where it works), or one named outright. Rendering is unaffected --
    #: the frames come from the GL context the session already has.
    encoder: str = "auto"

    def resolved(self, window: Tuple[int, int]) -> "RecordSpec":
        """This spec with `현재 창 크기` turned into the numbers, and both sides made even.

        x264's `yuv420p` subsamples by two in each direction, so an odd width or height is refused
        by the encoder -- and a window is whatever size the user dragged it to.
        """
        w = int(self.width) or int(window[0])
        h = int(self.height) or int(window[1])
        return replace(self, width=max(2, w - (w % 2)), height=max(2, h - (h % 2)))


def default_video_path(scenario: str = "", when: Optional[float] = None) -> str:
    """`~/f1sim_videos/<map>_<time>.mp4`, with the map's punctuation made filename-safe."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(when if when is not None else time.time()))
    name = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(scenario or "f1sim"))
    name = "-".join(p for p in name.split("-") if p) or "f1sim"
    return os.path.join(VIDEO_DIR, f"{name}_{stamp}.mp4")


def describe_size(path: str) -> str:
    try:
        n = os.path.getsize(path)
    except OSError:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


class VideoRecorder:
    """A recording in progress: frames in from the render thread, an mp4 out.

    `submit` never blocks and never raises. That is the whole contract: it is called from the thread
    that paints the window, and a recorder that can make the window stutter is worse than no
    recorder. When the encoder cannot keep up the frame is dropped and counted, and the count is
    shown next to the clip -- `dropped` is not an error, it is the honest description of a clip whose
    encoder was slower than its renderer.
    """

    def __init__(self, spec: RecordSpec, queue_frames: int = QUEUE_FRAMES):
        self.spec = spec
        self.q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=max(1, int(queue_frames)))
        self.encoder = FfmpegEncoder(spec.path, spec.width, spec.height, spec.fps,
                                     encoder=spec.encoder)
        self.dropped = 0
        self.submitted = 0
        self.started = time.time()
        self.stopped: Optional[float] = None
        self.error: Optional[str] = self.encoder.error
        self._thread: Optional[threading.Thread] = None
        if self.encoder.ok:
            self._thread = threading.Thread(target=self._run, name="f1sim-encoder", daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- state
    @property
    def ok(self) -> bool:
        return self._thread is not None and self.error is None

    @property
    def written(self) -> int:
        return self.encoder.frames

    @property
    def elapsed(self) -> float:
        return (self.stopped or time.time()) - self.started

    @property
    def duration(self) -> float:
        """Seconds of *video*, which is the frames that were encoded -- not the wall clock.

        A clip whose renderer produced 20 frames in a second at 30 fps is 0.67 s long and shows a
        second of driving in it. Reporting the wall clock instead would call it a 1 s clip that
        plays too fast, which is the same mistake as not counting the drops.
        """
        return self.written / max(1, self.spec.fps)

    @property
    def due(self) -> float:
        """Seconds between frames at the configured rate."""
        return 1.0 / max(1, self.spec.fps)

    # ---------------------------------------------------------------- the two ends
    def submit(self, rgb: Optional[bytes]) -> bool:
        """Offer one frame. False when it was dropped (or the recording is already over)."""
        if not self.ok or rgb is None:
            return False
        self.submitted += 1
        try:
            self.q.put_nowait(rgb)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            if not self.encoder.write(item):
                self.error = self.encoder.error
                break

    def stop(self) -> "VideoRecorder":
        """Drain what is queued, close the pipe, wait for ffmpeg. Idempotent."""
        if self.stopped is not None:
            return self
        self.stopped = time.time()
        if self._thread is not None:
            try:
                self.q.put(None, timeout=5.0)
            except queue.Full:
                pass
            self._thread.join(timeout=30.0)
            self._thread = None
        self.encoder.close()
        if self.error is None:
            self.error = self.encoder.error
        return self

    def summary(self) -> str:
        """One line for the console: what was written, how long it is, and what it cost."""
        if self.error:
            return f"녹화 실패: {self.error}"
        drops = f" · 버린 프레임 {self.dropped}" if self.dropped else ""
        return (f"{os.path.basename(self.spec.path)} · {self.duration:.1f}초 · "
                f"{self.spec.width}×{self.spec.height} {self.spec.fps}fps · "
                f"{self.codec} · {describe_size(self.spec.path)}{drops}")

    @property
    def codec(self) -> str:
        """The encoder that actually ran -- not the one that was asked for."""
        return self.encoder.encoder


def scene_rgb(scene, width: Optional[int] = None, height: Optional[int] = None) -> bytes:
    """One frame of a headless `gl_scene.Scene` as bottom-up-corrected RGB bytes.

    The multisample resolve and the vertical flip are the two steps every caller needs and every
    caller used to write out again; GL's origin is bottom-left and every encoder and image format
    here wants top-left.
    """
    from PIL import Image
    w = int(width or scene.width)
    h = int(height or scene.height)
    if getattr(scene, "fbo_ms", None) is not None:
        scene.ctx.copy_framebuffer(scene.fbo, scene.fbo_ms)
    data = scene.fbo.read(components=3)
    return Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM).tobytes()
