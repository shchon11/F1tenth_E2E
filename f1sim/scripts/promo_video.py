"""Promo clip for the README: headless (EGL) renders of the simulator, a training swarm, the
policy's attention, and a race with opponents -> docs/promo.mp4 + docs/promo.gif.

    python3 scripts/promo_video.py            # ~2 min on the laptop GPU, runs next to training

Scenes are scripted cameras on the native viewer; captions are composited on the captured
frames. Never opens a window."""
import argparse, math, os, subprocess, sys, time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from f1sim import maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.learn.model import load_checkpoint
from f1sim.learn.obs import flatten_obs
from f1sim.learn.watch import Introspector, panel_image, sal_colors
from f1sim.viewer.native import NativeViewer

W, H, FPS = 1280, 720, 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs")


class PromoViewer(NativeViewer):
    cam_fn = None

    def _camera(self, fr):
        if self.cam_fn is None:
            return super()._camera(fr)
        eye, tgt, up = self.cam_fn(fr)
        return np.asarray(eye, float), np.asarray(tgt, float), np.asarray(up, float)


def grab(v) -> Image.Image:
    if v.scene.fbo_ms is not None:
        v.scene.ctx.copy_framebuffer(v.scene.fbo, v.scene.fbo_ms)
    return Image.frombytes("RGB", (v.width, v.height), v.scene.fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)


def caption(img: Image.Image, text: str, sub: str = ""):
    d = ImageDraw.Draw(img, "RGBA")
    f1 = ImageFont.truetype(FONT_B, 30); f2 = ImageFont.truetype(FONT, 22)
    d.rectangle([0, H - 118, W, H], fill=(0, 0, 0, 150))
    d.text((40, H - 104), text, font=f1, fill=(255, 255, 255, 255))
    if sub:
        d.text((40, H - 58), sub, font=f2, fill=(200, 210, 220, 255))
    d.text((W - 150, 24), "f1sim", font=ImageFont.truetype(FONT_B, 26), fill=(255, 255, 255, 220))
    return img


def card(lines, sizes, color=(12, 14, 20)):
    img = Image.new("RGB", (W, H), color); d = ImageDraw.Draw(img)
    y = H // 2 - sum(s + 18 for s in sizes) // 2
    for ln, sz in zip(lines, sizes):
        f = ImageFont.truetype(FONT_B if sz >= 40 else FONT, sz)
        w = d.textlength(ln, font=f); d.text(((W - w) / 2, y), ln, font=f, fill=(240, 240, 245)); y += sz + 18
    return img


class Sink:
    def __init__(self, path):
        self.proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                      "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "slow", path], stdin=subprocess.PIPE)
        self.n = 0

    def put(self, img: Image.Image, fade: float = 1.0):
        a = np.asarray(img, np.float32)
        if fade < 1.0:
            a = a * fade
        self.proc.stdin.write(a.astype(np.uint8).tobytes()); self.n += 1

    def hold(self, img, seconds, fade_s=0.5):
        n = int(seconds * FPS); nf = int(fade_s * FPS)
        for i in range(n):
            f = min(1.0, (i + 1) / nf, (n - i) / nf) if nf else 1.0
            self.put(img, f)

    def close(self):
        self.proc.stdin.close(); self.proc.wait()


def scene(sink, env, act_fn, seconds, cam_fn, text, sub="", focus=0, panel_fn=None, colors_fn=None, fade_s=0.5, cars=None):
    env.sim.warmup()
    v = PromoViewer(env.sim, headless=True, max_cars=cars or env.B, width=W, height=H)
    v.scene.set_hud([]); orig = v.scene.set_hud; v.scene.set_hud = lambda lines: orig([])      # no debug HUD
    v.focus = focus; v.cam_fn = cam_fn; v.show_trails = False
    n = int(seconds * FPS); nf = int(fade_s * FPS)
    t0 = time.time()
    for i in range(n):
        act_fn()
        v.update(env.last_result)
        if colors_fn is not None: v.point_colors = colors_fn()
        if panel_fn is not None: v.panel = panel_fn()
        v.frame_t = i / FPS
        v.render()
        img = caption(grab(v), text, sub)
        sink.put(img, min(1.0, (i + 1) / nf, (n - i) / nf))
    print(f"  scene '{text[:40]}' {n} frames in {time.time() - t0:.0f}s", flush=True)
    v.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DOCS, "promo.mp4"))
    ap.add_argument("--policy", default="", help="checkpoint for the swarm/chase/race scenes (default: newest ppo_v4, else ppo_v3 final)")
    ap.add_argument("--swarm", type=int, default=256)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = a.policy or next((p for p in (os.path.join(common.RUNS_DIR, "ppo_v4", "ppo_latest.pt"),) if os.path.exists(p)),
                          os.path.join(common.RUNS_DIR, "ppo_v3", "ppo_final.pt"))
    ck_fast = os.path.join(common.RUNS_DIR, "ppo_v3", "ppo_final.pt")            # the quick one on procedural tracks
    print("policy:", ck, flush=True)
    os.makedirs(DOCS, exist_ok=True)
    sink = Sink(a.out)

    # ---- title
    sink.hold(card(["f1sim", "a realistic F1TENTH simulator", "for LiDAR-only end-to-end racing"], [72, 30, 30]), 2.5)

    # ---- 1. hero: teacher lap on the Korea championship map, orbiting camera, 3D LiDAR
    tracks, rls = common.load_tracks(["real:korea_2025_iccas"], racelines=True)
    env = common.make_env(tracks, 1, dev, EnvConfig(speed_cap=3.5, resample_track_on_reset=False), seed=7)
    teacher = common.make_teacher(rls, env); env.reset(seed=7)
    rl = rls[0]; t0_ = tracks[0]                                                # start 4 m before the hairpin around the hoses
    rr_, cc_ = np.nonzero(t0_.duct); hose_c = np.array([t0_.origin[0] + cc_.mean() * t0_.resolution, t0_.origin[1] + rr_.mean() * t0_.resolution])
    s_hair = float(rl.s[int(np.argmin(np.linalg.norm(rl.xy - hose_c, axis=1)))])
    pose = env.sim.sample_spawn(1, 0.0, 0.0, s=torch.tensor([(s_hair - 4.0) % rl.length], device=dev), tid=torch.zeros(1, dtype=torch.long, device=dev))
    env.sim.reset(torch.arange(1, device=dev), poses=pose, speed=torch.full((1,), 2.0, device=dev))
    for _ in range(3): env.step(env.teacher_action_to_normalized(teacher(env.sim.state, env.sim.P, env.sim.tid)))
    def act_teacher(): env.step(env.teacher_action_to_normalized(teacher(env.sim.state, env.sim.P, env.sim.tid)))
    tr0 = tracks[0]
    def clear_at(x, y):
        c = int(round((x - tr0.origin[0]) / tr0.resolution)); r = int(round((y - tr0.origin[1]) / tr0.resolution))
        return float(tr0.edt[r, c]) if 0 <= r < tr0.edt.shape[0] and 0 <= c < tr0.edt.shape[1] else 0.0
    def cam_orbit(fr, d=3.2, el=0.38):
        f = fr["focus"]; az = 2.6 + 0.35 * getattr(vstate, "t", 0.0)
        x, y = fr["x"][f], fr["y"][f]
        while d > 1.3 and clear_at(x + d * math.cos(el) * math.cos(az), y + d * math.cos(el) * math.sin(az)) < 0.7:
            d -= 0.2                                                            # keep the camera out of the walls / hoses
        return ([x + d * math.cos(el) * math.cos(az), y + d * math.cos(el) * math.sin(az), 0.1 + d * math.sin(el)], [x, y, 0.15], [0, 0, 1])
    vstate = type("S", (), {})()
    def act_hero():
        act_teacher(); vstate.t = getattr(vstate, "t", 0.0) + 1.0 / FPS
    scene(sink, env, act_hero, 7.0, cam_orbit, "1080-beam LiDAR traced in 3D over the real venue",
          "the scan plane tilts with the sprung mass: beams hit the floor, pass over the 33 cm hoses and see beyond them")

    # ---- 2. swarm: hundreds of agents on the same map, overview zooming in
    tracks, _ = common.load_tracks(["real:korea_2025_iccas"])
    env = common.make_env(tracks, a.swarm, dev, EnvConfig(speed_cap=6.0, resample_track_on_reset=False), seed=3)
    model, extra = load_checkpoint(ck, dev); model.eval()
    obs = env.reset(seed=3)[0]; state = {"obs": obs}
    def act_policy():
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = model.act(scan, pro, deterministic=False)
        state["obs"] = env.step(act)[0]
    b = None
    def cam_over(fr):
        nonlocal b
        xs, ys = fr["x"], fr["y"]
        t = min(1.0, getattr(vstate, "t2", 0.0) / 8.0)
        cx, cy = float(np.median(xs)), float(np.median(ys))
        hgt = 34.0 - 14.0 * t
        return ([cx + 6.0, cy - 16.0 + 6.0 * t, hgt], [cx, cy, 0.0], [0, 0, 1])
    def act_swarm():
        act_policy(); vstate.t2 = getattr(vstate, "t2", 0.0) + 1.0 / FPS
    steps_m = extra.get("steps", 0) / 1e6
    scene(sink, env, act_swarm, 7.0, cam_over, f"{a.swarm} agents train in parallel on real competition maps",
          "PPO with an asymmetric critic, domain randomization, static obstacles, both lap directions",
          cars=a.swarm)

    # ---- 3. chase: the policy at speed with saliency + activations
    tracks, _ = common.load_tracks(["gen:competition:2"])
    env = common.make_env(tracks, 16, dev, EnvConfig(speed_cap=7.0, resample_track_on_reset=False), seed=5)
    mf, exf = load_checkpoint(ck_fast, dev); mf.eval(); intro = Introspector(mf)
    state["obs"] = env.reset(seed=5)[0]
    for _ in range(120):
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = mf.act(scan, pro, deterministic=True)
        state["obs"] = env.step(act)[0]
    focus = int(torch.argmax(env.ep_progress).item())
    sal_holder = {}
    def act_chase():
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = mf.act(scan, pro, deterministic=True)
        sal, mu = intro.saliency(scan, pro, focus)
        sal_holder["cols"] = sal_colors(sal)
        std = mf.actor.log_std.exp().detach().cpu().numpy()
        sal_holder["panel"] = panel_image(intro.h["hidden"][0].float().cpu().numpy(), intro.h["stem"][0].float().cpu().numpy(), mu, std, float("nan"), 7.0, "")
        state["obs"] = env.step(act)[0]
    def cam_chase(fr):
        f = fr["focus"]; x, y, yaw = fr["x"][f], fr["y"][f], fr["yaw"][f]
        return ([x - 2.0 * math.cos(yaw), y - 2.0 * math.sin(yaw), 0.9], [x + 1.5 * math.cos(yaw), y + 1.5 * math.sin(yaw), 0.1], [0, 0, 1])
    scene(sink, env, act_chase, 7.0, cam_chase, "LiDAR-only policy: no map, no localization",
          "point colours = the beams the network attends to; panel = neuron activations and the action distribution",
          focus=focus, panel_fn=lambda: sal_holder.get("panel"), colors_fn=lambda: sal_holder.get("cols"))

    # ---- 4. race: opponents in the scan, rear detection boxes
    tracks, rls = common.load_tracks(["real:korea_2025_iccas"], racelines=True)
    env = common.make_env(tracks, 3, dev, EnvConfig(race_size=3, opponent="teacher", speed_cap=4.0, opp_speed_range=(0.45, 0.55),
                                                   spawn_gap=(1.8, 2.6), resample_track_on_reset=False), seed=11, rls=rls)
    state["obs"] = env.reset(seed=11)[0]
    def act_race():
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = mf.act(scan, pro, deterministic=True)     # the policy drives the chasing car (slot 2)
        state["obs"] = env.step(act)[0]
    env.learner[:] = False; env.learner[2] = True
    def cam_race(fr):
        f = 2; x, y, yaw = fr["x"][f], fr["y"][f], fr["yaw"][f]
        return ([x - 2.6 * math.cos(yaw), y - 2.6 * math.sin(yaw), 1.1], [x + 2.5 * math.cos(yaw), y + 2.5 * math.sin(yaw), 0.15], [0, 0, 1])
    scene(sink, env, act_race, 7.0, cam_race, "Racing others: opponents appear in the scan as porous car bodies",
          "plus the rule-mandated rear detection box; car-to-car contact counts as a crash for both",
          focus=2, cars=3)

    # ---- end card
    sink.hold(card(["github.com/shchon11/F1tenth_E2E", "ROS 2 Humble  ·  f1tenth_system topics  ·  ONNX / TensorRT export for the Jetson"], [40, 24]), 3.5)
    sink.close()
    print("wrote", a.out, f"({sink.n} frames)", flush=True)
    gif = os.path.splitext(a.out)[0] + ".gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf",
                    "fps=8,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=80[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5",
                    gif], check=True)
    print("wrote", gif, os.path.getsize(gif) // 1024, "KB", flush=True)


if __name__ == "__main__":
    main()
