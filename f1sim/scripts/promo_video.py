"""README clip: raw simulator footage with the viewer's own telemetry HUD, hard cuts, no captions.
Headless (EGL). -> docs/promo.mp4 + docs/promo.gif.    python3 scripts/promo_video.py"""
import argparse, math, os, subprocess, sys, time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from f1sim import Config
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.learn.model import load_checkpoint
from f1sim.learn.obs import flatten_obs
from f1sim.learn.watch import Introspector, panel_image, sal_colors
from f1sim.viewer.native import NativeViewer

W, H, FPS = 1280, 720, 30
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs")


class CamViewer(NativeViewer):
    cam_fn = None

    def _camera(self, fr):
        if self.cam_fn is None:
            return super()._camera(fr)
        eye, tgt, up = self.cam_fn(fr)
        return np.asarray(eye, float), np.asarray(tgt, float), np.asarray(up, float)


def grab(v):
    if v.scene.fbo_ms is not None:
        v.scene.ctx.copy_framebuffer(v.scene.fbo, v.scene.fbo_ms)
    return Image.frombytes("RGB", (v.width, v.height), v.scene.fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)


class Sink:
    def __init__(self, path):
        self.proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                      "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "slow", path], stdin=subprocess.PIPE)
        self.n = 0

    def put(self, img):
        self.proc.stdin.write(np.asarray(img, np.uint8).tobytes()); self.n += 1

    def close(self):
        self.proc.stdin.close(); self.proc.wait()


def tag(img, text):
    """small monospace line bottom-left, same style as the HUD"""
    d = ImageDraw.Draw(img, "RGBA"); f = ImageFont.truetype(MONO, 17)
    w = d.textlength(text, font=f); d.rectangle([16, H - 40, 16 + w + 16, H - 12], fill=(0, 0, 0, 140))
    d.text((24, H - 36), text, font=f, fill=(230, 230, 235, 255))
    return img


def cut(sink, env, act_fn, seconds, cam_fn, label, focus=0, panel_fn=None, colors_fn=None, plan=False, cars=None, sim_steps=1):
    env.sim.warmup()
    v = CamViewer(env.sim, headless=True, max_cars=cars or env.B, width=W, height=H)
    v.focus = focus; v.cam_fn = cam_fn; v.show_trails = False
    for i in range(int(seconds * FPS)):
        for _ in range(sim_steps): act_fn()
        v.update(env.last_result)
        if colors_fn is not None: v.point_colors = colors_fn()
        if panel_fn is not None: v.panel = panel_fn()
        if plan: v.plan = env.plan_world(focus); v.plan_pred = env.plan_world(focus, predicted=True)
        v.render()
        sink.put(tag(grab(v), label))
    v.close()


def chase(fr, back=2.0, up=0.85, ahead=1.5, h=0.1):
    f = fr["focus"]; x, y, yaw = fr["x"][f], fr["y"][f], fr["yaw"][f]
    return ([x - back * math.cos(yaw), y - back * math.sin(yaw), up], [x + ahead * math.cos(yaw), y + ahead * math.sin(yaw), h], [0, 0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DOCS, "promo.mp4"))
    ap.add_argument("--swarm", type=int, default=256)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck_v3 = os.path.join(common.RUNS_DIR, "ppo_v3", "ppo_final.pt")
    ck_v4 = next((p for p in (os.path.join(common.RUNS_DIR, "ppo_v4", "ppo_latest.pt"),) if os.path.exists(p)), ck_v3)
    sink = Sink(a.out); state = {}
    nom = Config(); nom.rand.enabled = False

    # 1. chase: ppo_v3 flat out on a procedural competition track, saliency + activations
    tracks, _ = common.load_tracks(["gen:competition:2"])
    env = common.make_env(tracks, 16, dev, EnvConfig(speed_cap=7.0, resample_track_on_reset=False), seed=5)
    mf, exf = load_checkpoint(ck_v3, dev); mf.eval(); intro = Introspector(mf)
    state["obs"] = env.reset(seed=5)[0]
    for _ in range(150):
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = mf.act(scan, pro, deterministic=True)
        state["obs"] = env.step(act)[0]
    focus = int(torch.argmax(env.ep_progress).item()); hold = {}
    def act_chase():
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = mf.act(scan, pro, deterministic=True)
        sal, mu = intro.saliency(scan, pro, focus); hold["cols"] = sal_colors(sal)
        std = mf.actor.log_std.exp().detach().cpu().numpy()
        hold["panel"] = panel_image(sal, mu, std, None, 7.0, f"ppo_v3 {exf.get('steps', 0) / 1e6:.0f}M steps")
        state["obs"] = env.step(act)[0]
    cut(sink, env, act_chase, 6.0, chase, "ppo_v3  gen:competition:2  cap 7 m/s  deterministic  | points: saliency", focus=focus,
        panel_fn=lambda: hold.get("panel"), colors_fn=lambda: hold.get("cols"))

    # 2. the planner: teacher plans on the Korea championship map, tracked by the iLQR, plan drawn with its speed profile
    tracks, rls = common.load_tracks(["real:korea_2025_iccas"], racelines=True)
    env = common.make_env(tracks, 1, dev, EnvConfig(speed_cap=5.0, resample_track_on_reset=False, action_mode="plan"), cfg=nom, seed=7)
    teacher = common.make_teacher(rls, env); env.reset(seed=7)
    def act_plan(): env.step(env.teacher_label(teacher))
    for _ in range(30): act_plan()
    cut(sink, env, act_plan, 7.0, lambda fr: chase(fr, back=1.6, up=0.7, ahead=2.2), "real:korea_2025_iccas  plan action space: local trajectory (colour = speed) -> iLQR tracker", plan=True)

    # 3. swarm on the same map from above: ppo_v4 as it trains
    tracks, _ = common.load_tracks(["real:korea_2025_iccas"])
    env = common.make_env(tracks, a.swarm, dev, EnvConfig(speed_cap=6.0, resample_track_on_reset=False), seed=3)
    m4, ex4 = load_checkpoint(ck_v4, dev); m4.eval()
    state["obs"] = env.reset(seed=3)[0]
    def act_swarm():
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = m4.act(scan, pro, deterministic=False)
        state["obs"] = env.step(act)[0]
    def cam_top(fr):
        xs, ys = fr["x"], fr["y"]; cx, cy = float(np.median(xs)), float(np.median(ys))
        return ([cx + 4.0, cy - 9.0, 21.0], [cx, cy, 0.0], [0, 0, 1])
    cut(sink, env, act_swarm, 5.0, cam_top, f"ppo_v4 {ex4.get('steps', 0) / 1e6:.0f}M steps  {a.swarm} cars  stochastic actions  (red = crashed, respawns)", cars=a.swarm)

    # 4. race: ppo_v3 chasing two slower teacher cars, rear detection boxes in the scan
    tracks, rls = common.load_tracks(["real:korea_2025_iccas"], racelines=True)
    env = common.make_env(tracks, 3, dev, EnvConfig(race_size=3, opponent="teacher", speed_cap=4.0, opp_speed_range=(0.45, 0.55),
                                                   spawn_gap=(1.8, 2.6), resample_track_on_reset=False), seed=11, rls=rls)
    state["obs"] = env.reset(seed=11)[0]; env.learner[:] = False; env.learner[2] = True
    def act_race():
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = mf.act(scan, pro, deterministic=True)
        state["obs"] = env.step(act)[0]
    cut(sink, env, act_race, 6.0, lambda fr: chase(fr, back=2.4, up=1.0, ahead=2.5), "race: ppo_v3 (car 2) vs two teacher cars at 0.5x  | yellow points: other cars", focus=2, cars=3)

    # 5. close orbit at speed on the procedural track (ppo_v3)
    tracks, _ = common.load_tracks(["gen:competition:2"])
    env = common.make_env(tracks, 16, dev, EnvConfig(speed_cap=7.0, resample_track_on_reset=False), seed=5)
    state["obs"] = env.reset(seed=5)[0]
    for _ in range(150):
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = mf.act(scan, pro, deterministic=True)
        state["obs"] = env.step(act)[0]
    focus = int(torch.argmax(env.ep_progress).item()); t_ = {"t": 0.0}
    def act_orbit():
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad(): act, _ = mf.act(scan, pro, deterministic=True)
        state["obs"] = env.step(act)[0]; t_["t"] += 1.0 / FPS
    def cam_orbit(fr):
        f = fr["focus"]; x, y = fr["x"][f], fr["y"][f]; az = fr["yaw"][f] + math.pi + 0.8 * math.sin(0.5 * t_["t"])
        return ([x + 1.4 * math.cos(az), y + 1.4 * math.sin(az), 0.5], [x, y, 0.12], [0, 0, 1])
    cut(sink, env, act_orbit, 5.0, cam_orbit, "ppo_v3  gen:competition:2  7 m/s", focus=focus)

    # end: plain address
    img = Image.new("RGB", (W, H), (8, 9, 12)); d = ImageDraw.Draw(img); f = ImageFont.truetype(MONO, 30)
    t = "f1sim   github.com/shchon11/F1tenth_E2E"; w = d.textlength(t, font=f); d.text(((W - w) / 2, H / 2 - 15), t, font=f, fill=(220, 220, 225))
    for _ in range(int(2.0 * FPS)): sink.put(img)
    sink.close(); print("wrote", a.out, f"({sink.n} frames)", flush=True)
    gif = os.path.splitext(a.out)[0] + ".gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf",
                    "fps=7,scale=560:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=80[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5", gif], check=True)
    print("wrote", gif, os.path.getsize(gif) // 1024, "KB", flush=True)


if __name__ == "__main__":
    main()
