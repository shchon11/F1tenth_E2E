"""평가 배치 주행을 영상으로 (11-06 09:30, 사용자: "중간중간 그림 뽑는거 아예 영상으로 뽑아서 보여줄 수 있어?"). console_layouts.py와 같은
세션·배치 순서(시드, env.reset)로 배치마다 달리며 0.1 s마다 한 장: 왼쪽 = 맵 전체(상자, 지나온 길 3 s, 차, 계획 경로, 접촉 표시),
오른쪽 = 차 주변 8 m 확대. 위 글자 = checkpoint, 배치 번호, 시간, 속도, 지금까지 접촉 수. 재생은 실제의 2배 빠르기(20 fps). 배치들을 한 mp4로 이어 붙임.
인자: 체크포인트 맵id 배치당초 배치수 시드 출력mp4 [첫배치 번호, 기본 0: 그 앞 배치는 영상 없이 리셋만]"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import font_manager
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.viewer import recorder as REC
from f1sim import props as _props, maps, tracks
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, n_lay, seed, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
first_L = int(sys.argv[7]) if len(sys.argv) > 7 else 0
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
env = s["env"]; proc = env.procedural
tr = maps.load(tracks.resolve(mp))
occ = tr.occupancy; res = tr.resolution; H, W = occ.shape
ext = [tr.origin[0], tr.origin[0] + W * res, tr.origin[1], tr.origin[1] + H * res]
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car")]
i_hd = [CC.index(k) for k in ("hard_wall", "hard_prop", "hard_car")]
hl, hw = 0.5 * float(env.cfg.vehicle.length), 0.5 * float(env.cfg.vehicle.width)
_fp = {}
name = f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run).replace('state_', '').replace('.pt', '')}"


def footprints():
    out_ = []
    sid = proc.p_sid[0].cpu().numpy(); pose = proc.p_poses[0].cpu().numpy()
    for j, k in enumerate(sid):
        if k < 0:
            continue
        sh = proc.shapes[int(k)]
        if int(k) not in _fp:
            _fp[int(k)] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float)
        p = _fp[int(k)]; c, sn = np.cos(pose[j, 2]), np.sin(pose[j, 2])
        out_.append(np.stack([pose[j, 0] + p[:, 0] * c - p[:, 1] * sn, pose[j, 1] + p[:, 0] * sn + p[:, 1] * c], 1))
    return out_


def car_poly(x, y, yaw):
    c, sn = np.cos(yaw), np.sin(yaw)
    pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.stack([x + pts[:, 0] * c - pts[:, 1] * sn, y + pts[:, 0] * sn + pts[:, 1] * c], 1)


WPX, HPX, DPI = 1280, 720, 80
fig = plt.figure(figsize=(WPX / DPI, HPX / DPI), dpi=DPI, facecolor="white")
ax = fig.add_axes([0.01, 0.03, 0.58, 0.86]); bx = fig.add_axes([0.61, 0.03, 0.38, 0.86])
title = fig.text(0.01, 0.95, "", fontsize=13, va="center")
sub = fig.text(0.01, 0.91, "흰 = 달릴 수 있는 곳, 회색 = 벽, 갈색 = obstacle, 파랑 = 차, 검은 선 = policy의 계획 경로, 색 점 = 지나온 3 s(속도 색), "
               "X = 접촉(빨강 벽 · 주황 obstacle, 굵게 = 세게)", fontsize=9, color="0.3", va="center")
enc = REC.FfmpegEncoder(out, WPX, HPX, 20)
KCOL = ("red", "darkorange", "magenta")
tot_frames = 0
for L in range(n_lay):
    if L > 0:
        s["obs"], _ = env.reset()
    if L < first_L:
        continue
    props = footprints()
    for a in (ax, bx):
        a.clear(); a.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.6)
        for poly in props:
            a.fill(poly[:, 0], poly[:, 1], color="saddlebrown", alpha=0.9, lw=0)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    xs_all = []
    hits = []
    trail_sc = [a.scatter([], [], c=[], s=6, cmap="turbo", vmin=0, vmax=8) for a in (ax, bx)]
    car_p = [a.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc="tab:blue", ec="k", lw=1.0, zorder=5)) for a in (ax, bx)]
    plan_l = [a.plot([], [], "k-", lw=1.4, zorder=4)[0] for a in (ax, bx)]
    hit_art = []
    prev_on = env.sim.contact_frame()[0].float().cpu().numpy()[i_on]; prev_hd = env.sim.contact_frame()[0].float().cpu().numpy()[i_hd]
    cnt = np.zeros(3, int); hcnt = np.zeros(3, int)
    trail = []
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        st = env.sim.state[0].cpu().numpy()
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        on, hd = cf[i_on], cf[i_hd]
        for j in range(3):
            if on[j] > prev_on[j]:
                hard = bool(hd[j] > prev_hd[j]); cnt[j] += 1; hcnt[j] += int(hard)
                for a in (ax, bx):
                    hit_art.append(a.plot(st[0], st[1], "x", color=KCOL[j], ms=(14 if hard else 8), mew=(3 if hard else 1.5), zorder=6)[0])
        prev_on, prev_hd = on, hd
        trail.append((st[0], st[1], st[3])); xs_all.append((st[0], st[1]))
        trail = trail[-120:]
        if k % 4:
            continue
        tp = np.array(trail)
        fr = w._snapshot(s)
        plan = np.asarray(fr["plan"], dtype=float)[:, :2] if fr.get("plan") is not None else np.zeros((0, 2))
        cp = car_poly(st[0], st[1], st[2])
        for i, a in enumerate((ax, bx)):
            trail_sc[i].set_offsets(tp[:, :2]); trail_sc[i].set_array(tp[:, 2])
            car_p[i].set_xy(cp)
            plan_l[i].set_data(plan[:, 0], plan[:, 1])
        xa = np.array(xs_all)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        bx.set_xlim(st[0] - 4, st[0] + 4); bx.set_ylim(st[1] - 4, st[1] + 4)
        title.set_text(f"{name} · obstacle 배치 {L} · {k * 0.025:4.1f} s · 속도 {st[3]:4.1f} m/s · "
                       f"obstacle 접촉 {cnt[1]}(세게 {hcnt[1]}) · 벽 접촉 {cnt[0]}(세게 {hcnt[0]})")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        enc.write(np.ascontiguousarray(buf).tobytes()); tot_frames += 1
    for h in hit_art:
        h.remove()
    print(f"배치 {L}: obstacle 접촉 {cnt[1]}(세게 {hcnt[1]}) 벽 {cnt[0]}(세게 {hcnt[0]})", flush=True)
rc = enc.close()
print(f"영상 {out} | {tot_frames}장, {tot_frames / 20:.0f} s | {REC.describe_size(out)} | ffmpeg {rc} {enc.error or ''}", flush=True)
