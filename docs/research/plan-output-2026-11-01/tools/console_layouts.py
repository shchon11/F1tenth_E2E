"""콘솔 워커(SimWorker) 한 세션에서 학습 장애물 배치를 여러 개 (11-02): 콘솔의 리셋 버튼과 같이(CMD_RESET = env.reset())
배치마다 새로 뽑아 초씩 달리고, 배치마다 랩·접촉(콘솔 계기와 같은 수)·멈춤을 한 줄로. 접촉이 난 배치는 그림(맵 + 상자 +
주행 경로 + 궤적 + 접촉, 오른쪽에 접촉이 몰린 곳 확대).
11-02 이유: 시드 6개가 모두 깨끗했는데 콘솔에서 무작위로 뽑힌 한 배치에서 상자 9회(세게 4) -> 몇 배치에 한 번인지, 어떤 장면인지.
인자: 체크포인트 맵id 배치당초 배치수 시드 그림머리"""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim import props as _props
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, n_lay, seed, head = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0,
                                    randomize=bool(int(os.environ.get("F1SIM_EVAL_RANDOMIZE", "0")))), 0)   # 11-06: physics randomisation as in training
env = s["env"]; proc = env.procedural
from f1sim import maps, tracks
tr = maps.load(tracks.resolve(mp))
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
occ = tr.occupancy; res = tr.resolution; H, W = occ.shape
ext = [tr.origin[0], tr.origin[0] + W * res, tr.origin[1], tr.origin[1] + H * res]
KCOL = ("red", "darkorange", "magenta"); KNAME = ("벽", "장애물", "차")
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car")]
i_hd = [CC.index(k) for k in ("hard_wall", "hard_prop", "hard_car")]
_fp = {}


def footprints():
    out = []
    sid = proc.p_sid[0].cpu().numpy(); pose = proc.p_poses[0].cpu().numpy()
    for j, k in enumerate(sid):
        if k < 0:
            continue
        sh = proc.shapes[int(k)]
        if int(k) not in _fp:
            _fp[int(k)] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float)
        p = _fp[int(k)]; c, sn = np.cos(pose[j, 2]), np.sin(pose[j, 2])
        out.append(np.stack([pose[j, 0] + p[:, 0] * c - p[:, 1] * sn, pose[j, 1] + p[:, 0] * sn + p[:, 1] * c], 1))
    return out


def picture(L, props0, props1, xs, ys, vs, plans, hits, png):
    fig = plt.figure(figsize=(20, 11))

    def draw(ax, lw_plan=0.7):
        ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.6)
        for poly in props0:
            ax.fill(poly[:, 0], poly[:, 1], color="0.35", alpha=0.6, lw=0)
        for poly in props1:
            ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], "b-", lw=0.8)
        sc = ax.scatter(xs, ys, c=vs, s=2, cmap="turbo", vmin=0, vmax=10)
        for t, p in plans:
            ax.plot(p[:, 0], p[:, 1], "k-", lw=lw_plan)
        for t, j, hard, x, y, v in hits:
            ax.plot(x, y, "x", color=KCOL[j], ms=(16 if hard else 9), mew=(3 if hard else 1.5))
        return sc

    ax = fig.add_axes([0.02, 0.05, 0.56, 0.88])
    sc = draw(ax)
    ax.set_xlim(xs.min() - 1, xs.max() + 1); ax.set_ylim(ys.min() - 1, ys.max() + 1); ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.03, label="속도 m/s")
    cnt = [sum(1 for h in hits if h[1] == j) for j in range(3)]; hc = [sum(1 for h in hits if h[1] == j and h[2]) for j in range(3)]
    ax.set_title(f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)} @ {mp} 학습 장애물 배치 {L} (시드 {seed}): "
                 f"{secs:.0f} s | 접촉 벽 {cnt[0]}(세게 {hc[0]}) 장애물 {cnt[1]}(세게 {hc[1]}) | 회색 = 처음 상자, 파란 테두리 = 끝 위치")
    bx = fig.add_axes([0.60, 0.05, 0.38, 0.88])
    hx = np.array([[h[3], h[4]] for h in hits])
    first = hx[0]                                          # the first contact: what started it
    draw(bx, lw_plan=1.0)
    bx.set_xlim(first[0] - 1.6, first[0] + 1.6); bx.set_ylim(first[1] - 1.6, first[1] + 1.6); bx.set_aspect("equal")
    near = [h for h in hits if abs(h[3] - first[0]) < 1.6 and abs(h[4] - first[1]) < 1.6]
    bx.set_title("첫 접촉 확대: " + ", ".join(f"{h[0]:.1f}s {KNAME[h[1]]}{'(세게)' if h[2] else ''} {h[5]:.1f}m/s"
                                       for h in near[:6]), fontsize=9)
    fig.savefig(png, dpi=60); plt.close(fig)


tot = np.zeros(3, int); toth = np.zeros(3, int); bad = 0; t_all = time.time()
stuck_tot, laps_tot, stopped_lay = 0.0, 0, 0          # 11-06: the stopped time too (a car standing in front of a box has no contact)
for L in range(n_lay):
    if L > 0:
        s["obs"], _ = env.reset()                          # what the console's 리셋 does (CMD_RESET)
    props0 = footprints()
    xs, ys, vs, plans, hits, laps = [], [], [], [], [], []
    lap0 = None; t_last = None
    prev_on = env.sim.contact_frame()[0].float().cpu().numpy()[i_on]; prev_hd = env.sim.contact_frame()[0].float().cpu().numpy()[i_hd]
    on0, hd0 = prev_on.copy(), prev_hd.copy()
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        st = env.sim.state[0]
        xs.append(float(st[0])); ys.append(float(st[1])); vs.append(float(st[3]))
        lp = int(env.last_result.lap[0])
        if lap0 is None:
            lap0 = lp
        elif lp != lap0:
            t = (k + 1) * 0.025
            if t_last is not None:
                laps.append(round(t - t_last, 2))
            t_last, lap0 = t, lp
        cf = env.sim.contact_frame()[0].float().cpu().numpy()
        on, hd = cf[i_on], cf[i_hd]
        for j in range(3):
            if on[j] > prev_on[j]:
                hits.append((k * 0.025, j, bool(hd[j] > prev_hd[j]), xs[-1], ys[-1], vs[-1]))
        prev_on, prev_hd = on, hd
        if k % 20 == 0:
            fr = w._snapshot(s)
            if fr.get("plan") is not None:
                plans.append((k * 0.025, np.asarray(fr["plan"], dtype=float)[:, :2]))
    c = np.round(prev_on - on0).astype(int); ch = np.round(prev_hd - hd0).astype(int)
    tot += c; toth += ch
    vs_a = np.asarray(vs); stuck = 0.025 * int(np.sum(np.abs(vs_a[120:]) < 0.3))
    flag = ""
    if hits:
        bad += 1
        png = f"{head}_L{L}.png"
        picture(L, props0, footprints(), np.asarray(xs), np.asarray(ys), vs_a, plans, hits, png)
        flag = f" | 그림 {png}"
    stuck_tot += stuck; laps_tot += len(laps); stopped_lay += int(stuck >= 5.0)
    print(f"배치 {L}: 장애물 {len(props0)}개 | 랩 {laps} | 평균 속도 {vs_a.mean():.2f} | 접촉 벽 {c[0]}(세게 {ch[0]}) 장애물 {c[1]}(세게 {ch[1]}) "
          f"| 멈춤 {stuck:.1f} s{flag}", flush=True)
km = 0.0
print(f"합계 {n_lay}배치 x {secs:.0f} s ({time.time() - t_all:.0f} s 걸림): 접촉 난 배치 {bad}/{n_lay} | 벽 {tot[0]}(세게 {toth[0]}) "
      f"장애물 {tot[1]}(세게 {toth[1]}) | 멈춰 있던 시간 {stuck_tot:.0f} s / {n_lay * (secs - 3):.0f} s, 5 s 넘게 멈춘 배치 {stopped_lay}/{n_lay} | "
      f"완주한 랩 {laps_tot}", flush=True)
