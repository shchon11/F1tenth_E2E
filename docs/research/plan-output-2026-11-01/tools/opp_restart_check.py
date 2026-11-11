"""선 상대 차를 다시 출발시키는 수정(state_rl.OppRestart)의 전·후 (11-11). `drive_record.py`의 통째 기록 둘(같은 시드, 뒤 것은 덮어쓰기
JSON에 "opp_restart_s")을 읽음(GPU 안 씀). 상자 있는 경주의 상대 차: 2 s 동안 0.5 m도 못 간 시간 비율, 상자에 닿은 횟수; 학습 차의 세게
닿음(벽·상자·상대 차); 다시 출발한 건마다 다시 나타난 자리(학습 차까지 거리, 벽까지, 가장 가까운 상자까지)와 2 s 뒤 속도. 장면 8장:
전체 트랙에 서 있던 자리(빨강 테두리)와 5 s 자취, 다시 나타난 자리(빨강 채움)와 그 뒤 4 s 자취, 그 순간 학습 차(파랑), 상자.
인자: 전.npz 후.npz 출력폴더"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, trk_name, C_BOX, C_CAR, plt
from matplotlib.patches import Polygon

f_a, f_b, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
WIN, MOVE = 2.0, 0.5


def stats(R):
    dt = R.dt; K = int(round(WIN / dt))
    on = R.on[:-1].astype(bool); box = (R.sid[:-1] >= 0).any(2); op = R.op[:-1]
    ors = R.z["ors"] if "ors" in R.z.files else np.zeros_like(R.rs)
    disp = np.hypot(op[K:, :, 0] - op[:-K, :, 0], op[K:, :, 1] - op[:-K, :, 1])
    c0 = np.concatenate([np.zeros((1, R.B), int), (R.rs | ors).astype(int).cumsum(0)], 0)
    same = (c0[K:R.T] - c0[:R.T - K]) == 0                             # no restart of either car inside the window
    m = on[K:] & on[:-K] & box[K:] & same
    oc = R.z["ocnt"]; d_o = np.clip(np.diff(np.concatenate([np.zeros((1, R.B, 6), np.float32), oc], 0), axis=0), 0, None)
    mb = on & box
    mins_o = mb.sum() * dt / 60.0
    lm = R.on[:-1].astype(bool)                                        # learner rows with an opponent
    out = {"stuck": float((disp < MOVE)[m].mean()) * 100, "slow": float((np.abs(op[..., 3]) < 0.3)[mb].mean()) * 100,
           "opp_box": float((d_o[..., 1] * mb).sum() / mins_o), "mins_o": mins_o, "restarts": int(ors.sum())}
    for j, k in ((3, "wall"), (4, "box"), (5, "car")):
        out[f"hard_{k}"] = float((R.d[..., j] > 0.5).sum() / (R.T * R.B * dt / 60.0))
        out[f"hard_{k}_opp"] = float(((R.d[..., j] > 0.5) & lm).sum() / (lm.sum() * dt / 60.0))
    near = (np.hypot(R.st[:-1, :, 0] - op[..., 0], R.st[:-1, :, 1] - op[..., 1]) < 3.0) & mb
    out["near_stuck"] = float((near[K:] & (disp < MOVE) & m).sum() / max(m.sum(), 1)) * 100
    return out


A, B_ = Rec(f_a), Rec(f_b)
sa, sb = stats(A), stats(B_)
print(f"{A.name}: 학습 세계, 학습 조건, 학습 차 {A.B}대 x {A.T * A.dt:.0f} s, 같은 시드 | 뒤: {WIN:g} s 동안 {MOVE:g} m도 못 간 상대 차 다시 출발 {sb['restarts']}번", flush=True)
for lab, k, u in (("상자 있는 경주에서 상대 차가 2 s 동안 0.5 m도 못 감", "stuck", "%"), ("  (0.3 m/s 아래로 서 있음)", "slow", "%"),
                  ("  그런 상대 차 3 m 안에 학습 차가 있음", "near_stuck", "%"), ("상대 차가 상자에 닿음", "opp_box", "/분"),
                  ("학습 차 세게 닿음: 벽", "hard_wall", "/분"), ("학습 차 세게 닿음: 상자", "hard_box", "/분"),
                  ("학습 차 세게 닿음: 상대 차(상대 차 있는 차만)", "hard_car_opp", "/분")):
    print(f"  {lab}: 전 {sa[k]:.2f} -> 후 {sb[k]:.2f} {u}", flush=True)

# every restart: where the opponent reappears
R = B_; dt = R.dt; ors = R.z["ors"]
ev = []
for t, b in zip(*np.nonzero(ors)):
    if t + 1 >= R.T:
        continue
    k = int(R.tid[t, b]); new = R.op[t + 1, b]; old = R.op[t, b]
    polys = R.boxes(t + 1, b)
    dbox = min((float(np.hypot(*(p.mean(0) - new[:2]))) for p in polys), default=np.nan)
    t2 = min(t + 1 + int(2.0 / dt), R.T)
    ev.append(dict(t=int(t), b=int(b), k=k, d_learner=float(np.hypot(*(R.st[t + 1, b, :2] - new[:2]))),
                   d_old=float(np.hypot(*(old[:2] - new[:2]))), wall=float(R.wall_dist(k, new[None, :2])[0]), dbox=dbox,
                   v0=float(new[3]), v2=float(R.op[t2, b, 3]), again=bool(ors[t + 1:t + 1 + int(5.0 / dt), b].any()),
                   hit5=bool((R.d[t + 1:t + 1 + int(5.0 / dt), b, 2] > 0.5).any())))
if ev:
    g = lambda key: np.array([e[key] for e in ev])
    print(f"[다시 출발 {len(ev)}건] 다시 나타난 자리: 학습 차까지 중앙 {np.median(g('d_learner')):.1f} m (가장 가까운 것 {g('d_learner').min():.1f} m, 3 m 안 {int((g('d_learner') < 3).sum())}건) | "
          f"벽까지 중앙 {np.median(g('wall')):.2f} m (가장 가까운 것 {g('wall').min():.2f} m) | 가장 가까운 상자까지 중앙 {np.nanmedian(g('dbox')):.1f} m "
          f"(0.6 m 안 {int((g('dbox') < 0.6).sum())}건) | 2 s 뒤 속도 중앙 {np.median(g('v2')):.1f} m/s (0.3 m/s 아래 {int((np.abs(g('v2')) < 0.3).sum())}건) | "
          f"5 s 안에 또 다시 출발 {int(g('again').sum())}건 | 5 s 안에 학습 차와 닿음 {int(g('hit5').sum())}건", flush=True)
    cars = len({e["b"] for e in ev})
    print(f"[다시 출발] 상대 차 {cars}대가 {len(ev)}번 (한 대 최대 {max(np.bincount(g('b').astype(int)))}번)", flush=True)
    fig, axs = plt.subplots(2, 4, figsize=(22, 11.6))
    pick = ev[:: max(len(ev) // 8, 1)][:8]
    for ax, e in zip(axs.ravel(), pick):
        t, b, k = e["t"], e["b"], e["k"]
        R.draw_map(ax, k)
        for p in R.boxes(t, b):
            ax.add_patch(Polygon(p, closed=True, fc=C_BOX, ec="none", zorder=2))
        a0 = max(t - int(5.0 / dt), 0); a1 = min(t + 1 + int(4.0 / dt), R.T)
        ax.plot(R.op[a0:t + 1, b, 0], R.op[a0:t + 1, b, 1], ".", color="#ff8a80", ms=5, zorder=3)
        ax.plot(R.op[t + 1:a1, b, 0], R.op[t + 1:a1, b, 1], ".", color=C_CAR, ms=5, zorder=3)
        ax.add_patch(Polygon(R.body(*R.op[t, b, :3]), closed=True, fc="none", ec=C_CAR, lw=2.0, zorder=5))
        ax.add_patch(Polygon(R.body(*R.op[t + 1, b, :3]), closed=True, fc=C_CAR, ec="k", lw=0.8, zorder=5))
        ax.add_patch(Polygon(R.body(*R.st[t + 1, b, :3]), closed=True, fc="tab:blue", ec="k", lw=0.8, zorder=5))
        ax.annotate("", xy=R.op[t + 1, b, :2], xytext=R.op[t, b, :2], arrowprops=dict(arrowstyle="->", color=C_CAR, lw=1.2, ls="--"), zorder=4)
        ax.set_title(f"{trk_name(k)} · {t * dt:.1f} s · {R.kinds[int(R.kind[t, b])]} x{R.scale[t, b]:.2f}\n"
                     f"다시 나타난 자리: 학습 차까지 {e['d_learner']:.1f} m, 벽까지 {e['wall']:.2f} m · 2 s 뒤 {e['v2']:.1f} m/s", fontsize=11)
    for ax in axs.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"{R.name} · 선 상대 차 다시 출발 -- 빨강 테두리 = 서 있던 자리(옅은 점 = 그 앞 5 s), 빨강 채움 = 다시 나타난 자리(짙은 점 = 그 뒤 4 s), "
                 f"파랑 = 그 순간 학습 차, 갈색 = 상자", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0); plt.savefig(os.path.join(out_dir, "opp_restart_scenes.png"), dpi=70); plt.close(fig)
