"""출발 직후 접촉과 벽으로 직진하는 건을 장면으로 (11-11: 세게 닿은 장면 분류에서 상대 차 접촉이 출발 5 s 안에 두 배, 벽 접촉이 몇 자리에
몰리고 출발 1-2 s 건이 섞여 있었음. 사용자 승인: 학습 없이 장면으로 더 파 봄). `drive_record.py`의 기록 둘을 읽음(GPU 안 씀):
  starts = 5 s 에피소드를 반복한 기록(출발 2,048번)   whole = 40 s 통째 기록
(가) 출발: 에피소드 안 시각별 세게 닿음 /분(벽·상자·상대 차, 0.5 s 칸; 점선 = 통째 기록의 5 s 뒤 수준), 출발 때 상대 차가 우리 차 기준
     어디 있었나(앞 dx, 왼쪽 dy)에 2 s 안 접촉 여부를 색으로, 출발 때 두 차 사이 거리별 2 s 안 접촉 비율, 출발 때 진행 방향으로 벽까지
     거리 / 출발 속도(= 그대로 가면 벽까지 걸리는 시간)별 2 s 안 벽 접촉 비율. 장면: 출발 2 s 안 상대 차 접촉 12장, 출발 3 s 안 벽 접촉 12장
     (테두리만 = 출발 자세, 채움 = 닿는 순간, 점 = 그 사이 자취, 검은 선 = 첫 plan, 분홍 선 = 닿는 결정의 plan).
(나) 벽에 몰린 자리(통째 기록, 출발 3 s 뒤): 같은 트랙 1.5 m 안에 모인 세게 닿은 건끼리 묶어 많은 순으로 8자리 -- 닿은 차의 4 s 자취(속도 색)와
     같은 트랙에서 그 자리 3 m 안을 닿지 않고 지나간 다른 차들의 자취(회색), 닿기 1 s 전 속도 vs 지나간 차들의 가장 가까울 때 1 s 전 속도.
인자: starts.npz whole.npz 출력폴더"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
f_st, f_wh, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
from rec_lib import Rec, trk_name, MAPC, C_WALL, C_BOX, C_CAR

S, W = Rec(f_st), Rec(f_wh)
dt = S.dt
print(f"{S.name}: 학습 세계, 학습 조건 | 출발 기록 학습 차 {S.B}대 x {S.T * dt:.0f} s, 통째 기록 {W.B}대 x {W.T * dt:.0f} s", flush=True)

# ---------------------------------------------------------------------------------------------- (가) 출발
# episodes of the start record: (b, k0, k1) with k1 the decision that restarted it (not counted)
eps = []
for b in range(S.B):
    k0 = 0
    for t in np.nonzero(S.rs[:, b])[0]:
        eps.append((b, k0, int(t))); k0 = int(t) + 1
    if S.T - k0 >= 20:
        eps.append((b, k0, S.T))
n_ep = len(eps)
ep_len = np.array([k1 - k0 for (_, k0, k1) in eps]) * dt
print(f"[출발] 에피소드 {n_ep}개 (길이 중앙 {np.median(ep_len):.1f} s)", flush=True)
# hard contacts by time in the episode, 0.5 s bins
NB = 10
hist = np.zeros((3, NB)); expo = np.zeros(NB)
for (b, k0, k1) in eps:
    n = min(k1 - k0, NB * 5)
    tt = (np.arange(n) // 5)
    np.add.at(expo, tt, dt / 60.0)
    for j in range(3):
        np.add.at(hist[j], tt, (S.d[k0:k0 + n, b, 3 + j] > 0.5).astype(float))
rate = hist / np.maximum(expo, 1e-9)[None]
# the whole record's level after 5 s
t_in = (np.arange(W.T)[:, None] - W.k0) * dt
late = t_in >= 5.0
late_min = late.sum() * dt / 60.0
late_rate = [(W.d[..., 3 + j][late] > 0.5).sum() / late_min for j in range(3)]
early = ~late
early_rate = [(W.d[..., 3 + j][early] > 0.5).sum() / (early.sum() * dt / 60.0) for j in range(3)]
print("[출발] 세게 닿음 /분, 에피소드 안 시각별 (출발 기록) -- 벽 / 상자 / 상대 차", flush=True)
for i in range(NB):
    print(f"   {i * 0.5:.1f}-{i * 0.5 + 0.5:.1f} s: {rate[0, i]:.2f} / {rate[1, i]:.2f} / {rate[2, i]:.2f}   ({int(hist[0, i])}/{int(hist[1, i])}/{int(hist[2, i])}건)", flush=True)
print(f"   통째 기록: 5 s 안 {early_rate[0]:.2f} / {early_rate[1]:.2f} / {early_rate[2]:.2f}, 5 s 뒤 {late_rate[0]:.2f} / {late_rate[1]:.2f} / {late_rate[2]:.2f}", flush=True)

# start geometry and what follows in the first 2 s
rows = []
for (b, k0, k1) in eps:
    s0, o0 = S.st[k0, b], S.op[k0, b]
    k = int(S.tid[k0, b])
    n2 = min(k1 - k0, 20)
    dd = S.d[k0:k0 + n2, b]
    car_any = dd[:, 2] > 0.5; car_hard = dd[:, 5] > 0.5; wall_any = dd[:, 0] > 0.5; wall_hard = dd[:, 3] > 0.5
    on = bool(S.on[k0, b])
    ddx, ddy = o0[0] - s0[0], o0[1] - s0[1]
    c, s_ = np.cos(s0[2]), np.sin(s0[2])
    ray = S.ray(k, s0[0], s0[1], s0[2])
    rows.append(dict(b=b, k0=k0, k1=k1, tid=k, on=on, dx=ddx * c + ddy * s_, dy=-ddx * s_ + ddy * c, dist=float(np.hypot(ddx, ddy)),
                     v=float(s0[3]), vo=float(o0[3]), dyaw=float(np.arctan2(np.sin(o0[2] - s0[2]), np.cos(o0[2] - s0[2]))),
                     ray=ray, wall0=float(S.wall_dist(k, s0[None, :2])[0]),
                     car=int(car_hard.any()) * 2 if car_hard.any() else int(car_any.any()),
                     t_car=float(np.argmax(car_any) * dt) if car_any.any() else np.nan,
                     wall=int(wall_hard.any()) * 2 if wall_hard.any() else int(wall_any.any()),
                     t_wall=float(np.argmax(wall_any) * dt) if wall_any.any() else np.nan,
                     kind=S.kinds[int(S.kind[k0, b])], scale=float(S.scale[k0, b])))
R_on = [r for r in rows if r["on"]]
n_on = len(R_on)
print(f"[출발] 상대 차가 있는 출발 {n_on}번: 2 s 안 상대 차 접촉 {sum(r['car'] > 0 for r in R_on)}번({sum(r['car'] > 0 for r in R_on) / max(n_on, 1) * 100:.1f} %), "
      f"그중 세게 {sum(r['car'] == 2 for r in R_on)}번 | 출발 때 두 차 사이 거리 중앙 {np.median([r['dist'] for r in R_on]):.2f} m "
      f"(5 % {np.percentile([r['dist'] for r in R_on], 5):.2f}, 최소 {min(r['dist'] for r in R_on):.2f})", flush=True)
DB = (1.0, 2.0, 3.0, 4.5, 6.5)
lab_d = [f"{DB[0]:g} m 안"] + [f"{DB[i]:g}-{DB[i + 1]:g}" for i in range(len(DB) - 1)] + [f"{DB[-1]:g} 넘게"]
dist_tab = []
for i, lab in enumerate(lab_d):
    sel = [r for r in R_on if np.searchsorted(DB, r["dist"]) == i]
    if sel:
        dist_tab.append((lab, len(sel), sum(r["car"] > 0 for r in sel), sum(r["car"] == 2 for r in sel)))
        print(f"   두 차 거리 {lab}: 출발 {len(sel)}번, 2 s 안 접촉 {dist_tab[-1][2]}번({dist_tab[-1][2] / len(sel) * 100:.0f} %), 세게 {dist_tab[-1][3]}번", flush=True)
for nm, f in (("상대 차가 뒤(dx < -0.3 m)", lambda r: r["dx"] < -0.3), ("나란히(|dx| <= 0.3 m)", lambda r: abs(r["dx"]) <= 0.3),
              ("상대 차가 앞(dx > 0.3 m)", lambda r: r["dx"] > 0.3)):
    sel = [r for r in R_on if f(r)]
    if sel:
        print(f"   {nm}: 출발 {len(sel)}번, 2 s 안 접촉 {sum(r['car'] > 0 for r in sel)}번({sum(r['car'] > 0 for r in sel) / len(sel) * 100:.0f} %), "
              f"세게 {sum(r['car'] == 2 for r in sel)}번 | 옆 간격 |dy| 중앙 {np.median([abs(r['dy']) for r in sel]):.2f} m", flush=True)
# side-by-side starts against the rest: hard car contacts per minute over the first 3 s, and the level later in a whole drive
def rate3(sel):
    n = mins = 0.0
    for r in sel:
        k = min(r["k1"] - r["k0"], 30)
        n += float((S.d[r["k0"]:r["k0"] + k, r["b"], 5] > 0.5).sum()); mins += k * dt / 60.0
    return n, n / max(mins, 1e-9)
side = [r for r in R_on if r["dist"] < 1.0]; rest = [r for r in R_on if r["dist"] >= 1.0]
late_on = late & W.on[:-1].astype(bool)
lvl_on = (W.d[..., 5][late_on] > 0.5).sum() / (late_on.sum() * dt / 60.0)
(n_s, r_s), (n_r, r_r) = rate3(side), rate3(rest)
print(f"[출발] 나란히 출발(두 차 1 m 안) {len(side)}번 = 상대 차 있는 출발의 {len(side) / max(n_on, 1) * 100:.0f} %, 옆 간격 중앙 {np.median([abs(r['dy']) for r in side]):.2f} m"
      f"(차 폭 {2 * S.hw:.2f} m -> 차 사이 {np.median([abs(r['dy']) for r in side]) - 2 * S.hw:.2f} m) | 출발 3 s 안 상대 차와 세게 닿음: 나란히 {r_s:.2f}/분({int(n_s)}건), "
      f"그 밖의 출발 {r_r:.2f}/분({int(n_r)}건) | 견줄 값: 통째 주행에서 상대 차가 있는 차의 5 s 뒤 {lvl_on:.2f}/분", flush=True)
# in the side-by-side contacts, who moved sideways toward the other (start frame of our car, up to the contact)
mv_o, mv_p = [], []
for r in side:
    if r["car"] == 0:
        continue
    b, k0 = r["b"], r["k0"]; kc = k0 + int(round(r["t_car"] / dt))
    s0 = S.st[k0, b]; nx, ny = -np.sin(s0[2]), np.cos(s0[2]); sg = np.sign(r["dy"])
    mv_o.append(sg * ((S.st[kc, b, 0] - s0[0]) * nx + (S.st[kc, b, 1] - s0[1]) * ny))
    mv_p.append(-sg * ((S.op[kc, b, 0] - S.op[k0, b, 0]) * nx + (S.op[kc, b, 1] - S.op[k0, b, 1]) * ny))
if mv_o:
    mv_o, mv_p = np.array(mv_o), np.array(mv_p)
    print(f"[출발] 나란히 출발해 닿은 {len(mv_o)}건: 닿을 때까지 상대 쪽으로 옆으로 간 거리 -- 우리 차 중앙 {np.median(mv_o):+.2f} m, 상대 차 중앙 {np.median(mv_p):+.2f} m "
          f"(상대 차가 더 많이 온 것 {int((mv_p > mv_o).sum())}건, 우리 차가 더 많이 간 것 {int((mv_o >= mv_p).sum())}건) | 닿기까지 중앙 {np.median([r['t_car'] for r in side if r['car'] > 0]):.1f} s", flush=True)
# the wall: time to the wall straight ahead at the start speed
ttw = np.array([r["ray"] / max(r["v"], 0.3) for r in rows])
wall2 = np.array([r["wall"] for r in rows])
print(f"[출발] 모든 출발 {n_ep}번: 2 s 안 벽 접촉 {int((wall2 > 0).sum())}번({(wall2 > 0).mean() * 100:.1f} %), 세게 {int((wall2 == 2).sum())}번 | 출발 때 진행 "
      f"방향으로 벽까지 중앙 {np.median([r['ray'] for r in rows]):.1f} m (5 % {np.percentile([r['ray'] for r in rows], 5):.2f} m), 벽까지 시간(거리/속도) "
      f"5 % {np.percentile(ttw, 5):.2f} s", flush=True)
TB = (0.5, 1.0, 2.0, 4.0)
lab_t = [f"{TB[0]:g} s 안"] + [f"{TB[i]:g}-{TB[i + 1]:g} s" for i in range(len(TB) - 1)] + [f"{TB[-1]:g} s 넘게"]
ttw_tab = []
for i, lab in enumerate(lab_t):
    m = np.searchsorted(TB, ttw) == i
    if m.any():
        ttw_tab.append((lab, int(m.sum()), int((wall2[m] > 0).sum()), int((wall2[m] == 2).sum())))
        print(f"   벽까지 시간 {lab}: 출발 {int(m.sum())}번, 2 s 안 벽 접촉 {ttw_tab[-1][2]}번({ttw_tab[-1][2] / m.sum() * 100:.0f} %), 세게 {ttw_tab[-1][3]}번", flush=True)
json.dump(rows, open(os.path.join(out_dir, "starts.json"), "w"), ensure_ascii=False, default=float)

# figure: the start statistics
fig, axs = plt.subplots(1, 3, figsize=(21, 6.6), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
ax = axs[0]
xs = np.arange(NB) * 0.5 + 0.25
for j, (lab, col, mk) in enumerate((("벽", C_WALL, "s"), ("상자", C_BOX, "o"), ("상대 차", C_CAR, "^"))):
    ax.plot(xs, rate[j], color=col, marker=mk, lw=2, ms=7, label=lab)
    ax.axhline(late_rate[j], color=col, ls=":", lw=1.3)
ax.set_xlabel("에피소드 시작 뒤 시각 [s]"); ax.set_ylabel("세게 닿음 [/분]"); ax.set_ylim(bottom=0)
ax.set_title(f"출발 뒤 시각별 세게 닿음 (출발 {n_ep}번)\n점선 = 통째 주행의 5 s 뒤 수준", fontsize=12)
ax.legend(frameon=False); ax.grid(alpha=0.25)
ax = axs[1]
for val, col, lab, sz, zo in ((0, "#bbbbbb", "2 s 안 접촉 없음", 9, 1), (1, "#ff9f1c", "살짝 닿음", 34, 2), (2, C_CAR, "세게 닿음", 46, 3)):
    sel = [r for r in R_on if r["car"] == val]
    ax.scatter([r["dy"] for r in sel], [r["dx"] for r in sel], s=sz, c=col, label=f"{lab} ({len(sel)})", zorder=zo, edgecolors="none" if val == 0 else "k", linewidths=0.4)
ax.add_patch(Polygon(np.array([[-S.hw, -S.hl], [S.hw, -S.hl], [S.hw, S.hl], [-S.hw, S.hl]]), closed=True, fc="tab:blue", ec="k", zorder=4))
ax.set_xlim(5.5, -5.5); ax.set_ylim(-7, 7); ax.set_aspect("equal")
ax.set_xlabel("왼쪽 <- 옆 [m] -> 오른쪽"); ax.set_ylabel("뒤 <- 앞뒤 [m] -> 앞")
ax.set_title("출발 때 상대 차 자리 (가운데 파랑 = 우리 차, 위가 앞)\n색 = 출발 2 s 안에 닿았나", fontsize=12)
ax.legend(frameon=True, fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
ax = axs[2]
x_ = np.arange(len(dist_tab))
top = max(a[2] / a[1] * 100 for a in dist_tab)
ax.bar(x_, [a[2] / a[1] * 100 for a in dist_tab], color="#ff9f1c", width=0.62, label="닿음(전체)")
ax.bar(x_, [a[3] / a[1] * 100 for a in dist_tab], color=C_CAR, width=0.62, label="그중 세게")
for i, a in enumerate(dist_tab):
    ax.text(i, a[2] / a[1] * 100 + top * 0.02, f"{a[2]}/{a[1]}", ha="center", fontsize=10)
ax.set_ylim(0, top * 1.15)
ax.set_xticks(x_); ax.set_xticklabels([a[0] for a in dist_tab]); ax.set_xlabel("출발 때 두 차 사이 거리 [m]"); ax.set_ylabel("출발 2 s 안 상대 차 접촉 [%]")
ax.set_title("출발 때 두 차 거리별 접촉 비율\n(1 m 안 = 나란히 출발)", fontsize=12); ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
fig.suptitle(f"{S.name} · 학습 세계 · 학습 조건 -- 출발 직후 접촉", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(os.path.join(out_dir, "start_stats.png"), dpi=75); plt.close(fig)


def start_sheet(sel, key, title, fn, zoom=3.0):
    fig, axs = plt.subplots(3, 4, figsize=(20, 15.6))
    for ax in axs.flat:
        ax.set_axis_off()
    for ax, r in zip(axs.flat, sel[:12]):
        ax.set_axis_on()
        b, k0 = r["b"], r["k0"]
        kc = k0 + int(round(r[key] / dt))
        S.draw_map(ax, r["tid"])
        for q in S.boxes(k0, b):
            ax.fill(q[:, 0], q[:, 1], color="saddlebrown", alpha=0.9, lw=0, zorder=2)
        tr = S.st[k0:kc + 2, b]
        ax.scatter(tr[:, 0], tr[:, 1], c=np.linspace(0, 1, len(tr)), cmap="Blues", vmin=-0.4, vmax=1, s=14, zorder=3)
        ax.add_patch(Polygon(S.body(*S.st[k0, b, :3]), closed=True, fc="none", ec="tab:blue", lw=2, zorder=5))
        ax.add_patch(Polygon(S.body(*S.st[kc + 1, b, :3]), closed=True, fc="tab:blue", ec="k", zorder=6))
        if r["on"]:
            to = S.op[k0:kc + 2, b]
            ax.scatter(to[:, 0], to[:, 1], c=np.linspace(0, 1, len(to)), cmap="Reds", vmin=-0.4, vmax=1, s=14, zorder=3)
            ax.add_patch(Polygon(S.body(*S.op[k0, b, :3]), closed=True, fc="none", ec="tab:red", lw=2, zorder=5))
            ax.add_patch(Polygon(S.body(*S.op[kc + 1, b, :3]), closed=True, fc="tab:red", ec="k", zorder=6))
        p0 = np.concatenate([S.st[k0, b, None, :2], S.pl[k0, b]], 0)
        ax.plot(p0[:, 0], p0[:, 1], "-", color="k", lw=1.6, marker=".", ms=4, zorder=7)
        if kc > k0:
            p1 = np.concatenate([S.st[kc, b, None, :2], S.pl[kc, b]], 0)
            ax.plot(p1[:, 0], p1[:, 1], "-", color="deeppink", lw=1.6, marker=".", ms=4, zorder=7)
        h = S.held[kc, b]
        if h[0] > 0:
            ax.plot([h[1]], [h[2]], marker="*", color="gold", mec="k", ms=17, zorder=9)
        cx, cy = 0.5 * (S.st[k0, b, 0] + S.st[kc + 1, b, 0]), 0.5 * (S.st[k0, b, 1] + S.st[kc + 1, b, 1])
        ax.set_xlim(cx - zoom, cx + zoom); ax.set_ylim(cy - zoom, cy + zoom)
        hard = r["car" if key == "t_car" else "wall"] == 2
        t1 = (f"출발 {r[key]:.1f} s 뒤 {'세게 ' if hard else '살짝 '}닿음 · 출발 속도 우리 {r['v']:.1f}" + (f" / 상대 {r['vo']:.1f} m/s" if r["on"] else " m/s"))
        t2 = (f"출발 때 상대 차: 앞 {r['dx']:+.2f} m, 옆 {r['dy']:+.2f} m · {r['kind']} x{r['scale']:.2f}" if key == "t_car"
              else f"출발 때 벽까지 {r['ray']:.1f} m ({r['ray'] / max(r['v'], 0.3):.1f} s) · 벽과 옆 거리 {r['wall0']:.2f} m")
        ax.set_title(f"{t1}\n{t2} · {trk_name(r['tid'])}", fontsize=10)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.965], h_pad=3.0); plt.savefig(os.path.join(out_dir, fn), dpi=70); plt.close(fig)


LEG = "테두리만 = 출발 자세, 채움 = 닿는 순간, 점 = 자취 · 파랑 = 우리 차, 빨강 = 상대 차, 갈색 = 상자, 검은 선 = 첫 plan, 분홍 선 = 닿는 결정의 plan, 별 = 닿은 점"
sel = sorted([r for r in R_on if r["car"] > 0], key=lambda r: (-r["car"], r["t_car"]))
start_sheet(sel, "t_car", f"출발 2 s 안 상대 차 접촉 (세게 닿은 것 먼저, 이른 순) -- {LEG}", "start_car.png")
sel = sorted([r for r in rows if r["wall"] > 0], key=lambda r: (-r["wall"], r["t_wall"]))
start_sheet(sel, "t_wall", f"출발 2 s 안 벽 접촉 (세게 닿은 것 먼저, 이른 순) -- {LEG}", "start_wall.png", zoom=3.5)

# ---------------------------------------------------------------------------------------------- (나) 벽에 몰린 자리
ev = []
for t, b in zip(*np.nonzero(W.d[..., 3] > 0.5)):
    if (t - W.k0[t, b]) * dt < 3.0:
        continue
    h = W.held[t, b]
    x, y = (h[1], h[2]) if int(h[0]) == 1 else (W.st[t + 1, b, 0], W.st[t + 1, b, 1])
    ev.append(dict(t=int(t), b=int(b), tid=int(W.tid[t, b]), x=float(x), y=float(y), v1=float(W.st[max(t - 10, 0), b, 3]),
                   v0=float(W.st[t, b, 3]), box=bool((W.sid[t, b] >= 0).any()), on=bool(W.on[t, b]),
                   d_opp=float(np.hypot(*(W.op[t, b, :2] - W.st[t, b, :2]))) if W.on[t, b] else np.inf))
print(f"[벽] 통째 기록, 출발 3 s 뒤 벽에 세게 닿은 건 {len(ev)}건", flush=True)
# which of the four cases the car was in, and whether the boxes had been taken away
box_now = (W.sid[:-1] >= 0).any(2); opp_now = W.on[:-1].astype(bool)
had_box = (W.sid[0] >= 0).any(1)
case = np.where(box_now, np.where(opp_now, 3, 2), np.where(opp_now, 1, 0))
CASES = ("빈 트랙", "상대 차만", "상자만", "상자 + 상대 차")
hardw = W.d[..., 3] > 0.5
case_tab = []
print("[벽] 경우별 벽에 세게 닿음 (통째 기록 전체)", flush=True)
for c, lab in enumerate(CASES):
    m = case == c; mins = m.sum() * dt / 60.0
    case_tab.append((lab, mins, int(hardw[m].sum())))
    print(f"   {lab}: {mins:.0f}대·분, 세게 {case_tab[-1][2]}건 = {case_tab[-1][2] / max(mins, 1e-9):.2f}/분", flush=True)
gone = ~box_now & had_box[None]
for lab, m in (("상자를 치운 뒤(빈 트랙 + 상대 차만)", gone), ("처음부터 상자 없음", ~box_now & ~had_box[None])):
    mins = m.sum() * dt / 60.0
    print(f"   {lab}: {mins:.0f}대·분, 세게 {int(hardw[m].sum())}건 = {hardw[m].sum() / max(mins, 1e-9):.2f}/분 (차 {len(set(np.nonzero(hardw & m)[1]))}대)", flush=True)
# the nearest live box to the car at each decision, and the closest it came in the 2 s before
pos = W.st[:-1, :, None, :2]
dbox = np.hypot(W.pose[:-1, :, :, 0] - pos[..., 0], W.pose[:-1, :, :, 1] - pos[..., 1])
dbox = np.where(W.sid[:-1] >= 0, dbox, np.inf).min(2)
near2 = dbox.copy()
for sh in range(1, 21):
    near2[sh:] = np.minimum(near2[sh:], dbox[:-sh])
for e in ev:
    e["box2s"] = float(near2[e["t"], e["b"]]); e["case"] = int(case[e["t"], e["b"]])
# the usual path: where the cars drive on each track when it has no boxes (cases 빈 트랙 / 상대 차만, above 2 m/s, no wall contact
# within 1 s) -- how far off it, and how far off its heading, was the car 1 s and 0.5 s before the hit
from scipy.spatial import cKDTree
usual = {}
okm = ~box_now & (W.st[:-1, :, 3] > 2.0)
wany = W.d[..., 0] > 0.5
for sh in range(1, 11):
    okm[sh:] &= ~wany[:-sh]; okm[:-sh] &= ~wany[sh:]
for k in np.unique(W.tid[0]):
    m = okm & (W.tid[:-1] == k)
    if m.sum() >= 200:
        usual[int(k)] = (cKDTree(W.st[:-1][m][:, :2]), W.st[:-1][m][:, 2])


def off_usual(k, xy, yaw):
    """[m] to the nearest no-box sample of this track, [rad] |heading difference| to it"""
    if int(k) not in usual:
        return np.full(len(xy), np.nan), np.full(len(xy), np.nan)
    tree, yw = usual[int(k)]
    dd_, ii_ = tree.query(xy)
    return dd_, np.abs(np.arctan2(np.sin(yaw - yw[ii_]), np.cos(yaw - yw[ii_])))


for e in ev:
    for back, nm in ((10, "1"), (5, "05")):
        tk = max(e["t"] - back, 0)
        a_, b_ = off_usual(e["tid"], W.st[tk, e["b"], None, :2], W.st[tk, e["b"], None, 2])
        e["off" + nm], e["dyaw" + nm] = float(a_[0]), float(b_[0])
base_off, base_dy = [], []
for k in usual:
    m = box_now & (W.tid[:-1] == k) & (W.st[:-1, :, 3] > 2.0)
    a_, b_ = off_usual(k, W.st[:-1][m][:, :2], W.st[:-1][m][:, 2])
    base_off.append(a_); base_dy.append(b_)
base_off, base_dy = np.concatenate(base_off), np.concatenate(base_dy)
# was the box the car had just passed ON the usual path (within 0.5 m of it)? against all the boxes of the box races
on_line_all = []
for b in range(W.B):
    k = int(W.tid[0, b])
    if k in usual and had_box[b]:
        live = W.sid[0, b] >= 0
        on_line_all.append(usual[k][0].query(W.pose[0, b][live][:, :2])[0] < 0.5)
on_line_all = np.concatenate(on_line_all) if on_line_all else np.zeros(0, bool)
for e in ev:
    e["box_on_line"] = None
    if e["case"] >= 2 and e["tid"] in usual:
        t, b = e["t"], e["b"]
        live = W.sid[t, b] >= 0
        a0 = max(t - 20, 0)
        dd_ = np.hypot(W.pose[t, b][live][:, None, 0] - W.st[a0:t + 1, b, 0][None], W.pose[t, b][live][:, None, 1] - W.st[a0:t + 1, b, 1][None]).min(1)
        j = int(np.argmin(dd_))
        e["box_on_line"] = bool(usual[e["tid"]][0].query(W.pose[t, b][live][j, :2])[0] < 0.5)
bl = [e["box_on_line"] for e in ev if e["box_on_line"] is not None]
print(f"[벽] 닿기 2 s 안에 가장 가까이 지난 상자가 다니는 길 위(0.5 m 안)에 있던 것 {sum(bl)}/{len(bl)}건 | 견줄 값: 상자 경주의 모든 상자 중 길 위 "
      f"{on_line_all.mean() * 100:.0f} %", flush=True)
# the plan's points inside a wall, by how far ahead: all decisions against the decisions 0.5 s before a hard wall hit
inw = np.zeros(W.pl.shape[:3], bool)
for k in np.unique(W.tid[0]):
    cols = np.nonzero(W.tid[0] == k)[0]
    pts = W.pl[:, cols].reshape(-1, 2).astype(float)
    inw[:, cols] = (W.wall_dist(int(k), pts) <= 0.0).reshape(W.T, len(cols), -1)
mv = W.st[:-1, :, 3] > 2.0
e5 = np.array([inw[max(e["t"] - 5, 0), e["b"]] for e in ev]); e10 = np.array([inw[max(e["t"] - 10, 0), e["b"]] for e in ev])
print("[벽] plan의 점이 벽 안에 있는 비율, 몇 번째 점(0.1 s 간격)인가별 -- 달리는 모든 결정 / 닿기 1 s 전 / 닿기 0.5 s 전", flush=True)
for i in (0, 2, 4, 6, 9):
    print(f"   {i + 1}번째 점({(i + 1) * 0.1:.1f} s 뒤): {inw[..., i][mv].mean() * 100:.1f} % / {e10[:, i].mean() * 100:.0f} % / {e5[:, i].mean() * 100:.0f} %", flush=True)
print(f"   어느 점이든: {inw.any(2)[mv].mean() * 100:.1f} % / {e10.any(1).mean() * 100:.0f} % / {e5.any(1).mean() * 100:.0f} %", flush=True)
in_box = [e for e in ev if e["case"] >= 2]
eo = np.array([e["off1"] for e in in_box]); ey = np.array([e["dyaw05"] for e in in_box]); eo5 = np.array([e["off05"] for e in in_box])
print(f"[벽] 상자 없는 트랙에서 다니는 길(같은 트랙, 상자 없는 차들의 자취)에서 벗어난 거리 -- 상자 경주의 벽 접촉: 1 s 전 중앙 {np.nanmedian(eo):.2f} m, "
      f"0.5 s 전 {np.nanmedian(eo5):.2f} m (0.5 m 넘게 벗어난 것 1 s 전 {int(np.nansum(eo > 0.5))}/{int(np.isfinite(eo).sum())}건) | 상자 경주의 아무 순간: "
      f"중앙 {np.median(base_off):.2f} m, 0.5 m 넘게 {np.mean(base_off > 0.5) * 100:.0f} %", flush=True)
print(f"[벽] 그 길의 방향과 차 방향의 차이, 0.5 s 전: 벽 접촉 중앙 {np.degrees(np.nanmedian(ey)):.0f} deg (20 deg 넘게 {int(np.nansum(ey > np.radians(20)))}/"
      f"{int(np.isfinite(ey).sum())}건) | 상자 경주의 아무 순간: 중앙 {np.degrees(np.median(base_dy)):.0f} deg, 20 deg 넘게 {np.mean(base_dy > np.radians(20)) * 100:.0f} %", flush=True)
base = float((near2[box_now] < 1.2).mean())
print(f"[벽] 상자가 있는 경주의 벽 접촉 {len(in_box)}건 중 닿기 2 s 안에 상자 1.2 m 안을 지난 것 {sum(e['box2s'] < 1.2 for e in in_box)}건"
      f"({sum(e['box2s'] < 1.2 for e in in_box) / max(len(in_box), 1) * 100:.0f} %) | 견줄 값: 상자 있는 경주의 아무 순간 {base * 100:.0f} %", flush=True)
# greedy clusters: same track, within 1.5 m
left = list(range(len(ev))); cl = []
while left:
    best = max(left, key=lambda i: sum(ev[j]["tid"] == ev[i]["tid"] and np.hypot(ev[j]["x"] - ev[i]["x"], ev[j]["y"] - ev[i]["y"]) < 1.5 for j in left))
    mem = [j for j in left if ev[j]["tid"] == ev[best]["tid"] and np.hypot(ev[j]["x"] - ev[best]["x"], ev[j]["y"] - ev[best]["y"]) < 1.5]
    cl.append(mem); left = [j for j in left if j not in mem]
cl.sort(key=len, reverse=True)
in_multi = sum(len(m) for m in cl if len(m) >= 2)
print(f"[벽] 같은 트랙 1.5 m 안에 2건 넘게 모인 자리 {sum(len(m) >= 2 for m in cl)}곳에 {in_multi}건({in_multi / max(len(ev), 1) * 100:.0f} %), "
      f"그중 같은 차가 되풀이한 건 {sum(len(m) - len({ev[j]['b'] for j in m}) for m in cl)}건", flush=True)
fig, axs = plt.subplots(2, 4, figsize=(24, 12.6))
for ax in axs.flat:
    ax.set_axis_off()
cm = plt.get_cmap("viridis")
for ax, mem in zip(axs.flat, cl[:8]):
    ax.set_axis_on()
    k = ev[mem[0]]["tid"]
    cx, cy = np.mean([ev[j]["x"] for j in mem]), np.mean([ev[j]["y"] for j in mem])
    W.draw_map(ax, k)
    # the cars on this track that passed within 3 m of the spot without a wall contact within 2 s either side
    hit_keys = {(ev[j]["b"], ev[j]["t"]) for j in mem}
    v_pass = []
    n_pass = 0
    for b in np.nonzero(W.tid[0] == k)[0]:
        dd = np.hypot(W.st[:-1, b, 0] - cx, W.st[:-1, b, 1] - cy)
        near = dd < 3.0
        t = 0
        while t < W.T:
            if not near[t]:
                t += 1; continue
            t1 = t
            while t1 < W.T and near[t1]:
                t1 += 1
            tc = t + int(np.argmin(dd[t:t1]))
            lo, hi = max(tc - 20, 0), min(tc + 20, W.T)
            clean = (W.d[lo:hi, b, 0] < 0.5).all() and not W.rs[lo:hi, b].any() and W.tid[tc, b] == k and W.st[tc, b, 3] > 1.5
            if clean and dd[tc] < 2.0:
                a0 = max(tc - 25, 0)
                ax.plot(W.st[a0:tc + 10, b, 0], W.st[a0:tc + 10, b, 1], "-", color="#8a8a8a", lw=0.8, alpha=0.55, zorder=2)
                v_pass.append(W.st[max(tc - 10, 0), b, 3]); n_pass += 1
            t = t1
    for j in mem:
        e = ev[j]; t, b = e["t"], e["b"]
        a0 = max(t - 40, W.k0[t, b])
        tr = W.st[a0:t + 2, b]
        ax.scatter(tr[:, 0], tr[:, 1], c=tr[:, 3], cmap=cm, vmin=0, vmax=8, s=16, zorder=4)
        ax.add_patch(Polygon(W.body(*W.st[t + 1, b, :3]), closed=True, fc="tab:blue", ec="k", zorder=6))
        for q in W.boxes(t, b):
            ax.fill(q[:, 0], q[:, 1], color="saddlebrown", alpha=0.75, lw=0, zorder=3)
        if e["on"] and e["d_opp"] < 4.0:
            ax.add_patch(Polygon(W.body(*W.op[t + 1, b, :3]), closed=True, fc="tab:red", ec="k", zorder=6))
        ax.plot([e["x"]], [e["y"]], marker="*", color="gold", mec="k", ms=16, zorder=9)
    ax.set_xlim(cx - 6, cx + 6); ax.set_ylim(cy - 6, cy + 6)
    cars = len({ev[j]["b"] for j in mem})
    vp = f"{np.median(v_pass):.1f} m/s ({n_pass}번)" if v_pass else "없음"
    ax.set_title(f"{trk_name(k)} ({cx:.1f}, {cy:.1f}) · 세게 {len(mem)}건(차 {cars}대)\n닿기 1 s 전 속도 {np.median([ev[j]['v1'] for j in mem]):.1f} m/s\n"
                 f"닿지 않고 지나간 차(회색) 1 s 전 {vp}", fontsize=10)
    print(f"   {trk_name(k)} ({cx:.1f}, {cy:.1f}): 세게 {len(mem)}건, 차 {cars}대, 1 s 전 속도 {np.median([ev[j]['v1'] for j in mem]):.1f} m/s, 상자 있던 건 "
          f"{sum(ev[j]['box'] for j in mem)}, 상대 차 2 m 안 {sum(ev[j]['d_opp'] < 2.0 for j in mem)} | 닿지 않고 지나감 {n_pass}번, 그때 1 s 전 속도 {vp}", flush=True)
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 8)); sm.set_array([])
fig.colorbar(sm, ax=axs, fraction=0.012, pad=0.01, label="닿은 차의 속도 [m/s]")
fig.suptitle(f"{W.name} · 벽에 세게 닿은 건이 모인 자리(많은 순) -- 색 점 = 닿은 차의 4 s 자취, 회색 선 = 같은 자리를 닿지 않고 지나간 차, 갈색 = 닿은 차의 상자, 별 = 닿은 점",
             fontsize=13)
plt.savefig(os.path.join(out_dir, "wall_spots.png"), dpi=70, bbox_inches="tight"); plt.close(fig)
json.dump(ev, open(os.path.join(out_dir, "wall_events.json"), "w"), ensure_ascii=False, default=float)

# every scene: the last 3 s, the plans 1.0 / 0.5 / 0 s before
order = sorted(range(len(ev)), key=lambda i: (ev[i]["case"] < 2, ev[i]["box2s"]))
for page in range(0, min(len(order), 24), 12):
    fig, axs = plt.subplots(3, 4, figsize=(20, 15.9))
    for ax in axs.flat:
        ax.set_axis_off()
    for ax, i in zip(axs.flat, order[page:page + 12]):
        ax.set_axis_on()
        e = ev[i]; t, b, k = e["t"], e["b"], e["tid"]
        W.draw_map(ax, k)
        a0 = max(t - 30, W.k0[t, b])
        if k in usual:
            pts_u = usual[k][0].data
            mu = (np.abs(pts_u[:, 0] - e["x"]) < 5) & (np.abs(pts_u[:, 1] - e["y"]) < 5)
            ax.scatter(pts_u[mu, 0], pts_u[mu, 1], s=5, c="#9ecae1", alpha=0.6, zorder=1, linewidths=0)
        for q in W.boxes(t, b):
            ax.fill(q[:, 0], q[:, 1], color="saddlebrown", alpha=0.9, lw=0, zorder=2)
        tr = W.st[a0:t + 2, b]
        ax.scatter(tr[:, 0], tr[:, 1], c=tr[:, 3], cmap=cm, vmin=0, vmax=8, s=18, zorder=4)
        if e["on"]:
            to = W.op[a0:t + 2, b]
            ax.plot(to[:, 0], to[:, 1], ":", color="tab:red", lw=1.4, zorder=3)
            ax.add_patch(Polygon(W.body(*W.op[t + 1, b, :3]), closed=True, fc="tab:red", ec="k", zorder=6))
        for back, col in ((10, "#f4a6c9"), (5, "#e75aa0"), (0, "#b0005a")):
            tk = t - back
            if tk < a0:
                continue
            pp = np.concatenate([W.st[tk, b, None, :2], W.pl[tk, b]], 0)
            ax.plot(pp[:, 0], pp[:, 1], "-", color=col, lw=1.7, marker=".", ms=4, zorder=7)
            ax.add_patch(Polygon(W.body(*W.st[tk, b, :3]), closed=True, fc="none", ec=col, lw=1.2, zorder=5))
        ax.add_patch(Polygon(W.body(*W.st[t + 1, b, :3]), closed=True, fc="tab:blue", ec="k", zorder=6))
        ax.plot([e["x"]], [e["y"]], marker="*", color="gold", mec="k", ms=17, zorder=9)
        ax.set_xlim(e["x"] - 4.5, e["x"] + 4.5); ax.set_ylim(e["y"] - 4.5, e["y"] + 4.5)
        bx = f"2 s 안 가장 가까웠던 상자 {e['box2s']:.1f} m" if np.isfinite(e["box2s"]) else ("상자 치운 뒤" if had_box[b] else "상자 없음")
        ax.set_title(f"{trk_name(k)} · {CASES[e['case']]} · {(t - W.k0[t, b]) * dt:.1f} s · 다니는 길에서 1 s 전 {e['off1']:.1f} m\n속도 1 s 전 {e['v1']:.1f} -> 닿을 때 {e['v0']:.1f} m/s · {bx}", fontsize=10)
    fig.suptitle(f"{W.name} · 벽에 세게 닿은 장면 (상자 가까이 지난 것 먼저) -- 색 점 = 3 s 자취(속도), 분홍 선 = 1.0 / 0.5 / 0 s 전 plan(옅음 -> 짙음), "
                 f"갈색 = 상자, 빨강 = 상대 차, 별 = 닿은 점, 하늘색 점 = 상자 없는 차들이 다닌 자리", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.965], h_pad=3.0); plt.savefig(os.path.join(out_dir, f"wall_scenes_{page // 12 + 1}.png"), dpi=70); plt.close(fig)
print(f"그림 {out_dir}/start_stats.png start_car.png start_wall.png wall_spots.png", flush=True)
