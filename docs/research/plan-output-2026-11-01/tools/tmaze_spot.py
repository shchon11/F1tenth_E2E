"""레이싱 T-미로(11-26, RQ3/4/6)의 상자 자리를 한 번 정해 world json에 적음. policy와 상관없이 고정된 환경이 되게, 기준 policy
(scan31b2 u600)의 exam3 기록에서 한 번만 계산함.
  1. 자리: U턴 입구(ANCHOR 근처 RADIUS m 안)의 '길 위의 상자'(exam_boxes.py: 빈 트랙 길이 상자 중심 0.40 m 안) 가운데 처음 보이는
     때(지나가기 몇 s 전, 통과마다)의 중앙값이 가장 짧은 상자 자리. 상자 모양(catalogue id)·방향은 그 상자 그대로.
  2. 빈 트랙 길: 같은 기록의 혼자·빈 트랙 세션(5 s 뒤, 2 m/s 넘게, 세게 닿은 뒤 3 s 뺌)이 지나간 길에서, 바퀴마다 상자 중심에 가장
     가까운 점(수선의 발)과 그때 진행 방향. 진행 방향 평균의 왼쪽 법선 n, 수선의 발들의 n 방향 위치 중앙값 = 그 자리의 길 위 점 P.
  3. 후보 둘: L = P + 0.15 n (진행 방향 왼쪽), R = P - 0.15 n.
인자: exam3 기록npz [world json, 기본 configs/plan_world_korea26scan_tmaze.json] [그림 png]
EXAM_ROWS_IN=npy: exam_boxes.py의 통과 행(EXAM_ROWS로 저장한 것)을 다시 씀 (없으면 exam_boxes.py를 돌림)."""
import os, sys, json, subprocess, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rec_lib import Rec, plt

ANCHOR = (-3.4, -6.9)        # the U-turn entry the user named (fast, blind)
RADIUS = 1.5                 # [m] the boxes counted as "at" it
OFFSET = 0.15                # [m] L / R either side of the line
P_LRN = [0.4, 0.4, 0.2]      # P(L), P(R), P(none) per episode
QUAL = os.path.normpath(os.path.join(HERE, "../../../../f1sim/f1sim/learn/configs/plan_world_korea26scan_qual.json"))
OUT = os.path.normpath(os.path.join(HERE, "../../../../f1sim/f1sim/learn/configs/plan_world_korea26scan_tmaze.json"))
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)

rec_path = sys.argv[1]
world_out = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else OUT
png = sys.argv[3] if len(sys.argv) > 3 else ""
rows_in = os.environ.get("EXAM_ROWS_IN", "")
if not rows_in:
    rows_in = os.path.join(tempfile.mkdtemp(), "rows.npy")
    subprocess.run([sys.executable, os.path.join(HERE, "exam_boxes.py"), rec_path], check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, EXAM_ROWS=rows_in))
rows = np.load(rows_in)
R = Rec(rec_path); z = R.z; dt = R.dt
st = R.st.astype(float); xy = st[:, :, :2]; yaw = st[:, :, 2]; V = np.abs(st[:, :, 3])
T = R.T

# 1. the spot
on_line = rows[:, IGAP] < 0.40
near = np.hypot(rows[:, IBX] - ANCHOR[0], rows[:, IBY] - ANCHOR[1]) < RADIUS
groups = {}
for i in np.nonzero(on_line & near)[0]:
    groups.setdefault((round(rows[i, IBX], 3), round(rows[i, IBY], 3)), []).append(i)
if not groups:
    sys.exit(f"{ANCHOR} 둘레 {RADIUS} m 안에 길 위의 상자가 없음")
cand = sorted(groups.items(), key=lambda kv: (np.median(rows[kv[1], ISEE]), np.hypot(kv[0][0] - ANCHOR[0], kv[0][1] - ANCHOR[1])))
print(f"U턴 입구 {ANCHOR} 둘레 {RADIUS} m 안 길 위의 상자 {len(cand)}자리 (처음 보임 중앙값 짧은 순):")
for (bx, by), v in cand:
    v = np.array(v)
    print(f"  ({bx:+.3f}, {by:+.3f}) 통과 {len(v)}번, 처음 보임 중앙 {np.median(rows[v, ISEE]):.2f} s (거리 {np.median(rows[v, ISEED]):.2f} m), "
          f"길에서 {np.median(rows[v, IGAP]):.2f} m, 세게 닿음 {int(rows[v, IHARD].sum())}번")
(bx, by), v_ = cand[0]
box = np.array([bx, by])
d0 = np.where(R.sid[0] >= 0, np.hypot(R.pose[0, ..., 0] - bx, R.pose[0, ..., 1] - by), 1e9)
b_, j_ = np.unravel_index(np.argmin(d0), d0.shape)
sid = int(R.sid[0, b_, j_]); byaw = float(R.pose[0, b_, j_, 2])
see_med = float(np.median(rows[np.array(v_), ISEE]))

# 2. the empty-track line there: per lap of the solo empty-track sessions, the foot of the perpendicular from the box
boxed = (R.sid[0] >= 0).any(-1); opp = R.on[:T].any(0)
hard_any = R.d[:, :, 3:6].sum(-1) > 0
feet, tang = [], []
for b in np.nonzero(~boxed & ~opp)[0]:
    ok = V[:T, b] > 2.0; ok[:int(round(5.0 / dt))] = False
    for t in np.nonzero(hard_any[:, b])[0]:
        ok[t: t + int(round(3.0 / dt))] = False
    ok = ok[:-1] & ok[1:]
    A, Bp = xy[:T - 1, b], xy[1:T, b]
    D = Bp - A; Lq = (D ** 2).sum(1).clip(min=1e-9)
    u = (((box - A) * D).sum(1) / Lq).clip(0.0, 1.0)
    Fp = A + u[:, None] * D
    dist = np.where(ok, np.linalg.norm(Fp - box, axis=1), 1e9)
    # one foot per pass: the nearest segment within each visit (a visit = segments within 2 m, 3 s apart from the next)
    idx = np.nonzero(dist < 2.0)[0]
    if idx.size == 0:
        continue
    cuts = np.nonzero(np.diff(idx) > int(round(3.0 / dt)))[0]
    for seg in np.split(idx, cuts + 1):
        k = seg[np.argmin(dist[seg])]
        if 0.0 < u[k] < 1.0 or dist[k] < 1.0:
            feet.append(Fp[k]); tang.append(D[k] / np.sqrt(Lq[k]))
feet, tang = np.array(feet), np.array(tang)
u_m = tang.mean(0); u_m /= np.linalg.norm(u_m)
n_m = np.array([-u_m[1], u_m[0]])                          # left of the driving direction
lat = (feet - box) @ n_m
lon = (feet - box) @ u_m
P = box + np.median(lat) * n_m
L = P + OFFSET * n_m; Rr = P - OFFSET * n_m
print(f"\n고른 자리: 상자 ({bx:+.3f}, {by:+.3f}), 모양 {sid}, 방향 {byaw:.3f} rad, 처음 보임 중앙 {see_med:.2f} s")
print(f"빈 트랙 길: 바퀴 {len(feet)}개, 진행 방향 ({u_m[0]:+.3f}, {u_m[1]:+.3f}), 상자에서 길까지 왼쪽으로 {np.median(lat):+.3f} m "
      f"(바퀴 사이 표준편차 {lat.std():.3f} m, 앞뒤 어긋남 중앙 {np.median(lon):+.3f} m) -> 길 위 점 ({P[0]:+.3f}, {P[1]:+.3f})")
print(f"L = ({L[0]:+.4f}, {L[1]:+.4f})   R = ({Rr[0]:+.4f}, {Rr[1]:+.4f})   (길에서 각각 왼쪽·오른쪽 {OFFSET} m)")

# 3. the world
w = json.load(open(world_out if os.path.exists(world_out) else QUAL))
w.update({"name": "plan_world_korea26scan_tmaze", "warps": 0, "episode_s": 32.0, "prop_bank": "", "prop_count": "fixed",
          "prop_fraction": P_LRN[0] + P_LRN[1], "removal_p": 0.0, "opp_present": 0.0, "stage": "props", "exam_name": "exam_tmaze",
          "exam_bank": ""})
w["tmaze"] = {"L": [round(float(L[0]), 4), round(float(L[1]), 4), round(byaw, 4), sid],
              "R": [round(float(Rr[0]), 4), round(float(Rr[1]), 4), round(byaw, 4), sid],
              "p": P_LRN,
              "source": {"record": os.path.basename(rec_path), "policy": str(z["name"]), "anchor": list(ANCHOR), "radius_m": RADIUS,
                         "box": [bx, by], "see_median_s": round(see_med, 3), "passes": len(v_), "line_point": [round(float(P[0]), 4), round(float(P[1]), 4)],
                         "left_normal": [round(float(n_m[0]), 4), round(float(n_m[1]), 4)], "laps": int(len(feet)),
                         "lat_sd_m": round(float(lat.std()), 4), "offset_m": OFFSET}}
w["_note_tmaze"] = ("11-26 racing T-maze (RQ3/4/6, docs/.../analysis/rq_plan_2026-11-26.md): one map, no variants; exactly one real box at a "
                    "blind U-turn entry, at L or R (0.15 m either side of scan31b2 u600's empty-track line there; tools/tmaze_spot.py from "
                    "its exam3 record: the on-line box with the shortest median first-sight time) or none, P = (0.4, 0.4, 0.2) per "
                    "episode, never removed; 32 s episodes (~4 laps), no opponents; train with --scene-replay 0")
with open(world_out, "w") as fh:
    json.dump(w, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(f"저장 {world_out}")

if png:
    fig, ax = plt.subplots(figsize=(8, 8))
    R.draw_map(ax, int(R.tid[0, 0]))
    for b in np.nonzero(~boxed & ~opp)[0][:24]:
        ax.plot(xy[50:, b, 0], xy[50:, b, 1], "-", lw=0.3, color="tab:blue", alpha=0.35)
    fp = z[f"fp_{sid}"].astype(float)
    for (cx, cy), col in ((L, "tab:green"), (Rr, "tab:red"), (box, "0.5")):
        c, s = np.cos(byaw), np.sin(byaw)
        q = np.stack([cx + fp[:, 0] * c - fp[:, 1] * s, cy + fp[:, 0] * s + fp[:, 1] * c], 1)
        ax.fill(q[:, 0], q[:, 1], color=col, alpha=0.6 if col != "0.5" else 0.3)
    ax.plot(feet[:, 0], feet[:, 1], ".", ms=2, color="k")
    ax.plot(*P, "x", color="k", ms=10)
    ax.annotate("", xy=P + 0.6 * u_m, xytext=P, arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(box[0] - 2.5, box[0] + 2.5); ax.set_ylim(box[1] - 2.5, box[1] + 2.5)
    ax.set_title(f"T-미로 상자: L(초록) R(빨강), 원래 시험 상자(회색), 처음 보임 {see_med:.2f} s")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    print(f"그림 {png}")
