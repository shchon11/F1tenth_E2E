"""레이싱 T-미로(11-26, RQ3/4/6)의 상자 자리를 한 번 정해 world json에 적음. policy와 상관없는 고정 환경이 되게, 기준 policy(scan31b2
u600)의 exam3 기록에서 한 번만 계산함.
(11-26 둘째 판, coordinator 결정: 첫 판의 '길에서 좌우 0.15 m'는 U턴 바닥에서 둘 다 바깥으로 지나가게 만들어 '어느 쪽'이 아니라 '얼마나
바깥'이 됐음 -> 두 자리가 서로 반대쪽을 강제하게)
  1. 기준 길: 같은 기록의 혼자·빈 트랙 세션(3 s 뒤, 0.5 m/s 넘게)이 달린 길, 트랙 따라 0.1 m마다 중앙값(tmaze_measure.ref_line), 그
     자리 속도의 중앙값.
  2. 후보: 길 위 0.2 m마다 그 점에 상자가 있다고 보고
     - 처음 보임: 그 길을 그 속도로 달리는 차의 스캐너(진행 방향, LIDAR_MOUNT 앞)에서 상자 중심이 보이기 시작한 때 -> 지나갈 때까지
       시간 (exam_boxes.py 규칙: 시야각 135도, 10 m, 상자 중심 0.25 m 앞까지 벽(EDT <= 0, 덕트 호스 포함)에 안 가림)
     - 속도: 그 자리 중앙값
     - 폭: 길의 법선 따라 양쪽 벽까지 (EDT 지도)
     조건: 처음 보임 <= SIGHT_MAX s, 속도 >= V_MIN m/s, 폭 >= W_MIN m.
  3. 자리 둘 (catalogue BOX를 N_BOX개 나란히, 긴 변을 법선 방향으로): L은 왼쪽 벽과 GAP_NARROW m(차 0.31 m가 못 지남), R은 오른쪽
     벽과 GAP_NARROW m. 틈 = 상자 외곽선 점에서 그쪽(법선 +-60도) 벽까지 가장 가까운 거리. 넓은 쪽 틈 >= GAP_WIDE.
     '공통 통로' = L과 R이 함께 비워 두는 가운데 폭; 차 폭보다 좁아야 L이면 오른쪽, R이면 왼쪽으로만 지나감(서로 배타).
     상자 1개로는 이 지도 어디서도 배타가 안 됨(길 따라 가장 좁은 폭 1.73 m) -> 기본 N_BOX = 2.
     조건을 다 맞는 자리 가운데 (가) 배타인 자리를 먼저, (나) 처음 보임이 짧은 순, (다) 빠른 순.
인자: exam3 기록npz [world json, 기본 configs/plan_world_korea26scan_tmaze.json] [그림 png]"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rec_lib import Rec, plt, along_track
from tmaze_measure import ref_line, project, BIN

SIGHT_MAX = 0.7              # [s] a box on the line first visible at most this long before the pass
V_MIN = 4.5                  # [m/s] the reference policy's speed there
W_MIN = 1.3                  # [m] free width wall to wall
GAP_NARROW = 0.15            # [m] box <-> the near wall: the car (0.31 m) cannot pass (< 0.25 asked)
GAP_WIDE = 0.6               # [m] box <-> the far wall: at least this
BOX = 38                     # catalogue id: the largest real box, 0.36 x 0.30 m
# boxes per placement, side by side from the wall. 2: one real box cannot force opposite sides on this map -- the corridor is
# >= 1.73 m everywhere along the line, a box (<= 0.36 m) with a < 0.25 m gap covers <= 0.61 m from its wall, and the middle
# left free by both L and R (>= 0.5 m) passes the car whichever box is there (TMAZE_BOXES=1: that variant)
N_BOX = int(os.environ.get("TMAZE_BOXES", "2"))
GAP_BOX = 0.01               # [m] between two boxes of one placement
HALF_CAR = 0.155             # [m] half the car's width
FOV, RMAX = math.radians(135.0), 10.0
P_LRN = [0.4, 0.4, 0.2]
OUT = os.path.normpath(os.path.join(HERE, "../../../../f1sim/f1sim/learn/configs/plan_world_korea26scan_tmaze.json"))
QUAL = os.path.normpath(os.path.join(HERE, "../../../../f1sim/f1sim/learn/configs/plan_world_korea26scan_qual.json"))

rec_path = sys.argv[1]
world_out = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else OUT
png = sys.argv[3] if len(sys.argv) > 3 else ""
R = Rec(rec_path); z = R.z; T, dt = R.T, R.dt
tk = int(R.tid[0, 0])
mount = float(z["lidar_mount_x"]) if float(z["lidar_mount_x"]) > 0 else 0.297
st = R.st.astype(float); xy = st[:, :, :2]; V = np.abs(st[:, :, 3])
s, lap, L = along_track(z, T + 1)
boxed = (R.sid[0] >= 0).any(-1); opp = R.on[:T].any(0)
ok = np.zeros((T + 1, R.B), bool); ok[int(round(3.0 / dt)):, ~boxed & ~opp] = True
ok &= V > 0.5
P = ref_line(xy[ok], s[ok], np.ones(int(ok.sum()), bool), L)
nb = P.shape[0]
kb = np.floor(s[ok] / BIN).astype(int) % nb
vb = np.array([np.median(V[ok][kb == j]) if (kb == j).sum() >= 3 else np.nan for j in range(nb)])
good = np.isfinite(vb); idx = np.arange(nb)
vb = np.interp(idx, np.concatenate([idx[good] - nb, idx[good], idx[good] + nb]), np.concatenate([vb[good]] * 3))
tan = np.roll(P, -1, 0) - np.roll(P, 1, 0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)                 # left of the driving direction
wall = lambda pts: R.wall_dist(tk, pts)


def ray(p0, d, maxd=3.0, step=0.005):
    """[m] from p0 along unit d to the first wall cell"""
    r = np.arange(0.0, maxd, step)
    hit = wall(p0[None] + r[:, None] * d[None]) <= 0.0
    return float(r[hit.argmax()]) if hit.any() else maxd


def visible(sp, syaw, c):
    """the box centre c in direct view from a scanner at sp heading syaw (exam_boxes.sees)"""
    dx, dy = c[0] - sp[0], c[1] - sp[1]; dist = math.hypot(dx, dy)
    a = (math.atan2(dy, dx) - syaw + math.pi) % (2 * math.pi) - math.pi
    if dist > RMAX or abs(a) > FOV:
        return False
    n = max(int((dist - 0.25) / 0.05), 1)
    f = np.linspace(0.0, max(dist - 0.25, 0.0) / max(dist, 1e-6), n)
    return bool((wall(np.stack([sp[0] + f * dx, sp[1] + f * dy], 1)) > 0.0).all())


def first_sight(j, c):
    """[s] before reaching line bin j that a box at c is first seen by a car driving the line at its speed (the visible
    stretch that ends at the pass), and the distance [m]"""
    t_acc, k = 0.0, j
    if not visible(P[j] + mount * tan[j], math.atan2(tan[j, 1], tan[j, 0]), c):
        for back in range(1, 6):                           # the pass itself may be past the view: the last 0.5 m
            jj = (j - back) % nb
            if visible(P[jj] + mount * tan[jj], math.atan2(tan[jj, 1], tan[jj, 0]), c):
                k = jj; break
        else:
            return 0.0, 0.0
    while True:
        kk = (k - 1) % nb
        if (j - kk) % nb > int(12.0 / BIN) or not visible(P[kk] + mount * tan[kk], math.atan2(tan[kk, 1], tan[kk, 0]), c):
            break
        k = kk
    steps = [(k + i) % nb for i in range((j - k) % nb)]
    return float(sum(BIN / max(vb[q], 0.3) for q in steps)), (j - k) % nb * BIN


def box_outline(cx, cy, yaw, fp, step=0.02):
    pts = []
    ring = np.concatenate([fp, fp[:1]])
    for a, b in zip(ring[:-1], ring[1:]):
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts += [a + (b - a) * i / m for i in range(m)]
    q = np.array(pts); c, s_ = math.cos(yaw), math.sin(yaw)
    return np.stack([cx + q[:, 0] * c - q[:, 1] * s_, cy + q[:, 0] * s_ + q[:, 1] * c], 1)


def fan(p, d, half=math.radians(60.0), k=9):
    """[m] the nearest wall from p within +-half of direction d (the passage beside a box on that side, an oblique wall too)"""
    a0 = math.atan2(d[1], d[0])
    return min(ray(p, np.array([math.cos(a0 + a), math.sin(a0 + a)])) for a in np.linspace(-half, half, k))


def pieces(center, n, sg):
    """the placement's box centres: N_BOX boxes side by side along n, `center` the middle of the row"""
    w = 2.0 * half_across + GAP_BOX
    return [center + (i - 0.5 * (N_BOX - 1)) * w * n for i in range(N_BOX)]


def gaps(center, yaw, fp, n):
    """[m] (left, right) the narrowest gap between the placement's boxes and the wall on each side: from the outline points on
    that side, the nearest wall within +-60 deg of +n / -n"""
    o = np.concatenate([box_outline(c_[0], c_[1], yaw, fp) for c_ in pieces(center, n, 1.0)])
    side = (o - center) @ n
    gl = min(fan(p, n) for p in o[side > 0.0][::3])
    gr = min(fan(p, -n) for p in o[side < 0.0][::3])
    return gl, gr


fp = z[f"fp_{BOX}"].astype(float)
half_across = 0.5 * np.ptp(fp[:, 0])                      # the long side laid across the corridor
span = N_BOX * 2.0 * half_across + (N_BOX - 1) * GAP_BOX     # [m] a placement across the corridor
cand = []
for j in range(0, nb, 2):                                  # every 0.2 m of the line
    if vb[j] < V_MIN:
        continue
    dl, dr = ray(P[j], nrm[j]), ray(P[j], -nrm[j])
    if dl + dr < W_MIN:
        continue
    see, see_d = first_sight(j, P[j])
    if see > SIGHT_MAX:
        continue
    cand.append((j, see, see_d, vb[j], dl, dr))
print(f"조건 (처음 보임 <= {SIGHT_MAX} s, 속도 >= {V_MIN} m/s, 폭 >= {W_MIN} m)을 맞는 길 위 점 {len(cand)}개 (0.2 m마다, 한 바퀴 {L:.1f} m)")

rows = []
for j, see, see_d, v, dl, dr in cand:
    n = nrm[j]; yaw = math.atan2(n[1], n[0])
    out = {}
    for side, sg in (("L", 1.0), ("R", -1.0)):
        # start the box GAP_NARROW + its half across from the near wall, then move it until the measured narrow gap is GAP_NARROW
        near = dl if sg > 0 else dr
        c_off = sg * (near - GAP_NARROW - 0.5 * span)
        for _ in range(6):
            ctr = P[j] + c_off * n
            gl, gr = gaps(ctr, yaw, fp, n)
            g_n = gl if sg > 0 else gr
            c_off += sg * (g_n - GAP_NARROW)
        ctr = P[j] + c_off * n
        gl, gr = gaps(ctr, yaw, fp, n)
        # the box across the line car's body? (box edge nearer the line within the car's half width of it)
        inner = c_off - sg * 0.5 * span
        on_line = (inner < HALF_CAR) if sg > 0 else (inner > -HALF_CAR)
        see_b, _ = first_sight(j, ctr)
        out[side] = dict(ctr=ctr, off=c_off, gl=gl, gr=gr, on_line=bool(on_line), see=see_b, inner=inner)
    okL = out["L"]["gl"] < 0.25 and out["L"]["gr"] >= GAP_WIDE
    okR = out["R"]["gr"] < 0.25 and out["R"]["gl"] >= GAP_WIDE
    # the passage both placements leave free (along the normal at the line point): wider than the car = one path passes both
    common = max(0.0, (out["L"]["inner"]) - (out["R"]["inner"]))
    rows.append(dict(j=j, s=j * BIN, see=see, see_d=see_d, v=v, dl=dl, dr=dr, yaw=yaw, ok=okL and okR, common=common,
                     both=out["L"]["on_line"] and out["R"]["on_line"], **{k + "_" + q: v_ for k, d in out.items() for q, v_ in d.items()}))
ok_rows = [r for r in rows if r["ok"]]
# exclusive (no path through both placements) first, then the shortest first sight, then the faster
ok_rows.sort(key=lambda r: (r["common"] >= 2 * HALF_CAR, r["see"], -r["v"]))
print(f"그 가운데 L·R 틈 조건(좁은 쪽 < 0.25 m, 넓은 쪽 >= {GAP_WIDE} m)까지 맞는 점 {len(ok_rows)}개; 둘 다 길 위 차 몸에 걸리는 점 "
      f"{sum(r['both'] for r in ok_rows)}개")
print(f"한 자리 = 상자 {N_BOX}개 ({span:.2f} m 가로). 두 자리가 함께 남기는 통로(공통)가 차 폭 {2 * HALF_CAR:.2f} m보다 넓으면 한 길로 둘 다 "
      f"지나감 -> 서로 반대쪽이 강제되지 않음")
print(f"{'공통':>6}{'트랙 m':>7}{'x':>8}{'y':>8}{'보임 s':>8}{'보임 m':>8}{'속도':>7}{'폭':>7}{'길-왼벽':>8}{'길-오른벽':>9}  L(왼틈/오른틈, 길 막음, 보임)  R(...)")
for r in ok_rows[:25]:
    print(f"{r['common']:6.2f}{r['s']:7.1f}{P[r['j'], 0]:8.2f}{P[r['j'], 1]:8.2f}{r['see']:8.2f}{r['see_d']:8.2f}{r['v']:7.2f}{r['dl'] + r['dr']:7.2f}{r['dl']:8.2f}{r['dr']:9.2f}"
          f"  {r['L_gl']:.2f}/{r['L_gr']:.2f} {'막음' if r['L_on_line'] else '안 막음'} {r['L_see']:.2f}  {r['R_gl']:.2f}/{r['R_gr']:.2f} "
          f"{'막음' if r['R_on_line'] else '안 막음'} {r['R_see']:.2f}")
if not ok_rows:
    sys.exit("조건을 다 맞는 자리가 없음")
c = ok_rows[0]
# contiguous stretches of qualifying points, for the report
qs = sorted(r["s"] for r in ok_rows)
runs, cur = [], [qs[0]]
for a in qs[1:]:
    if a - cur[-1] <= 0.25:
        cur.append(a)
    else:
        runs.append(cur); cur = [a]
runs.append(cur)
print("조건 맞는 구간 (트랙 m): " + ", ".join(f"{r_[0]:.1f}-{r_[-1]:.1f}" for r_ in runs))
# the exam3 first sights of real boxes near the spot, as a cross-check of the computed one
print(f"\n고른 자리: 트랙 {c['s']:.1f} m, 길 위 점 ({P[c['j'], 0]:+.3f}, {P[c['j'], 1]:+.3f}), 진행 방향 ({tan[c['j'], 0]:+.3f}, {tan[c['j'], 1]:+.3f})")
print(f"  길 위 상자 처음 보임 {c['see']:.2f} s ({c['see_d']:.1f} m 앞), 속도 {c['v']:.2f} m/s, 폭 {c['dl'] + c['dr']:.2f} m (길에서 왼벽 {c['dl']:.2f}, 오른벽 {c['dr']:.2f})")
print(f"  두 자리가 함께 남기는 통로 {c['common']:.2f} m (차 폭 {2 * HALF_CAR:.2f} m) -> "
      + ("서로 반대쪽만 지나갈 수 있음" if c["common"] < 2 * HALF_CAR else "가운데 한 길로 L·R 둘 다 지나갈 수 있음 (반대쪽이 강제되지 않음)"))
for side in ("L", "R"):
    print(f"  {side}: 중심 ({c[side + '_ctr'][0]:+.4f}, {c[side + '_ctr'][1]:+.4f}) = 길에서 {c[side + '_off']:+.3f} m, 틈 왼쪽 {c[side + '_gl']:.3f} m / "
          f"오른쪽 {c[side + '_gr']:.3f} m, 길 위 차 몸 {'막음' if c[side + '_on_line'] else '안 막음'}, 처음 보임 {c[side + '_see']:.2f} s")

w = json.load(open(world_out if os.path.exists(world_out) else QUAL))
w.update({"name": "plan_world_korea26scan_tmaze", "warps": 0, "episode_s": 32.0, "prop_bank": "", "prop_count": "fixed",
          "prop_fraction": P_LRN[0] + P_LRN[1], "removal_p": 0.0, "opp_present": 0.0, "stage": "props", "exam_name": "exam_tmaze",
          "exam_bank": ""})
r4 = lambda v_: round(float(v_), 4)
lay = lambda side: [[r4(q[0]), r4(q[1]), r4(c["yaw"]), BOX] for q in pieces(c[side + "_ctr"], nrm[c["j"]], 1.0)]
w["tmaze"] = {"L": lay("L")[0] if N_BOX == 1 else lay("L"), "R": lay("R")[0] if N_BOX == 1 else lay("R"),
              "p": P_LRN,
              "source": {"record": os.path.basename(rec_path), "policy": str(z["name"]), "rule": "tmaze_spot.py (opposite sides)",
                         "track_s_m": round(c["s"], 2), "line_point": [r4(P[c["j"], 0]), r4(P[c["j"], 1])],
                         "left_normal": [r4(nrm[c["j"], 0]), r4(nrm[c["j"], 1])], "line_box_first_sight_s": round(c["see"], 3),
                         "speed_mps": round(float(c["v"]), 2), "width_m": round(c["dl"] + c["dr"], 3),
                         "line_to_left_wall_m": round(c["dl"], 3), "line_to_right_wall_m": round(c["dr"], 3),
                         "L_gaps_m": [round(c["L_gl"], 3), round(c["L_gr"], 3)], "R_gaps_m": [round(c["R_gl"], 3), round(c["R_gr"], 3)],
                         "L_blocks_line": c["L_on_line"], "R_blocks_line": c["R_on_line"],
                         "L_first_sight_s": round(c["L_see"], 3), "R_first_sight_s": round(c["R_see"], 3),
                         "boxes_per_placement": N_BOX, "common_passage_m": round(c["common"], 3),
                         "limits": {"sight_max_s": SIGHT_MAX, "v_min": V_MIN, "w_min": W_MIN, "gap_narrow": GAP_NARROW, "gap_wide": GAP_WIDE}}}
w["_note_tmaze"] = ("11-26 racing T-maze (RQ3/4/6, docs/.../analysis/rq_plan_2026-11-26.md): one map, no variants; " + f"{N_BOX} real box(es) "
                    "(0.36 x 0.30 m each, long side across) at a blind fast spot, L = against the left wall (0.15 m gap, the car cannot "
                    "pass: forced right) or R = the mirror (forced left) or none, P = (0.4, 0.4, 0.2) per episode, never removed; "
                    + ("the middle both leave free is narrower than the car (exclusive sides); " if c["common"] < 2 * HALF_CAR else
                       "NOT exclusive: the middle both leave free passes the car; ") + 
                    "tools/tmaze_spot.py from scan31b2 u600's exam3 record (line box first seen <= 0.7 s, speed >= 4.5 m/s, width "
                    ">= 1.3 m); 32 s episodes (~4 laps), no opponents; train with --scene-replay 0")
with open(world_out, "w") as fh:
    json.dump(w, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(f"저장 {world_out}")

if png:
    fig, ax = plt.subplots(figsize=(8, 8))
    R.draw_map(ax, tk)
    for b in np.nonzero(~boxed & ~opp)[0][:24]:
        ax.plot(xy[30:, b, 0], xy[30:, b, 1], "-", lw=0.3, color="tab:blue", alpha=0.35)
    for side, col in (("L", "tab:green"), ("R", "tab:red")):
        for q0 in pieces(c[side + "_ctr"], nrm[c["j"]], 1.0):
            q = box_outline(q0[0], q0[1], c["yaw"], fp, 0.05)
            ax.fill(q[:, 0], q[:, 1], color=col, alpha=0.6)
    for r in ok_rows:
        ax.plot(*P[r["j"]], ".", ms=3, color="tab:orange")
    pj = P[c["j"]]
    ax.plot(*pj, "x", color="k", ms=10)
    ax.annotate("", xy=pj + 0.7 * tan[c["j"]], xytext=pj, arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(pj[0] - 3, pj[0] + 3); ax.set_ylim(pj[1] - 3, pj[1] + 3)
    ax.set_title(f"T-미로: L(초록) R(빨강), 길 위 상자 처음 보임 {c['see']:.2f} s, {c['v']:.1f} m/s, 폭 {c['dl'] + c['dr']:.2f} m")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    print(f"그림 {png}")
