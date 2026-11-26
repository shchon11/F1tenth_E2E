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
(11-26 셋째 판, coordinator 결정: 반응만으로는 한쪽을 피할 수 없게 -- 그래야 기억이 벌이가 있고 RL이 미리 움직일 까닭이 생김. u600은
둘째 판의 R(상자 2개)을 2번째 이후 통과에서 0 %로 피했음)
  4. 막는 쪽 = 기준 길이 붙어 달리는 벽 쪽. 거기서 상자 n_b개 줄(벽과 틈 GAP_NARROW)이 길을 덮어, 차가 먼 쪽으로 옮겨야 할 폭 y_req.
     줄의 어느 부분이든 처음 보이는 때(외곽선 점마다 같은 시선 규칙, 0.05 m 앞까지) t_s에서 지나갈 때까지 옆으로 갈 수 있는 최대
     y_max = a_lat t_s^2 / 2 (A_LAT: plan 한계 = 타이어 x plan_lat_margin 0.95, 6-8 m/s에서 7.8-7.9 m/s^2). 비 y_req / y_max >= 1.5.
     반대쪽 = 먼 벽에서 상자 n_o개 줄: 길을 막지 않고(옮길 필요 없음), 두 줄이 함께 비우는 통로 < 차 폭(배타), 그 옆 통로 >= GAP_WIDE.
     후보는 비가 큰 순 (같으면 상자가 적은 순). TMAZE_PICK=k: k번째 후보.
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
A_LAT = 7.9                  # [m/s^2] the plan's lateral limit at 6-8 m/s (Limits.from_nominal, margin 0.95; the tyres 8.3)
RATIO_MIN = 1.5              # the shift needed / the shift reachable after first sight
MODE = os.environ.get("TMAZE_MODE", "react")          # "react" (third design) | "exclusive" (the second: N_BOX each side)
PICK = int(os.environ.get("TMAZE_PICK", "0"))
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


def pieces(center, n, sg, k=None):
    """the placement's box centres: k (default N_BOX) boxes side by side along n, `center` the middle of the row"""
    k = N_BOX if k is None else k
    w = 2.0 * half_across + GAP_BOX
    return [center + (i - 0.5 * (k - 1)) * w * n for i in range(k)]


def gaps(center, yaw, fp, n, k=None):
    """[m] (left, right) the narrowest gap between the placement's boxes and the wall on each side: from the outline points on
    that side, the nearest wall within +-60 deg of +n / -n"""
    o = np.concatenate([box_outline(c_[0], c_[1], yaw, fp) for c_ in pieces(center, n, 1.0, k)])
    side = (o - center) @ n
    gl = min(fan(p, n) for p in o[side > 0.0][::3])
    gr = min(fan(p, -n) for p in o[side < 0.0][::3])
    return gl, gr


def seen_any(sp, syaw, pts):
    """any of `pts` (k, 2) in direct view from a scanner at sp heading syaw (the exam's rule, up to 0.05 m short of the point)"""
    d = pts - sp[None]; dist = np.linalg.norm(d, axis=1)
    a = (np.arctan2(d[:, 1], d[:, 0]) - syaw + math.pi) % (2 * math.pi) - math.pi
    m = (dist <= RMAX) & (np.abs(a) <= FOV)
    if not m.any():
        return False
    d, dist = d[m], dist[m]
    f = np.linspace(0.0, 1.0, 120)[None, :] * (np.maximum(dist - 0.05, 0.0) / np.maximum(dist, 1e-6))[:, None]
    seg = sp[None, None, :] + f[..., None] * d[:, None, :]
    return bool((wall(seg.reshape(-1, 2)).reshape(seg.shape[:2]) > 0.0).all(1).any())


def first_sight_any(j, pts):
    """[s, m] before reaching line bin j that any of `pts` is first seen by a car driving the line at its speed"""
    pose = lambda q: (P[q] + mount * tan[q], math.atan2(tan[q, 1], tan[q, 0]))
    k = None
    for back in range(0, 6):
        if seen_any(*pose((j - back) % nb), pts):
            k = (j - back) % nb; break
    if k is None:
        return 0.0, 0.0
    while (j - (k - 1)) % nb <= int(12.0 / BIN) and seen_any(*pose((k - 1) % nb), pts):
        k = (k - 1) % nb
    steps = [(k + i) % nb for i in range((j - k) % nb)]
    return float(sum(BIN / max(vb[q], 0.3) for q in steps)), (j - k) % nb * BIN


def place_row(j, k, sg, dl, dr, yaw):
    """k boxes from the left (sg +1) or right (sg -1) wall at line bin j with the narrow gap GAP_NARROW: (centre, offset along
    the normal from the line point, inner edge offset, (gap left, gap right))"""
    n = nrm[j]; sp_ = k * 2.0 * half_across + (k - 1) * GAP_BOX
    near = dl if sg > 0 else dr
    c_off = sg * (near - GAP_NARROW - 0.5 * sp_)
    for _ in range(6):
        gl, gr = gaps(P[j] + c_off * n, yaw, fp, n, k)
        c_off += sg * ((gl if sg > 0 else gr) - GAP_NARROW)
    gl, gr = gaps(P[j] + c_off * n, yaw, fp, n, k)
    return P[j] + c_off * n, c_off, c_off - sg * 0.5 * sp_, (gl, gr)


fp = z[f"fp_{BOX}"].astype(float)
half_across = 0.5 * np.ptp(fp[:, 0])                      # the long side laid across the corridor
if MODE == "react":
    rows3 = []
    for j in range(0, nb, 2):
        v = vb[j]
        if v < V_MIN:
            continue
        dl, dr = ray(P[j], nrm[j]), ray(P[j], -nrm[j])
        sg = 1.0 if dl < dr else -1.0                     # the block side: the wall the line runs nearer
        see0, _ = first_sight(j, P[j])
        if see0 > 1.2:                                     # a box on the line seen early: no blind spot here
            continue
        n = nrm[j]; yaw = math.atan2(n[1], n[0])
        for kb_ in (2, 3, 4):
            if dl + dr < GAP_NARROW + kb_ * 2 * half_across + GAP_WIDE:
                break
            cb, ob, ib, gb = place_row(j, kb_, sg, dl, dr, yaw)
            far_gap = gb[1] if sg > 0 else gb[0]
            if (gb[0] if sg > 0 else gb[1]) >= 0.25 or far_gap < GAP_WIDE:
                continue
            y_req = max(0.0, HALF_CAR - sg * ib)
            if y_req <= 0.0:
                continue
            pts = np.concatenate([box_outline(q[0], q[1], yaw, fp, 0.1) for q in pieces(cb, n, 1.0, kb_)])
            t_s, d_s = first_sight_any(j, pts)
            y_max = 0.5 * A_LAT * t_s ** 2
            # the other side: the fewest boxes from the far wall that leave the line free and make the two exclusive
            other = None
            for ko in (1, 2, 3):
                co, oo, io, go = place_row(j, ko, -sg, dl, dr, yaw)
                common = sg * (ib - io)                    # the middle both rows leave free
                free_line = (sg * io) <= -HALF_CAR         # the other row's inner edge beyond the line's car body
                near_gap = go[1] if sg < 0 else go[0]      # (its passage: on the block side)
                narrow_o = go[0] if sg < 0 else go[1]
                if common < 2 * HALF_CAR and free_line and near_gap >= GAP_WIDE and narrow_o < 0.25:
                    pts_o = np.concatenate([box_outline(q[0], q[1], yaw, fp, 0.1) for q in pieces(co, n, 1.0, ko)])
                    other = dict(k=ko, c=co, off=oo, inner=io, gaps=go, common=common, see=first_sight_any(j, pts_o)[0])
                    break
            if other is None:
                continue
            rows3.append(dict(j=j, s=j * BIN, v=v, dl=dl, dr=dr, sg=sg, yaw=yaw, see0=see0, kb=kb_, cb=cb, ob=ob, ib=ib, gb=gb,
                              y_req=y_req, t_s=t_s, d_s=d_s, y_max=y_max, ratio=y_req / max(y_max, 1e-6), other=other))
    rows3.sort(key=lambda r: (not bool(r["ratio"] >= RATIO_MIN), -float(r["ratio"]), r["kb"]))
    print(f"막는 쪽 = 길이 붙은 벽 쪽, 상자 2-4개 줄; 옮겨야 할 폭 / (처음 보임 뒤 옆으로 갈 수 있는 폭 = {A_LAT} t^2 / 2) >= {RATIO_MIN}: "
          f"{sum(r['ratio'] >= RATIO_MIN for r in rows3)}개 / 후보 {len(rows3)}개")
    print(f"{'#':>3}{'트랙 m':>7}{'x':>7}{'y':>7}{'속도':>6}{'폭':>6}{'쪽':>3}{'상자':>4}{'옮길 폭':>8}{'보임 s':>7}{'갈 폭':>7}{'비':>6}"
          f"{'먼쪽 통로':>8}  반대쪽(상자, 공통 통로, 보임 s)")
    for q, r in enumerate(rows3[:20]):
        o = r["other"]
        print(f"{q:3d}{r['s']:7.1f}{P[r['j'], 0]:7.2f}{P[r['j'], 1]:7.2f}{r['v']:6.2f}{r['dl'] + r['dr']:6.2f}{'L' if r['sg'] > 0 else 'R':>3}"
              f"{r['kb']:4d}{r['y_req']:8.2f}{r['t_s']:7.2f}{r['y_max']:7.2f}{r['ratio']:6.2f}{(r['gb'][1] if r['sg'] > 0 else r['gb'][0]):8.2f}"
              f"  {o['k']}개, {o['common']:.2f} m, {o['see']:.2f}")
    if not rows3:
        sys.exit("후보 없음")
    c = rows3[min(PICK, len(rows3) - 1)]
    o = c["other"]
    blk, oth = ("L", "R") if c["sg"] > 0 else ("R", "L")
    print(f"\n고른 자리 (#{min(PICK, len(rows3) - 1)}): 트랙 {c['s']:.1f} m, 길 위 점 ({P[c['j'], 0]:+.3f}, {P[c['j'], 1]:+.3f}), 진행 방향 "
          f"({tan[c['j'], 0]:+.3f}, {tan[c['j'], 1]:+.3f}), 속도 {c['v']:.2f} m/s, 폭 {c['dl'] + c['dr']:.2f} m (길에서 왼벽 {c['dl']:.2f}, 오른벽 {c['dr']:.2f})")
    print(f"  막는 쪽 {blk}: 상자 {c['kb']}개, 틈 왼쪽 {c['gb'][0]:.3f} / 오른쪽 {c['gb'][1]:.3f} m, 옮겨야 할 폭 {c['y_req']:.2f} m, 줄이 처음 보임 "
          f"{c['t_s']:.2f} s ({c['d_s']:.1f} m 앞) -> 옆으로 갈 수 있는 폭 {c['y_max']:.2f} m, 비 {c['ratio']:.2f}")
    print(f"  반대쪽 {oth}: 상자 {o['k']}개, 틈 왼쪽 {o['gaps'][0]:.3f} / 오른쪽 {o['gaps'][1]:.3f} m, 길 안 막음(옮길 필요 없음), 두 줄이 함께 "
          f"비우는 통로 {o['common']:.2f} m (< 차 폭 0.31: 배타), 처음 보임 {o['see']:.2f} s")
    w = json.load(open(world_out if os.path.exists(world_out) else QUAL))
    w.update({"name": "plan_world_korea26scan_tmaze", "warps": 0, "episode_s": 32.0, "prop_bank": "", "prop_count": "fixed",
              "prop_fraction": P_LRN[0] + P_LRN[1], "removal_p": 0.0, "opp_present": 0.0, "stage": "props", "exam_name": "exam_tmaze",
              "exam_bank": ""})
    r4 = lambda v_: round(float(v_), 4)
    row_of = lambda ctr, k_: [[r4(q[0]), r4(q[1]), r4(c["yaw"]), BOX] for q in pieces(ctr, nrm[c["j"]], 1.0, k_)]
    lay = {blk: row_of(c["cb"], c["kb"]), oth: row_of(o["c"], o["k"])}
    w["tmaze"] = {"L": lay["L"], "R": lay["R"], "p": P_LRN,
                  "source": {"record": os.path.basename(rec_path), "policy": str(z["name"]), "rule": "tmaze_spot.py react (third design)",
                             "track_s_m": round(c["s"], 2), "line_point": [r4(P[c["j"], 0]), r4(P[c["j"], 1])],
                             "left_normal": [r4(nrm[c["j"], 0]), r4(nrm[c["j"], 1])], "speed_mps": round(float(c["v"]), 2),
                             "width_m": round(c["dl"] + c["dr"], 3), "line_to_left_wall_m": round(c["dl"], 3),
                             "line_to_right_wall_m": round(c["dr"], 3), "block_side": blk, "block_boxes": c["kb"],
                             "block_gaps_m": [round(x_, 3) for x_ in c["gb"]], "shift_needed_m": round(c["y_req"], 3),
                             "block_first_sight_s": round(c["t_s"], 3), "shift_reachable_m": round(c["y_max"], 3),
                             "ratio": round(c["ratio"], 3), "a_lat": A_LAT, "other_side": oth, "other_boxes": o["k"],
                             "other_gaps_m": [round(x_, 3) for x_ in o["gaps"]], "common_passage_m": round(o["common"], 3),
                             "other_first_sight_s": round(o["see"], 3), "line_box_first_sight_s": round(c["see0"], 3)}}
    w["_note_tmaze"] = ("11-26 racing T-maze, third design (RQ3/4/6; docs/.../analysis/rq_plan_2026-11-26.md): one map, no variants; at a blind "
                        f"fast spot the side the reference line hugs ({blk}) is a row of {c['kb']} real boxes (0.36 x 0.30 m, long side across, "
                        f"0.15 m from the wall): the car must move {c['y_req']:.2f} m; after the first sight of any part ({c['t_s']:.2f} s on "
                        f"the line) the zero-delay bound a_lat t^2 / 2 ({A_LAT} m/s^2) is {c['y_max']:.2f} m (ratio {c['ratio']:.2f}; no spot "
                        f"reaches {RATIO_MIN}) -- the exam says how often reaction fails; the other ({oth}) is {o['k']} box(es) "
                        "from the far wall: the line passes, the two exclusive; none; P = (0.4, 0.4, 0.2) per episode, never removed; "
                        "tools/tmaze_spot.py from scan31b2 u600's exam3 record; 32 s episodes, no opponents; train with --scene-replay 0")
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
            for q0 in lay[side]:
                q = box_outline(q0[0], q0[1], q0[2], fp, 0.05)
                ax.fill(q[:, 0], q[:, 1], color=col, alpha=0.6)
        pj = P[c["j"]]
        ax.plot(*pj, "x", color="k", ms=10)
        ax.annotate("", xy=pj + 0.7 * tan[c["j"]], xytext=pj, arrowprops=dict(arrowstyle="->"))
        ax.set_xlim(pj[0] - 3, pj[0] + 3); ax.set_ylim(pj[1] - 3, pj[1] + 3)
        ax.set_title(f"T-미로 셋째 판: L(초록) R(빨강), 막는 쪽 {blk}: {c['y_req']:.2f} m 옮김, 보임 {c['t_s']:.2f} s, 비 {c['ratio']:.1f}")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"그림 {png}")
    sys.exit(0)

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
