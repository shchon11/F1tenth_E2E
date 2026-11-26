"""T-미로 자리 찾기 (11-27, v4): 보이기 늦은 자리와 '보이기 전에 다른 길로 옮겨 둘 공간'이 있는 자리가 겹치는가.
기억으로 버는 시간 = 상자를 보기 전에 다른 길(막히지 않은 쪽)로 옮겨 두는 것. 그러려면 (가) 막는 줄이 늦게 보이고(처음 보임이 짧고) (나) 그보다 앞에서
이미 다른 쪽에 있을 공간(그쪽 벽까지 거리 >= 옮길 폭 + 0.3 m)이 이어져 있어야 함. tmaze_spot.py(셋째 판)의 줄 배치와 시선 규칙을 그대로 써서
길 위 0.4 m마다(기준 policy 속도 >= V_MIN):
  막는 줄 = 길이 붙은 벽에서 상자 k개(GAP_NARROW 틈), 옮길 폭 y_req, 반대쪽 통로, 줄 어느 부분이든 처음 보인 때(초, m 앞),
  '미리 m' = 자리에서 거꾸로, 반대쪽에 y_req + 0.1 m(차 몸 + 0.3 m 여유 안에서)로 머물 수 있는 구간이 이어지는 길이.
  '미리 m' >= 처음 보인 m + 1 이면 보이기 1 m 전부터 옮겨 둘 수 있음.
인자: exam3 기록npz(기본 scan31b2 u600)  [write S KB KO 출력json: 그 자리로 world json(셋째 판 형식, p 0.4/0.4/0.2)을 씀]
분석만 -- 학습에 쓰지 않음."""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rec_lib import Rec, along_track
from tmaze_measure import ref_line, BIN

REC = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".npz") else os.path.join(
    HERE, "../out/exam/exam3_scan31b2_u600_ideal_line3_n256.npz")
V_MIN, GAP_NARROW, GAP_BOX, HALF_CAR, BOX = 4.5, 0.15, 0.01, 0.155, 38
FOV, RMAX = math.radians(135.0), 10.0
V3 = os.path.normpath(os.path.join(HERE, "../../../../f1sim/f1sim/learn/configs/plan_world_korea26scan_tmaze.json"))

R = Rec(REC); z = R.z; T, dt = R.T, R.dt
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
nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
wall = lambda pts: R.wall_dist(tk, pts)
fp = z[f"fp_{BOX}"].astype(float)
half_across = 0.5 * np.ptp(fp[:, 0])


def ray(p0, d, maxd=4.0, step=0.005):
    r = np.arange(0.0, maxd, step)
    hit = wall(p0[None] + r[:, None] * d[None]) <= 0.0
    return float(r[hit.argmax()]) if hit.any() else maxd


def seen_any(sp, syaw, pts):
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
    """[s, m] before line bin j that any of `pts` is first seen by a car driving the line at its speed (tmaze_spot.py's)"""
    pose = lambda q: (P[q] + mount * tan[q], math.atan2(tan[q, 1], tan[q, 0]))
    k = None
    for back in range(0, 6):
        if seen_any(*pose((j - back) % nb), pts):
            k = (j - back) % nb; break
    if k is None:
        return 0.0, 0.0
    while (j - (k - 1)) % nb <= int(15.0 / BIN) and seen_any(*pose((k - 1) % nb), pts):
        k = (k - 1) % nb
    steps = [(k + i) % nb for i in range((j - k) % nb)]
    return float(sum(BIN / max(vb[q], 0.3) for q in steps)), (j - k) % nb * BIN


def box_outline(cx, cy, yaw, fp_, step=0.02):
    pts = []
    ring = np.concatenate([fp_, fp_[:1]])
    for a, b in zip(ring[:-1], ring[1:]):
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts += [a + (b - a) * i / m for i in range(m)]
    q = np.array(pts); c, s_ = math.cos(yaw), math.sin(yaw)
    return np.stack([cx + q[:, 0] * c - q[:, 1] * s_, cy + q[:, 0] * s_ + q[:, 1] * c], 1)


def fan(p, d, half=math.radians(60.0), k=9):
    a0 = math.atan2(d[1], d[0])
    return min(ray(p, np.array([math.cos(a0 + a), math.sin(a0 + a)]), 3.0) for a in np.linspace(-half, half, k))


def pieces(center, n, k):
    w = 2.0 * half_across + GAP_BOX
    return [center + (i - 0.5 * (k - 1)) * w * n for i in range(k)]


def gaps(center, yaw, n, k):
    o = np.concatenate([box_outline(c_[0], c_[1], yaw, fp) for c_ in pieces(center, n, k)])
    side = (o - center) @ n
    return min(fan(p, n) for p in o[side > 0.0][::3]), min(fan(p, -n) for p in o[side < 0.0][::3])


def place_row(j, k, sg, dl, dr, yaw):
    """k boxes from the left (sg +1) / right (-1) wall at line bin j, GAP_NARROW from it: centre, inner edge offset (left +),
    (gap left, gap right) -- tmaze_spot.py's"""
    n = nrm[j]; sp_ = k * 2.0 * half_across + (k - 1) * GAP_BOX
    near = dl if sg > 0 else dr
    c_off = sg * (near - GAP_NARROW - 0.5 * sp_)
    for _ in range(6):
        gl, gr = gaps(P[j] + c_off * n, yaw, n, k)
        c_off += sg * ((gl if sg > 0 else gr) - GAP_NARROW)
    gl, gr = gaps(P[j] + c_off * n, yaw, n, k)
    return P[j] + c_off * n, c_off - sg * 0.5 * sp_, (gl, gr)


def spot(j, k):
    dl, dr = ray(P[j], nrm[j]), ray(P[j], -nrm[j])
    sg = -1.0 if dr < dl else 1.0                         # the block side: the wall the line runs nearer
    n = nrm[j]; yaw = math.atan2(n[1], n[0])
    cb, ib, gb = place_row(j, k, sg, dl, dr, yaw)
    y_req = max(0.0, HALF_CAR - sg * ib)
    pts = np.concatenate([box_outline(q[0], q[1], yaw, fp, 0.1) for q in pieces(cb, n, k)])
    t_s, d_s = first_sight_any(j, pts)
    return dict(j=j, dl=dl, dr=dr, sg=sg, yaw=yaw, cb=cb, ib=ib, gb=gb, y_req=y_req, t_s=t_s, d_s=d_s, k=k)


if __name__ == "__main__":
    if "write" in sys.argv:
        i = sys.argv.index("write")
        S, KB, KO, out = float(sys.argv[i + 1]), int(sys.argv[i + 2]), int(sys.argv[i + 3]), sys.argv[i + 4]
        j = int(round(S / BIN)); c = spot(j, KB)
        n = nrm[j]
        co, io, go = place_row(j, KO, -c["sg"], c["dl"], c["dr"], c["yaw"])
        r4 = lambda v_: round(float(v_), 4)
        row = lambda ctr, k_: [[r4(q[0]), r4(q[1]), r4(c["yaw"]), BOX] for q in pieces(ctr, n, k_)]
        blk, oth = ("L", "R") if c["sg"] > 0 else ("R", "L")
        common = c["sg"] * (c["ib"] - io)                  # the middle both rows leave free
        w = json.load(open(V3))
        w["tmaze"] = {blk: row(c["cb"], KB), oth: row(co, KO), "p": [0.4, 0.4, 0.2],
                      "source": {"record": os.path.basename(REC), "policy": str(z["name"]), "rule": "tmaze_room.py write",
                                 "track_s_m": round(j * BIN, 2), "line_point": [r4(P[j, 0]), r4(P[j, 1])],
                                 "left_normal": [r4(n[0]), r4(n[1])], "speed_mps": round(float(vb[j]), 2),
                                 "width_m": round(c["dl"] + c["dr"], 3), "line_to_left_wall_m": round(c["dl"], 3),
                                 "line_to_right_wall_m": round(c["dr"], 3), "block_side": blk, "block_boxes": KB,
                                 "block_gaps_m": [round(x, 3) for x in c["gb"]], "shift_needed_m": round(c["y_req"], 3),
                                 "block_first_sight_s": round(c["t_s"], 3), "block_first_sight_m": round(c["d_s"], 2),
                                 "other_side": oth, "other_boxes": KO, "other_gaps_m": [round(x, 3) for x in go],
                                 "common_passage_m": round(float(common), 3)}}
        w["tmaze"] = {k_: w["tmaze"][k_] for k_ in ("L", "R", "p", "source")}
        json.dump(w, open(out, "w"), indent=1, ensure_ascii=False)
        print(json.dumps(w["tmaze"]["source"], ensure_ascii=False))
        sys.exit(0)
    dlr = np.array([[ray(P[j], nrm[j]), ray(P[j], -nrm[j])] for j in range(nb)])
    rows = []
    for j in range(0, nb, 4):
        if vb[j] < V_MIN:
            continue
        for k in (2, 3):
            try:
                c = spot(j, k)
            except Exception:
                continue
            far = c["gb"][1] if c["sg"] > 0 else c["gb"][0]
            if far < 0.45:
                continue
            k0 = int(round((c["d_s"] + 3.0) / BIN))
            free = dlr[[(j - q) % nb for q in range(k0 + 1)], 1 if c["sg"] > 0 else 0]
            okq = free - 0.3 >= c["y_req"] + 0.1
            pre = 0.0
            for q in range(len(okq)):
                if not okq[q]:
                    break
                pre = q * BIN
            rows.append((j * BIN, vb[j], "L" if c["sg"] > 0 else "R", k, c["y_req"], far, c["t_s"], c["d_s"], pre))
    print(f"한 바퀴 {L:.2f} m, 기준 길 {nb}칸; 속도 >= {V_MIN} m/s인 자리 x 상자 2/3개 = {len(rows)}")
    print(f"{'s m':>6}{'v':>5}{'쪽':>3}{'k':>3}{'옮길 폭':>8}{'통로':>6}{'보임 s':>7}{'보임 m':>7}{'미리 m':>7}")
    for r in rows:
        print(f"{r[0]:6.1f}{r[1]:5.1f}{r[2]:>3}{r[3]:3d}{r[4]:8.2f}{r[5]:6.2f}{r[6]:7.2f}{r[7]:7.1f}{r[8]:7.1f}")
    blind = [r for r in rows if r[6] <= 0.8 and r[4] >= 0.3]
    pre = [r for r in rows if r[8] >= r[7] + 1.0 and r[4] >= 0.5]
    print(f"\n보이기 늦은 자리 (처음 보임 <= 0.8 s, 옮길 폭 >= 0.3 m): {len(blind)}, 그 가운데 보이기 1 m 전부터 옮겨 둘 수 있는 자리: "
          f"{sum(r[8] >= r[7] + 1.0 for r in blind)}")
    print(f"옮겨 둘 수 있는 자리 (미리 >= 보임 + 1 m, 옮길 폭 >= 0.5 m): {len(pre)}, 그 처음 보임 가장 짧은 것: "
          f"{min((r[6] for r in pre), default=float('nan')):.2f} s")
