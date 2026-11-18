"""(11-18, 사용자: "학습환경 장애물 자체를 이 평가에 쓰는거 그대로 말고, 같은 방식으로 라인 위 균일 고르게 뽑힌 맵으로 가자. 대신 훨씬 많이",
"훨씬 많이가. 환경 수자체를 말하는거야. 맵당 장애물 수는 지금처럼 3개로") 학습 상자 은행: exam_line_bank.py(시험 exam2)와 같은 방식을 학습 트랙
전부(실제 지도 + 뒤튼 16개)에, 트랙마다 그 트랙의 빈 트랙 주행 길로 --
  - 배치마다 상자 3개. 한 바퀴를 (배치 수 x 3)개의 같은 토막으로 나눠 토막마다 하나 -> 모든 배치를 합치면 트랙 전체에 고르게
  - 기준 길에서 상자 중심까지 정규분포(표준편차 0.15 m)를 ±0.40 m에서 자름 -> 길 위에 밀집
  - 모양은 지금 학습 은행의 상자 모양(25~31)을 고르게, 방향 무작위
  - layout_bank.py와 같은 지나갈 길 판정(차 몸체가 상자·벽과 0.10 m 이상, 곡률이 조향 한계 이하); 못 지나가면 위치는 두고 막은 상자만
    옆으로 다시 놓음, 4번 넘게 못 지나가면 그 상자만 제 토막 안에서 옮김
  - 기준 길 = drive_record.py로 상자 없이 달린 기록에서 그 트랙의 차들 위치를 트랙 따라 0.25 m 칸마다 중앙값
시험 은행(exam2)과는 시드가 달라 같은 배치가 없음. 트랙 이름을 열쇠로 한 파일 하나 = 학습의 --prop-bank 디렉터리에 같이 두면 합쳐 읽힘.
인자: 세계json 주행기록npz 출력.pt 트랙당배치수 트랙번호들(예 0-8 또는 0,3,5) [시드]   (BANK_DEVICE=cuda|cpu)"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "f1sim")))
from f1sim.learn import state_rl as SR
from f1sim import layout_line as LL
from f1sim import props as _props
from f1sim.raceline import curvature
from f1sim.learn.plan_reward import GoalField
from rec_lib import Rec

WJ, RECF, OUT, N, TSEL = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 20261119
NEAR, SIG, N_BOX, ITERS, BIN = 0.40, 0.15, 3, 300, 0.25
NB = int(os.environ.get("BANK_NB", 64))
T_START = time.time()


def tick(what):
    print(f"  [{time.time() - T_START:7.1f} s] {what}", flush=True)
CLR_PROP, CLR_WALL = 0.10, 0.10
SHAPES = list(range(25, 32))                                     # the training bank's boxes, at even rates
dev = torch.device(os.environ.get("BANK_DEVICE", "cuda"))
tsel = (list(range(int(TSEL.split("-")[0]), int(TSEL.split("-")[1]) + 1)) if "-" in TSEL else [int(x) for x in TSEL.split(",")])
rng = np.random.default_rng(SEED + 1000 * tsel[0])

a = argparse.Namespace(**json.load(open(WJ)))
a.envs, a.seed, a.device = 64, SEED, str(dev)
a.prop_place, a.prop_count, a.prop_fraction, a.prop_bank = "uniform", "fixed", 1.0, ""
env = SR.build_env(a, dev)
tick("환경")
names = getattr(env, "_track_names", None) or [SR.KOREA]
_, rls = SR.common.load_tracks(names, racelines=True, **dict(SR.common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
tracks = env.sim.track.tracks
C, po = env.procedural.C, env.procedural
FP = {}
for k in SHAPES:
    sh = po.shapes[k]
    FP[k] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float)

R = Rec(RECF); z = R.z
tid0 = z["tid"][0].astype(int)
assert len(z["t_H"]) == len(tracks), (len(z["t_H"]), len(tracks))
goal = GoalField(env.sim.track, dev)          # the reward's walls-only distance to go: where a point is along the lap
tick("거리장")


def ref_line(ti):
    """(points (n, 2), their place along the lap (n,), the lap's length) from the record's cars on track ti"""
    cars = np.nonzero(tid0 == ti)[0]
    L = float(goal.lap[ti])
    xy = R.st[:R.T, cars, :2].reshape(-1, 2).astype(float)
    with torch.no_grad():
        xt = torch.tensor(xy, dtype=torch.float32, device=dev)
        rem = torch.cat([goal.remaining(xt[i:i + 65536], torch.full((min(65536, len(xt) - i),), ti, dtype=torch.long, device=dev))
                         for i in range(0, len(xt), 65536)]).float().cpu().numpy()
    s = np.mod(L - rem, L)
    ok = np.isfinite(s)
    s, xy = s[ok], xy[ok]
    nb = int(np.ceil(L / BIN)); b = np.minimum((s / BIN).astype(int), nb - 1)
    pts, ss = [], []
    for i in range(nb):
        m = b == i
        if m.sum() >= 5:
            pts.append(np.median(xy[m], 0)); ss.append((i + 0.5) * BIN)
    return np.array(pts), np.array(ss), L, len(cars)


def poly(k, x, y, yaw):
    q = FP[k]; c, s_ = np.cos(yaw), np.sin(yaw)
    return np.stack([x + q[:, 0] * c - q[:, 1] * s_, y + q[:, 0] * s_ + q[:, 1] * c], 1)


def lateral():
    while True:
        d = rng.normal(0.0, SIG)
        if abs(d) <= NEAR:
            return d


def gen_track(ti):
    track = tracks[ti]
    refp, refs, L, ncar = ref_line(ti)
    tick(f"트랙 {ti} 기준 길 ({len(refp)}점)")
    tan = np.roll(refp, -1, 0) - np.roll(refp, 1, 0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    builder = LL.LayoutLines(track, rls[ti], dev, veh_width=float(env.cfg.vehicle.width), vehicle=env.cfg.vehicle, iters=ITERS)
    kmax = builder.kmax
    tick("판정기")
    K = N * N_BOX
    phase0 = rng.uniform(0.0, L / K)
    st = dict(judged=0, failed=0, moved=0, redrawn=0)

    def stretch(k):
        return (phase0 + (k + rng.uniform(0.2, 0.8)) * L / K) % L

    def place(s):
        j = int(np.argmin(np.abs(((refs - s + 0.5 * L) % L) - 0.5 * L)))
        k = int(rng.choice(SHAPES))
        for _ in range(50):
            d = lateral(); yaw = rng.uniform(0.0, np.pi)
            x, y = refp[j] + d * nrm[j]
            if R.wall_dist(ti, np.vstack([poly(k, x, y, yaw), [[x, y]]])).min() >= CLR_WALL:
                return (k, x, y, yaw, refs[j], d)
        return None

    def fill(lay, jb):
        while True:
            b = place(lay["s"][jb])
            if b is not None:
                lay["boxes"][jb] = b
                return
            st["moved"] += 1
            lay["s"][jb] = stretch(lay["slot"][jb])

    w = N / 8
    p1 = np.argsort(np.arange(N) + rng.uniform(-w, w, N)); p2 = np.argsort(np.arange(N) + rng.uniform(-w, w, N))
    lays = []
    for i in range(N):
        slot = np.array([i, N + p1[i], 2 * N + p2[p1[i]]])
        lays.append(dict(slot=slot, s=np.array([stretch(k) for k in slot]), boxes=[None] * N_BOX, rnd=0))
    for lay in lays:
        for jb in range(N_BOX):
            fill(lay, jb)
    tick("첫 배치들")
    pend, laps_ok, t0, last = list(range(N)), {}, time.time(), 0.0
    while pend:
        batch, pend = pend[:NB], pend[NB:]
        lines, laps, clr, _ = builder.build([[poly(*b[:4]) for b in lays[i]["boxes"]] for i in batch])
        for i, xy, lp, (cp, cw) in zip(batch, lines, laps, clr):
            st["judged"] += 1
            kap = np.abs(uniform_filter1d(curvature(xy), 5, mode="wrap"))
            if cp >= CLR_PROP and cw >= CLR_WALL and kap.max() <= kmax and np.isfinite(lp):
                laps_ok[i] = float(lp)
                continue
            st["failed"] += 1
            lay = lays[i]; lay["rnd"] += 1
            P = [poly(*b[:4]) for b in lay["boxes"]]
            cpi = np.array([LL.check_line(xy, [p_], track)[0] for p_ in P])
            rm = cpi < CLR_PROP
            hot = xy[kap > kmax]
            if len(hot):
                cen = np.array([[b[1], b[2]] for b in lay["boxes"]])
                rm |= np.linalg.norm(cen[:, None] - hot[None], axis=-1).min(1) < 1.0
            if not rm.any():
                rm[int(np.argmin(cpi))] = True
            for jb in np.nonzero(rm)[0]:
                st["redrawn"] += 1
                if lay["rnd"] > 4:
                    st["moved"] += 1
                    lay["s"][jb] = stretch(lay["slot"][jb])
                fill(lay, jb)
            pend.append(i)
        tick(f"판정 묶음 {len(batch)}개")
        if time.time() - last > 60:
            last = time.time()
            print(f"  [트랙 {ti} {(time.time() - t0) / 60:5.1f} 분] 통과 {len(laps_ok)}/{N}, 판정 {st['judged']}, 못 지나감 {st['failed']}, "
                  f"다시 놓은 상자 {st['redrawn']} (트랙 따라 옮김 {st['moved']})", flush=True)
    sid = torch.full((N, C), -1, dtype=torch.long); pose = torch.zeros(N, C, 3)
    for i in range(N):
        for c, (k, x, y, yaw, _, _) in enumerate(lays[i]["boxes"]):
            sid[i, c] = k; pose[i, c] = torch.tensor([x, y, yaw])
    d_all = np.array([b[5] for lay in lays for b in lay["boxes"]])
    print(f"== 트랙 {ti} {track.name}: 배치 {N}개 x 상자 {N_BOX}개, {(time.time() - t0) / 60:.1f} 분 | 기준 길 차 {ncar}대, 점 {len(refp)}개, 한 바퀴 {L:.1f} m | "
          f"판정 {st['judged']} 중 못 지나감 {st['failed']}, 다시 놓은 상자 {st['redrawn']}, 옮긴 상자 {st['moved']} | 0.10 m 안 {100 * np.mean(np.abs(d_all) <= 0.1):.0f} %, "
          f"0.20 m 안 {100 * np.mean(np.abs(d_all) <= 0.2):.0f} %", flush=True)
    return track.name, dict(sid=sid, pose=pose, lap=torch.tensor([laps_ok[i] for i in range(N)]), length=float(env.sim.track.length[ti]),
                            refline=torch.tensor(refp, dtype=torch.float32))


per = {}
for ti in tsel:
    n_, d_ = gen_track(ti)
    per[n_] = d_
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(dict(per_track=per, C=C, judge=dict(clr_prop=CLR_PROP, clr_wall=CLR_WALL, iters=ITERS, near=NEAR, sig=SIG, n_box=N_BOX,
                                                   record=os.path.abspath(RECF), seed=SEED)), OUT)
print(f"== 끝: {OUT} 트랙 {len(per)}개", flush=True)
