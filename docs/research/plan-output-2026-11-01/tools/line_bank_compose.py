"""(11-18, 사용자: "학습환경 장애물 자체를 이 평가에 쓰는거 그대로 말고, 같은 방식으로 라인 위 균일 고르게 뽑힌 맵으로 가자. 대신 훨씬 많이",
"훨씬 많이가. 환경 수자체를 말하는거야. 맵당 장애물 수는 지금처럼 3개로") 학습 상자 은행 -- exam2와 같은 모양의 분포(맵마다 상자 3개, 트랙 따라
고르게, 빈 트랙 주행 길에 밀집)를, 판정을 새로 돌리지 않고 지금 학습 은행의 상자들로 짬:
  - 지금 학습 은행(layout_bank.py: 배치마다 지나갈 길 판정 통과)의 상자 가운데 그 트랙의 기준 길 0.40 m 안의 것만 후보
    (지나갈 길이 있는 배치의 상자 하나는 혼자서도 지나갈 수 있고, 맵 안의 세 상자는 한 바퀴의 1/3쯤씩 떨어져 서로의 비켜 가는 길에 닿지 않음)
  - 맵마다 한 바퀴를 세 칸으로 나눠 칸마다 하나: 세 칸의 시작은 맵들에 고르게 나눈 층, 칸 안에서는 가운데 60 % -> 모든 맵을 합치면 트랙 전체에 고르게
  - 그 자리 ±0.5 m(없으면 넓혀서) 안 후보 가운데 기준 길에서 떨어진 거리 d에 exp(-d^2 / (2 x 0.15^2)) 무게로 하나 -> 길 위에 밀집
  - 기준 길 = drive_record.py로 상자 없이 달린 기록의 그 트랙 차들 위치를 트랙 따라(보상의 벽 기준 거리장) 0.25 m 칸마다 중앙값
인자: 세계json 주행기록npz 원래은행디렉터리 출력.pt 트랙당맵수 [시드]"""
import os, sys, json, glob, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "f1sim")))
from f1sim.learn import state_rl as SR
from f1sim.learn.plan_reward import GoalField
from rec_lib import Rec

WJ, RECF, SRC, OUT, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 20261119
NEAR, SIG, N_BOX, BIN = 0.40, 0.15, 3, 0.25
rng = np.random.default_rng(SEED)
dev = torch.device("cpu")

a = argparse.Namespace(**json.load(open(WJ)))
a.envs, a.seed, a.device = 64, SEED, "cpu"
a.prop_place, a.prop_count, a.prop_fraction, a.prop_bank = "uniform", "fixed", 1.0, ""
env = SR.build_env(a, dev)
tracks = env.sim.track.tracks
C = env.procedural.C
goal = GoalField(env.sim.track, dev)
R = Rec(RECF); z = R.z
tid0 = z["tid"][0].astype(int)
assert len(z["t_H"]) == len(tracks), (len(z["t_H"]), len(tracks))
src = {}
for f in sorted(glob.glob(os.path.join(SRC, "*.pt"))):
    src.update(torch.load(f, map_location="cpu", weights_only=False)["per_track"])


def ref_line(ti):
    cars = np.nonzero(tid0 == ti)[0]
    L = float(goal.lap[ti])
    xy = R.st[:R.T, cars, :2].reshape(-1, 2).astype(float)
    with torch.no_grad():
        rem = goal.remaining(torch.tensor(xy, dtype=torch.float32), torch.full((len(xy),), ti, dtype=torch.long)).numpy()
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


per, report = {}, []
for ti, tr in enumerate(tracks):
    refp, refs, L, ncar = ref_line(ti)
    tan = np.roll(refp, -1, 0) - np.roll(refp, 1, 0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    b = src[tr.name]
    sid, pose = b["sid"].numpy(), b["pose"].numpy()
    live = sid >= 0
    xy = pose[live][:, :2].astype(float); ksid = sid[live]; yaw = pose[live][:, 2]
    dd = np.linalg.norm(xy[:, None, :] - refp[None], axis=2); j = dd.argmin(1)
    d = np.sign(((xy - refp[j]) * nrm[j]).sum(1)) * dd[np.arange(len(xy)), j]
    s = refs[j]
    cand = np.abs(d) <= NEAR
    cx, cs, cd, ck, cy = xy[cand], s[cand], d[cand], ksid[cand], yaw[cand]
    w = np.exp(-cd ** 2 / (2 * SIG ** 2))
    order = np.argsort(cs); cx, cs, cd, ck, cy, w = cx[order], cs[order], cd[order], ck[order], cy[order], w[order]

    def pick(target, taken):
        for win in (0.5, 1.0, 2.0, 4.0):
            gap = np.abs(((cs - target + 0.5 * L) % L) - 0.5 * L)
            m = (gap <= win) & ~np.isin(np.arange(len(cs)), taken)
            if m.any():
                idx = np.nonzero(m)[0]
                return int(rng.choice(idx, p=w[idx] / w[idx].sum())), win
        raise RuntimeError(f"track {ti}: no candidate near {target:.1f} m")

    ph = (rng.permutation(N) + rng.uniform(0.0, 1.0, N)) / N * L / N_BOX
    out_sid = torch.full((N, C), -1, dtype=torch.long); out_pose = torch.zeros(N, C, 3)
    wins, chosen = [], []
    for i in range(N):
        taken = []
        for jb in range(N_BOX):
            target = (ph[i] + (jb + rng.uniform(0.2, 0.8)) * L / N_BOX) % L
            k, win = pick(target, taken)
            taken.append(k); wins.append(win)
            out_sid[i, jb] = int(ck[k]); out_pose[i, jb] = torch.tensor([cx[k, 0], cx[k, 1], cy[k]])
        chosen.extend(taken)
    chosen = np.array(chosen)
    per[tr.name] = dict(sid=out_sid, pose=out_pose, lap=torch.zeros(N), length=float(b["length"]),
                        refline=torch.tensor(refp, dtype=torch.float32))
    report.append((ti, tr.name, ncar, len(refp), int(live.sum()), int(cand.sum()), len(np.unique(chosen)),
                   100 * np.mean(np.abs(cd[chosen]) <= 0.1), 100 * np.mean(np.abs(cd[chosen]) <= 0.2), 100 * np.mean(np.array(wins) > 0.5)))
    print(f"트랙 {ti:2d} {tr.name[-22:]:>22}: 기준 길 차 {ncar}대 점 {len(refp)}개 | 원래 상자 {int(live.sum())}개 중 길 0.4 m 안 {int(cand.sum())}개 | "
          f"맵 {N}개에 쓴 서로 다른 상자 {len(np.unique(chosen))}개 | 0.1 m 안 {report[-1][7]:.0f} %, 0.2 m 안 {report[-1][8]:.0f} % | "
          f"±0.5 m 안에 후보 없어 넓힌 자리 {report[-1][9]:.1f} %", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(dict(per_track=per, C=C, judge=dict(source=os.path.abspath(SRC), near=NEAR, sig=SIG, n_box=N_BOX, record=os.path.abspath(RECF),
                                               seed=SEED, note="pieces of a judged bank, 3 a layout, composed (line_bank_compose.py)")), OUT)
print(f"== {OUT}: 트랙 {len(per)}개 x 맵 {N}개 x 상자 {N_BOX}개 = 맵 {len(per) * N}개", flush=True)
