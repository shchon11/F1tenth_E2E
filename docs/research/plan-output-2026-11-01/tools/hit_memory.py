"""(11-18, 사용자 최소 기준 "박는게 있어도 다음바퀴에는 안박아야지") 지난 통과에 세게 박은 상자에 다음 통과로 다가갈 때 기억이 그 일을
들고 오나, 들고 와도 policy가 읽나. F1SIM_EXAM_SAVE_MEM=1 시험 기록(결정마다 기억이 읽은 특징 feat과 그 뒤 상태 mem) + exam_boxes.py가
저장한 통과별 행(EXAM_ROWS). 같은 세션·같은 상자의 이어진 두 통과(k, k+1)마다, 각 통과에서 상자 중심이 스캔에 보이기 시작하기 0.3 s
전(상자를 아직 못 봄)의 순간을 잡아:
 (1) 두 순간의 기억 상태 차(값마다 기록 전체의 표준편차로 나눔, 값 수의 제곱근으로 나눔): k에서 세게 박은 짝 vs k에서 안 닿은 짝.
     박은 짝이 더 다르면 = 박은 일이 기억에 담김. 같은 비교를 '혼자 두면 한 바퀴(7 s) 넘게 남는 값'만으로도.
 (2) k+1 순간에 기억만 k 순간 직전의 것으로 바꿔 넣고(입력은 그대로) plan 출력이 변하는 정도: 같은 두 무리. 박은 짝에서 더 크게
     변하면 = 담긴 것을 policy가 읽음.
두 통과 사이 시간도 냄(박은 뒤 멈춰 있던 시간이 길면 그것만으로 기억이 달라질 수 있음).
인자: checkpoint 기록npz 행npy"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn.plan_policy import load_actor2

torch.set_grad_enabled(False)
ck_path, rec, rows_path = sys.argv[1:4]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
actor = load_actor2(ck, torch.device("cpu"), check_mount=False).eval()
gru = actor.gru; n = gru.hidden_size
M = np.load(rec[:-4] + "_mem.npz"); Z = np.load(rec, allow_pickle=True)
feat = torch.tensor(M["feat"].astype(np.float32)); mem = torch.tensor(M["mem"].astype(np.float32))
rs = np.asarray(Z["rs"]).astype(bool); T, B = rs.shape; DT = float(Z["dt"])
rows = np.load(rows_path)
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV = range(10)

# per value, how long it keeps itself along this drive (as mem_reach.py): the values that outlast a lap
Wi, Wh, bi, bh = gru.weight_ih, gru.weight_hh, gru.bias_ih, gru.bias_hh
zs = []
for t in range(1, T, 3):
    ok = torch.tensor(~rs[t - 1])
    x, h = feat[t][ok], mem[t - 1][ok]
    zs.append(torch.sigmoid((x @ Wi.T + bi)[:, n:2 * n] + (h @ Wh.T + bh)[:, n:2 * n]))
tau = DT / np.clip(1.0 - torch.cat(zs).mean(0).numpy(), 1e-6, None)
long_ = torch.tensor(tau > 7.0)
sd = mem.reshape(-1, n).std(0).clamp_min(1e-4)
chk = (gru(feat[200], mem[199]) - mem[200]).abs().max().item()


def plan(t, b, h_prev):
    """the plan output (10 turns, 10 speeds) at decision t of car b, from the memory state before it"""
    f, _ = actor.remember(feat[t, b][None], h_prev[None])
    return actor.heads(f)[0][0]


P = []
for (b, j) in sorted(set(map(tuple, rows[:, [IB, IJ]].astype(int).tolist()))):
    p = rows[(rows[:, IB] == b) & (rows[:, IJ] == j)]
    p = p[np.argsort(p[:, IN])]
    for a, c in zip(p[:-1], p[1:]):
        if c[IN] != a[IN] + 1 or a[ISEE] <= 0 or c[ISEE] <= 0:
            continue
        kind = 1 if a[IHARD] > 0 else (0 if a[ITOUCH] == 0 else -1)
        if kind < 0:
            continue
        ta = int(a[IT]) - int(round(a[ISEE] / DT)) - 3
        tc = int(c[IT]) - int(round(c[ISEE] / DT)) - 3
        if ta < 2 or tc <= ta or rs[ta:tc + 1, b].any():
            continue
        dv = (mem[tc, b] - mem[ta, b]) / sd
        dp = (plan(tc, b, mem[tc - 1, b]) - plan(tc, b, mem[ta - 1, b])).abs()
        P.append((kind, float(dv.norm() / np.sqrt(n)), float(dv[long_].norm() / np.sqrt(max(int(long_.sum()), 1))),
                  float(dp[:10].mean()), float(dp[10:].mean()), float(dp[0]), float(dp[10]), (tc - ta) * DT, float(c[IHARD] > 0)))
P = np.array(P)


def auc(s, y):
    """P(a hit pair's value > a clean pair's)"""
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return (r[y > 0].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


y = P[:, 0]
print(f"{ck_path.split('/')[-2]}/{ck_path.split('/')[-1]} | 기록된 기억과 다시 굴린 기억의 차 최대 {chk:.3f} (float16 저장) | "
      f"한 바퀴 넘게 남는 기억 값 {int(long_.sum())}개")
print(f"짝: 앞 통과에서 세게 박음 {int(y.sum())}쌍, 안 닿음 {int((1 - y).sum())}쌍 (같은 세션·같은 상자, 상자가 보이기 0.3 s 전끼리)")
names = ["기억 상태 차 (값 128개)", "기억 상태 차 (한 바퀴 넘게 남는 값)", "기억을 앞 통과 것으로 바꿀 때 plan 회전 출력 변화 (10점 평균)",
         "같은 때 plan 속도 출력 변화 (10점 평균)", "  첫 점 회전 출력 변화", "  첫 점 속도 출력 변화", "두 통과 사이 시간 (s)"]
for k, nm in enumerate(names):
    s = P[:, 1 + k]
    print(f"  {nm}: 박은 짝 중앙 {np.median(s[y > 0]):.4f} | 안 닿은 짝 중앙 {np.median(s[y == 0]):.4f} | "
          f"박은 짝이 더 클 확률(AUC) {auc(s, y):.2f}")
hit = y > 0
print(f"박은 짝의 다음 통과에서 또 세게 박음: {int(P[hit, 8].sum())}/{int(hit.sum())}")
# a hit stops the car: the next pass comes ~2 s later, which alone moves the memory. Compare each hit pair with the clean
# pairs whose gap between the passes is within 1 s of its own (the median over them), so the stop is in both
gap = P[:, 7]
for k, nm in ((0, "기억 상태 차 (값 128개)"), (1, "기억 상태 차 (한 바퀴 넘게 남는 값)"), (3, "plan 속도 출력 변화 (10점 평균)")):
    s = P[:, 1 + k]; above, ref = [], []
    for i in np.nonzero(hit)[0]:
        m = (~hit) & (np.abs(gap - gap[i]) <= 1.0)
        if m.sum() >= 5:
            r = np.median(s[m]); ref.append(r); above.append(s[i] > r)
    print(f"  두 통과 사이 시간을 ±1 s로 맞춘 안 닿은 짝과: {nm} -- 박은 짝이 맞춘 짝들의 중앙보다 큰 비율 {100 * np.mean(above):.0f} % "
          f"({len(above)}쌍; 차이가 없으면 50 %), 박은 짝 중앙 {np.median(s[hit]):.4f} vs 맞춘 짝 {np.median(ref):.4f}")
# the direction: plan speeds with the memory it really had minus with the memory from before the last pass
sgn = []
for (b, j) in sorted(set(map(tuple, rows[:, [IB, IJ]].astype(int).tolist()))):
    p = rows[(rows[:, IB] == b) & (rows[:, IJ] == j)]
    p = p[np.argsort(p[:, IN])]
    for a, c in zip(p[:-1], p[1:]):
        if c[IN] != a[IN] + 1 or a[ISEE] <= 0 or c[ISEE] <= 0 or a[IHARD] <= 0:
            continue
        ta = int(a[IT]) - int(round(a[ISEE] / DT)) - 3
        tc = int(c[IT]) - int(round(c[ISEE] / DT)) - 3
        if ta < 2 or tc <= ta or rs[ta:tc + 1, b].any():
            continue
        d = plan(tc, b, mem[tc - 1, b]) - plan(tc, b, mem[ta - 1, b])
        sgn.append((float(d[10:].mean()), float(d[10]), float(d[:10].mean())))
sgn = np.array(sgn)
print(f"  박은 짝, 실제 기억 - 앞 통과 기억: plan 속도 출력 평균 {sgn[:, 0].mean():+.4f} (줄이는 쪽 {100 * (sgn[:, 0] < 0).mean():.0f} %), "
      f"첫 점 속도 {sgn[:, 1].mean():+.4f}, 회전 출력 평균 {sgn[:, 2].mean():+.4f} | 속도 출력 1 = 한 단계(0.1 s)에 낼 수 있는 속도 폭의 절반")
