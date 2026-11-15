"""기억이 plan을 바꾸나 (11-15, 사용자: "이쯤되면 기억과 plan의 상관관계가 좀 나올 것 같은데").
같은 결정에서 기억을 그대로 둔 plan과 기억을 0으로 둔 plan을 계산(다른 입력은 기록 그대로) -> 차이 = 기억이 만든 plan 변화.
'비킴' = 상자 반대쪽으로 옮긴 거리(1.0 s · 0.5 s 점), '줄임' = 1.0 s 점 속도를 낮춘 양.
  (가) 길 위 상자 통과의 2.5-0.3 s 전 결정: 그 상자 몇 번째 통과인지(1번째 / 2번째 이후) x 상자가 아직 안 보임 / 보임
  (나) 상자를 50 s에 치운 뒤, 치운 상자가 있던 자리가 0.8-4 m 앞(옆 1.5 m 안)인 결정 -- 기억이 상자를 담고 있으면 여기서도 비킴
  (다) 대조: 상자가 없는 세션의 결정에서 무작위 방향으로 잰 같은 값(0 근처여야 함)
기억이 상자를 담는다면: (가)의 2번째 이후 · 안 보일 때 비킴이 +이고 1번째보다 큼, (나)가 + .
입력: 시험 기록(F1SIM_EXAM_SAVE_MEM=1, _mem.npz) + exam_boxes.py 행(EXAM_ROWS). 인자: checkpoint 기록npz 행npy"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec
from policy_lib import limits_for, plan_of, dev
from f1sim.learn.plan_policy import load_actor2

ck_path, npz, rows_fn = sys.argv[1:4]
ck = torch.load(ck_path, map_location=dev, weights_only=False)
lim = limits_for(ck)
actor = load_actor2(ck, dev, check_mount=False).eval()
R = Rec(npz); z = R.z
mm = np.load(npz[:-4] + "_mem.npz")
F = mm["feat"].astype(np.float32); H = mm["mem"].astype(np.float32)
T, B = F.shape[0], F.shape[1]
H0 = np.zeros_like(H); H0[1:] = H[:-1]; H0[1:][R.rs[:T - 1].astype(bool)] = 0.0
dt = R.dt; st = R.st.astype(float)
k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
rows = np.load(rows_fn); rows = rows[rows[:, IGAP] < 0.40]
name = f"{os.path.basename(os.path.dirname(ck_path))} u{int(ck.get('update', 0))}"
RNG = np.random.default_rng(0)


def effect(tb, side):
    """(away 1.0 s, away 0.5 s, slower 1.0 s) of memory vs no memory, for decisions tb (n, 2), away = side * lateral"""
    out = []
    for i in range(0, len(tb), 4096):
        t_, b_ = tb[i:i + 4096, 0], tb[i:i + 4096, 1]
        f = torch.from_numpy(F[t_, b_]).to(dev); h0 = torch.from_numpy(H0[t_, b_]).to(dev)
        with torch.no_grad():
            y, v, _, y5 = plan_of(actor, lim, f, h0); y0, v0, _, y50 = plan_of(actor, lim, f, torch.zeros_like(h0))
        out.append(np.stack([(y - y0).cpu().numpy(), (y5 - y50).cpu().numpy(), (v0 - v).cpu().numpy()], 1))
    d = np.concatenate(out, 0)
    return np.stack([side * d[:, 0], side * d[:, 1], d[:, 2]], 1)


def say(lab, e):
    if len(e) < 10:
        print(f"  {lab}: {len(e)}개 -- 너무 적음"); return
    m = e.mean(0); se = e.std(0, ddof=1) / np.sqrt(len(e))
    print(f"  {lab} ({len(e)}개): 비킴 1.0 s 점 {100 * m[0]:+.1f} ± {100 * se[0]:.1f} cm · 0.5 s 점 {100 * m[1]:+.1f} ± {100 * se[1]:.1f} cm · "
          f"줄임 {m[2]:+.3f} ± {se[2]:.3f} m/s | 기억이 plan을 옮긴 크기(방향 무관) 1.0 s 점 중앙 {100 * np.median(np.abs(e[:, 0])):.1f} cm")


# (가) on-path passes before the closest approach
groups = {}
for r in rows:
    b, j, tc, n, see = int(r[IB]), int(r[IJ]), int(r[IT]), int(r[IN]), float(r[ISEE])
    if tc >= k_rm:
        continue
    box = R.boxes(tc, b)
    for k in range(3, 26):
        t = tc - k
        if t < R.k0[t, b] + 3:
            continue
        x, y, yaw = st[t, b, :3]
        bx, by = float(r[IBX]), float(r[IBY])
        dy = -(bx - x) * np.sin(yaw) + (by - y) * np.cos(yaw)
        if abs(dy) < 0.02:
            continue
        key = ("1번째" if n == 1 else "2번째 이후", "안 보임" if k * dt > see else "보임")
        groups.setdefault(key, []).append((t, b, -np.sign(dy)))
print(f"== {name} ({os.path.basename(npz)}): 기억을 그대로 둔 plan − 기억을 0으로 둔 plan (+ = 상자 반대쪽으로 / 느리게)")
print("(가) 길 위 상자 통과 2.5-0.3 s 전")
for key in (("1번째", "안 보임"), ("1번째", "보임"), ("2번째 이후", "안 보임"), ("2번째 이후", "보임")):
    g = groups.get(key, [])
    if g:
        a = np.array(g); say(f"{key[0]} 통과 · 상자 {key[1]}", effect(a[:, :2].astype(int), a[:, 2]))
# (나) after removal: where a removed box stood, 0.8-4 m ahead
box_w = (R.sid[0] >= 0).any(-1)
gone = []
for b in np.nonzero(box_w)[0]:
    polys = R.boxes(k_rm - 1, b)
    if not polys:
        continue
    C = np.array([p.mean(0) for p in polys])
    for t in range(k_rm + 5, T - 1):
        if abs(st[t, b, 3]) < 1.0:
            continue
        x, y, yaw = st[t, b, :3]; c, s_ = np.cos(yaw), np.sin(yaw)
        ax_ = (C[:, 0] - x) * c + (C[:, 1] - y) * s_; ay_ = -(C[:, 0] - x) * s_ + (C[:, 1] - y) * c
        ok = (ax_ > 0.8) & (ax_ < 4.0) & (np.abs(ay_) < 1.5) & (np.abs(ay_) > 0.02)
        if ok.any():
            i = np.argmin(np.where(ok, ax_, 1e9)); gone.append((t, b, -np.sign(ay_[i])))
print("(나) 상자를 치운 뒤, 치운 상자가 있던 자리 0.8-4 m 앞")
if gone:
    a = np.array(gone); say("치운 상자 자리", effect(a[:, :2].astype(int), a[:, 2]))
# (다) control: sessions without boxes, random direction
ctrl = [(t, b) for b in np.nonzero(~box_w)[0] for t in range(30, T - 1, 7) if abs(st[t, b, 3]) > 1.0]
a = np.array(ctrl); say("대조(상자 없는 세션, 무작위 방향)", effect(a, RNG.choice([-1.0, 1.0], len(a))))

# (라) 0으로 지우는 것은 학습에 없던 상태라 큰 일반적 변화를 만듦 -> 더 공정한 비교: 같은 자리·같은 방향·비슷한 속도의 '상자 없는 세션' 결정에서
#      기억만 가져와 바꿔 넣음. 상자 세션의 기억이 상자를 담고 있으면, 상자 없는 세션의 기억으로 바꿨을 때보다 상자 반대쪽으로 비켜야 함.
from scipy.spatial import cKDTree
emp = [(t, b) for b in np.nonzero(~box_w)[0] for t in range(30, k_rm) if abs(st[t, b, 3]) > 1.0]
E_ = np.array(emp)
key_e = np.stack([st[E_[:, 0], E_[:, 1], 0], st[E_[:, 0], E_[:, 1], 1], 0.5 * np.cos(st[E_[:, 0], E_[:, 1], 2]),
                  0.5 * np.sin(st[E_[:, 0], E_[:, 1], 2]), 0.1 * st[E_[:, 0], E_[:, 1], 3]], 1)
tree = cKDTree(key_e)


def swap_effect(tb, side):
    q = np.stack([st[tb[:, 0], tb[:, 1], 0], st[tb[:, 0], tb[:, 1], 1], 0.5 * np.cos(st[tb[:, 0], tb[:, 1], 2]),
                  0.5 * np.sin(st[tb[:, 0], tb[:, 1], 2]), 0.1 * st[tb[:, 0], tb[:, 1], 3]], 1)
    d, idx = tree.query(q)
    ok = d < 0.15                                                   # same place (~15 cm), heading and speed close
    tb, side, idx = tb[ok], side[ok], idx[ok]
    f = torch.from_numpy(F[tb[:, 0], tb[:, 1]]).to(dev)
    h_box = torch.from_numpy(H0[tb[:, 0], tb[:, 1]]).to(dev); h_emp = torch.from_numpy(H0[E_[idx, 0], E_[idx, 1]]).to(dev)
    with torch.no_grad():
        y, v, _, y5 = plan_of(actor, lim, f, h_box); ye, ve, _, y5e = plan_of(actor, lim, f, h_emp)
    e = np.stack([side * (y - ye).cpu().numpy(), side * (y5 - y5e).cpu().numpy(), (ve - v).cpu().numpy()], 1)
    return e


print("(라) 상자 세션의 기억 − 같은 자리의 상자 없는 세션 기억 (다른 입력은 그대로; + = 상자 반대쪽으로 / 느리게)")
for key in (("1번째", "안 보임"), ("2번째 이후", "안 보임"), ("2번째 이후", "보임")):
    g = groups.get(key, [])
    if g:
        a = np.array(g); say(f"{key[0]} 통과 · 상자 {key[1]}", swap_effect(a[:, :2].astype(int), a[:, 2]))
