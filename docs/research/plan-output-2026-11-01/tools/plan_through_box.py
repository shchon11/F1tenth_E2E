"""(11-17) 상자에 세게 닿기 전에 plan이 얼마나 오래 상자를 지나가고 있었나 -- 보고도 상자를 지나는 plan을 냈나, 마지막 순간에야 냈나
(사용자 11-17: 예선용 학습에 "planner 출력에 벽이나 장애물 + 차 폭까지 해서 벌점"; 그 벌점이 고칠 수 있는 충돌인지 먼저 봄).

frozen exam 기록(exam_record.py npz, ideal = 차가 plan대로 정확히 감) + exam_boxes.py의 통과별 행(EXAM_ROWS):
- 결정마다 plan 10점이 그 상자(자리·방향·모양 그대로)에서 차 반폭 안을 지나면 '상자를 지나는 plan'.
- 통과마다(상자에 가장 가까이 간 순간 t 전 3 s): 상자를 지나는 plan이 이어진 가장 긴 시간, 처음 그런 plan이 나온 때, 상자가 스캔에
  보이기 시작한 때(exam_boxes.py의 값).
- 통과를 나눔: 길 위의 상자에 세게 닿음 / 약하게 닿음 / 안 닿음.
인자: 기록npz 행npy [기록npz 행npy ...]
"""
import sys

import numpy as np

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
BACK = 30                                         # decisions (3 s) before the closest moment


def overlap(pl, pose, fp, half):
    """pl (K, 10, 2) plans, pose (K, 3) the box, fp (4, 2) its footprint (box frame) -> (K,) any plan point within `half` of it"""
    lo, hi = fp.min(0), fp.max(0)
    d = pl - pose[:, None, :2]
    c, s = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
    lx, ly = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
    ex = np.maximum(np.maximum(lo[0] - lx, 0), lx - hi[0])
    ey = np.maximum(np.maximum(lo[1] - ly, 0), ly - hi[1])
    return (np.hypot(ex, ey) < half).any(-1)


args = sys.argv[1:]
for fn, rn in zip(args[0::2], args[1::2]):
    z = np.load(fn, allow_pickle=True)
    rows = np.load(rn)
    pl, pose, sid = z["pl"].astype(float), z["pose"].astype(float), z["sid"]
    dt = float(z["dt"])
    half = 0.5 * float(z["car"][1])
    fps = {int(k[3:]): z[k].astype(float) for k in z.files if k.startswith("fp_")}
    on_line = rows[:, IGAP] < 0.40
    out = []
    for r in rows[on_line]:
        b, j, t = int(r[IB]), int(r[IJ]), int(r[IT])
        k0 = max(0, t - BACK)
        ks = np.arange(k0, t)
        s_ = int(sid[t, b, j])
        if s_ < 0 or s_ not in fps or len(ks) == 0:
            continue
        ov = overlap(pl[ks, b], pose[ks, b, j], fps[s_], half)
        run, best = 0, 0
        for o in ov:
            run = run + 1 if o else 0
            best = max(best, run)
        first = (t - ks[np.argmax(ov)]) * dt if ov.any() else 0.0
        tail = 0
        for o in ov[::-1]:                     # the stretch that runs into the closest moment
            if not o:
                break
            tail += 1
        out.append((r[IHARD] > 0, r[ITOUCH] > 0, best * dt, first, tail * dt, r[ISEE], r[IVIN]))
    o = np.array(out, float).reshape(-1, 7)
    print(f"== {fn.split('/')[-1]} | 차 반폭 {half:.3f} m | 길 위의 상자 통과 {len(o)}번")
    print(f"  {'통과':<14}{'수':>5}{'상자 지나는 plan 낸 통과':>24}{'처음 낸 때 (중앙)':>18}{'가장 가까운 순간까지 이어짐 (중앙)':>32}"
          f"{'상자 보인 때 (중앙)':>20}{'1 s 전 속도':>12}")
    for nm, m in (("세게 닿음", o[:, 0] > 0), ("약하게만 닿음", (o[:, 1] > 0) & (o[:, 0] == 0)), ("안 닿음", o[:, 1] == 0)):
        if not m.any():
            continue
        q = o[m]
        any_ = q[:, 2] > 0
        med = lambda x: float(np.median(x)) if len(x) else float("nan")
        print(f"  {nm:<14}{m.sum():>5}{100 * any_.mean():>20.0f} %{med(q[any_, 3]):>14.2f} s 전{med(q[any_, 4]):>26.2f} s"
              f"{med(q[:, 5]):>16.2f} s 전{med(q[:, 6]):>10.1f} m/s")
    h = o[o[:, 0] > 0]
    if len(h):
        lead = h[:, 5] - h[:, 3]                # > 0: the box was visible before the first plan through it
        print(f"  세게 닿은 통과 {len(h)}번: 상자를 지나는 plan이 보이고 0.3 s 안에 나옴 {int((lead < 0.3).sum())}번, "
              f"보이고 0.3 s 넘게 지나서야 {int((lead >= 0.3).sum())}번 | 닿기 전 0.5 s 넘게 이어짐 {int((h[:, 4] >= 0.5).sum())}번, "
              f"0.2 s 안 {int((h[:, 4] <= 0.2).sum())}번")
