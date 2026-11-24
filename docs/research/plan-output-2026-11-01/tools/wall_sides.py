"""(11-24, MAP 다시 맞추기) 세게 닿은 벽을 어느 쪽인지로 나눔: 차가 도는 쪽 안의 벽(코너 자름) / 바깥 벽(넓게 돎) / 곧게 갈 때, 그리고
섬 끝(korea26_scan, x -1.3..0.3, y 2.5..4.5: 11-24 탐색에서 L1을 길게 하면 여기 안쪽을 7배 자주 긁음). 혼자·빈 트랙 세션과 상자 세션
따로, 세션당 평균 [95 %]. 닿은 자리 = 그 결정의 접촉 기록(held_x, held_y), 차 = 결정 끝 자세, 도는 쪽 = 앞뒤 결정의 요 각 차이(0.3 rad/s 넘을 때).
인자: 이름=기록 ..."""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rec_lib import Rec

RNG = np.random.default_rng(0)
TIP = (-1.3, 0.3, 2.5, 4.5)


def ci(v):
    v = np.asarray(v, float)
    s = v[RNG.integers(0, len(v), (4000, len(v)))].mean(1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return f"{v.mean():.2f} [{lo:.2f}-{hi:.2f}]"


def classify(fn):
    R = Rec(fn); z = R.z; c = [str(x) for x in z["cols"]]
    hw = R.d[:, :, c.index("hard_wall")] > 0
    st = R.st.astype(float); held = R.held.astype(float)
    T, B = hw.shape
    cls = {k: np.zeros(B) for k in ("inside", "outside", "straight", "tip")}
    for t, b in np.argwhere(hw):
        t1 = min(t + 1, st.shape[0] - 1); t0 = max(t - 1, 0)
        x, y, yaw = st[t1, b, 0], st[t1, b, 1], st[t1, b, 2]
        cx, cy = held[t, b, 1], held[t, b, 2]
        side = np.sign(np.cos(yaw) * (cy - y) - np.sin(yaw) * (cx - x))        # + = the contact is left of the car
        r = np.angle(np.exp(1j * (st[t1, b, 2] - st[t0, b, 2]))) / ((t1 - t0) * R.dt)
        k = "straight" if abs(r) < 0.3 else ("inside" if side == np.sign(r) else "outside")
        cls[k][b] += 1
        if TIP[0] <= cx <= TIP[1] and TIP[2] <= cy <= TIP[3]:
            cls["tip"][b] += 1
    box = (R.sid[0] >= 0).any(-1); opp = R.on[:T].any(0)
    return cls, ~box & ~opp, box & ~opp


if __name__ == "__main__":
    print("세션당 세게 닿은 벽: 안쪽(코너 자름) / 바깥(넓게 돎) / 곧게 / 그중 섬 끝")
    base = None
    for a in sys.argv[1:]:
        tag, fn = a.split("=", 1)
        cls, solo, boxs = classify(fn)
        for nm, m in (("혼자·빈 트랙", solo), ("상자 세션", boxs)):
            print(f"  {tag:<22} {nm:<8} 안쪽 {ci(cls['inside'][m])} | 바깥 {ci(cls['outside'][m])} | 곧게 {ci(cls['straight'][m])} | 섬 끝 {ci(cls['tip'][m])}")
        if base is None:
            base = (cls, solo, boxs)
        else:
            out = []
            for k, nm in (("inside", "안쪽"), ("outside", "바깥"), ("tip", "섬 끝")):
                for g, m in (("빈", solo), ("상자", boxs)):
                    out.append(f"{nm}({g}) {ci(cls[k][m] - base[0][k][m])}")
            print(f"  {'':22} 첫 기록과 세션 짝 차이: " + " · ".join(out))
