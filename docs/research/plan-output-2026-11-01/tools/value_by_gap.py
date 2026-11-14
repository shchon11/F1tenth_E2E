"""critic이 좁게 지나는 통과를 미리 값에 치나 (11-14 밤). 후보 (가) "닿을 뻔한 통과에도 예상 접촉 값"의 반증: critic이 이미
지나기 0.3~1.5 s 전에 좁은 통과(안 닿고 틈 < 10 cm)를 넓은 통과(> 20 cm)보다 낮게 매기면 그 신호는 이미 학습에 있음 -> (가) 버림.
값은 같은 결정의 속도·남은 시간·경우(상자·상대 차 유무)로 설명되는 몫을 빼고(선형 회귀, 모든 결정) 남은 몫으로 비교.
비교 크기: 좁은 통과가 실제로 더 비싼 만큼 = (다음 통과에서 닿을 확률 차) x 한 번 닿을 때 값 -> 대략 0.1 x 6 s = 0.6 s.
인자: 기록npz(F1SIM_EXAM_SAVE_VALUE=1) 상자행npy 차만남행npy 차틈npy"""
import sys
import numpy as np
from rec_lib import Rec
from box_clearance import one as box_one

T_REF = 7.5
fn, box_rows, enc_rows, enc_gap = sys.argv[1:5]
R = Rec(fn); z = R.z
pv = np.load(fn[:-4] + "_priv.npz")
V = pv["value"].astype(float) * T_REF                              # (T, B) seconds
if V.ndim == 3:
    V = V[..., 0]
T, B = V.shape
st = R.st.astype(float)
# what speed / time left / case explain
spd = np.abs(st[:T, :, 3]); tl = np.broadcast_to(np.arange(T)[:, None], (T, B)).astype(float)
case_box = np.broadcast_to((np.asarray(z["sid"])[0] >= 0).any(-1)[None, :], (T, B)).astype(float)
case_opp = np.asarray(z["on"])[:T].astype(float)
X = np.stack([np.ones_like(spd), spd, spd ** 2, tl, tl ** 2, case_box, case_opp, case_box * case_opp], -1).reshape(-1, 8)
ok = np.isfinite(V).reshape(-1)
coef, *_ = np.linalg.lstsq(X[ok], V.reshape(-1)[ok], rcond=None)
res = V - (X @ coef).reshape(T, B)
print(f"값(초): 전체 표준편차 {np.nanstd(V):.2f}, 속도·시간·경우를 뺀 나머지 {np.nanstd(res):.2f}")


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([a, b]))) + 1
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def table(name, t_close, b_idx, groups):
    print(f"== {name}: 가장 가까운 순간 k결정 전의 값 나머지 [s] (평균, 넓은 쪽보다 낮을 확률 = AUC)")
    hdr = "  ".join(f"{g:>22}" for g in groups)
    print(f"  {'k(0.1 s)':>8}  {hdr}")
    for k in (15, 10, 7, 5, 3, 1):
        cells = []
        vals = {}
        for g, m in groups.items():
            tt = t_close[m] - k; bb = b_idx[m]
            okk = tt >= 0
            vals[g] = res[tt[okk], bb[okk]]
        wide = vals[list(groups)[-1]]
        for g in groups:
            x = vals[g]
            a = auc(-x, -wide) if g != list(groups)[-1] else float("nan")
            cells.append(f"{np.nanmean(x):+6.2f} (n {len(x):3d}, AUC {a:.2f})" if g != list(groups)[-1] else f"{np.nanmean(x):+6.2f} (n {len(x):3d})")
        print(f"  {k:>8}  " + "  ".join(f"{c:>22}" for c in cells))


# boxes
rows = np.load(box_rows)
IB, IJ, IT, IN, IGAP = range(5)
_, gap, clean, touch, hard, _ = box_one(fn, box_rows)
rr = rows[rows[:, IGAP] < 0.40]
t_close = rr[:, IT].astype(int); b_idx = rr[:, IB].astype(int)
table("상자 통과", t_close, b_idx, {"세게 닿음": hard, "안 닿음 < 10 cm": clean & (gap < 0.10), "안 닿음 10-20 cm": clean & (gap >= 0.10) & (gap < 0.20),
                                    "안 닿음 > 20 cm": clean & (gap >= 0.20)})
# cars
er = np.load(enc_rows); eg = np.load(enc_gap)
b_idx = er[:, 0].astype(int); t_close = eg[:, 4].astype(int)
hardc = er[:, 4] > 0; cleanc = (er[:, 4] == 0) & (er[:, 5] == 0) & (er[:, 6] == 0) & (er[:, 3] == 0)
table("느린 차 앞지르기", t_close, b_idx, {"세게 닿음": hardc, "안 닿음 < 10 cm": cleanc & (eg[:, 0] < 0.10),
                                         "안 닿음 10-20 cm": cleanc & (eg[:, 0] >= 0.10) & (eg[:, 0] < 0.20), "안 닿음 > 20 cm": cleanc & (eg[:, 0] >= 0.20)})
