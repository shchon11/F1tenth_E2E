"""기록마다 한 줄: 속도 배율별 자기 선과의 옆 거리 RMSE / 99 % / 급한 굽이에서 안쪽으로 치우친 평균. 인자: 기록 npz ..."""
import sys, os, json
import numpy as np
SKIP = 30
for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True)
    off = z["off"].astype(float); T, B = off.shape
    st = z["st"].astype(float); kap = z["kap"][:T].astype(float); ps = z["pscale"][:T]
    rs = z["rs"][:T] | z["stuck"][:T]
    jump = np.zeros((T, B), bool); jump[1:] = np.hypot(*(st[1:T, :, :2] - st[:T - 1, :, :2]).transpose(2, 0, 1)) > 1.5
    new = jump | np.concatenate([np.ones((1, B), bool), rs[:-1]], 0)
    since = np.zeros((T, B), int)
    for t in range(T):
        since[t] = np.where(new[t], 0, since[t - 1] + 1 if t else 0)
    ok = (since >= SKIP) & ~z["avoid"][:T] & ~z["lost"][:T]
    inward = off * np.sign(kap)
    ms = json.loads(str(z["map_set"])) if "map_set" in z.files and len(str(z["map_set"])) > 4 else {}
    tag = os.path.basename(fn).replace(".npz", "")
    parts = []; tot = 0.0
    for sc in z["scales"]:
        m = ok & (np.abs(ps - sc) < 1e-3)
        rm = np.sqrt((off[m] ** 2).mean()); tot += rm
        tight = m & (np.abs(kap) > 0.5)
        v = np.abs(st[:T, :, 3])[m].mean()
        parts.append(f"{sc:g}배 {rm:.3f} m (99 % {np.percentile(np.abs(off[m]), 99):.2f}, 급한 굽이 안쪽 {inward[tight].mean():+.3f}, {v:.2f} m/s, 다시 출발 {int((rs & (np.abs(ps - sc) < 1e-3)).sum())})")
    keys = ("t_clip_min", "m_l1", "q_l1", "delay_comp")
    print(f"{tag:14s} | " + (" ".join(f"{k}={ms[k]:.3g}" for k in keys if k in ms) or "원본 조향") + f" | 세 배율 RMSE 합 {tot:.3f} | " + " | ".join(parts))
