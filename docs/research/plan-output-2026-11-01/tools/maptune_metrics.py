"""(11-24, 사용자: "제어기 튜닝 이상하게 된거 아님? 존나 흔들리네") MAP이 policy의 plan을 얼마나 그대로 따라가나 -- 시험 기록(exam_record.py) 하나의
점수. map_tune_scan.py가 탐색 목표로, 표로 씀.

- 따라가기 오차(목표, 10-20 규칙): 결정 t에 낸 plan(10점, 0.1 s 간격)과 k x 0.1 s 뒤(k = 1..5) 실제 위치의 거리(점 대 점, 속도 늦음 포함)와
  plan 경로까지의 옆 거리. plan 첫 구간 속도 2 m/s 넘는 결정만, 도중 다시 출발한 줄은 뺌.
  목표값 = 칸(plan 속도 x plan 옆가속도)마다 k = 1..5 점 대 점 RMSE, 지금 설정(c000)의 칸 수로 가중 평균 -- 느리게 달리면 오차가 작아서
  느린 설정이 이기는 것을 막음 (map_tune.py, 11-15와 같음).
- 보고만 하는 것: 요 각속도 흔들림(요 각속도 - 0.5 s 가운데 평균의 RMS, 2 m/s 넘게 달릴 때, 5 s 뒤부터, 혼자·빈 트랙 세션),
  미끄럼각 |atan2(vy, vx)| 90 %(2 m/s 넘을 때, 모든 세션), 혼자·빈 트랙 한 바퀴(세션 중앙값들의 중앙), 세션당 세게 닿음(벽·상자)."""
import numpy as np
from scipy.ndimage import uniform_filter1d

VB = (2.0, 4.0, 5.5, 7.0, 10.0)
AB = (0.0, 2.0, 4.0, 6.0, 20.0)
KS = (1, 2, 3, 4, 5)


def seg_dist(p, a, b):
    """distance from points p (N, 2) to the polylines a->b (N, S, 2): min over S"""
    ab = b - a; ap = p[:, None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(-1)


def laps(lapsig, dt):
    """per session: the lap times (a lap ends when the net count of line crossings reaches a new high; exam_table.py's rule)"""
    net = np.cumsum(lapsig, 0); out = []
    for b in range(lapsig.shape[1]):
        hi, t_prev, x = 0, None, []
        for t in np.nonzero(lapsig[:, b] > 0)[0]:
            if net[t, b] > hi:
                hi = net[t, b]
                if t_prev is not None:
                    x.append((t + 1 - t_prev) * dt)
                t_prev = t + 1
        out.append(x)
    return out


def score(fn, W=None):
    z = np.load(fn, allow_pickle=False)
    dt = float(z["dt"]); rs = z["rs"]; T, B = rs.shape
    st = z["st"].astype(np.float64); pl = z["pl"].astype(np.float64); vy = z["vy"].astype(np.float64)
    c = [str(x) for x in z["cols"]]; cnt = z["cnt"].astype(float)
    sid = z["sid"]; on = z["on"][:T]
    box = (sid[0] >= 0).any(-1); opp = on.any(0); solo = ~box & ~opp
    # restarted within the next k decisions: cumulative count trick
    rsc = np.concatenate([np.zeros((1, B)), np.cumsum(rs, 0)], 0)               # rsc[t] = restarts before decision t
    pos = {k: [] for k in KS}; lat = {k: [] for k in KS}; v0s, als, solo_m = [], [], []
    dv = []
    for t in range(T - max(KS)):
        path = np.concatenate([st[t, :, None, :2], pl[t]], 1)                   # (B, 11, 2)
        v0 = np.linalg.norm(pl[t, :, 0] - st[t, :, :2], axis=-1) / dt
        ok = (rsc[t + max(KS) + 1] - rsc[t] == 0) & (v0 > 2.0)
        if not ok.any():
            continue
        for k in KS:
            pos[k].append(np.linalg.norm(st[t + k, ok, :2] - pl[t, ok, k - 1], axis=-1))
            lat[k].append(seg_dist(st[t + k, ok, :2], path[ok, :-1], path[ok, 1:]))
        seg = path[ok, 1:6] - path[ok, :5]
        vv = np.linalg.norm(seg, axis=-1) / dt
        h = np.arctan2(seg[..., 1], seg[..., 0]); turn = np.abs(np.angle(np.exp(1j * (h[:, -1] - h[:, 0]))))
        als.append(vv.mean(1) ** 2 * turn / np.maximum(np.linalg.norm(seg, axis=-1).sum(1), 0.05))
        v0s.append(v0[ok]); solo_m.append(solo[ok])
        vp = np.linalg.norm(pl[t, ok, 2] - pl[t, ok, 1], axis=-1) / dt        # plan speed of 0.2-0.3 s
        dv.append(st[t + 3, ok, 3] - vp)
    r = {}
    for k in KS:
        r[f"pos{k}"] = float(np.sqrt(np.mean(np.concatenate(pos[k]) ** 2)))
        r[f"lat{k}"] = float(np.sqrt(np.mean(np.concatenate(lat[k]) ** 2)))
    sm = np.concatenate(solo_m)
    r["pos5_solo"] = float(np.sqrt(np.mean(np.concatenate(pos[5])[sm] ** 2)))
    r["lat5_p90"] = float(np.percentile(np.concatenate(lat[5]), 90))
    r["dv_med"] = float(np.median(np.concatenate(dv)))
    e2 = np.mean(np.stack([np.concatenate(pos[k]) ** 2 for k in KS], 1), 1)     # per decision: mean over k of e^2
    v0 = np.concatenate(v0s); al = np.concatenate(als)
    iv = np.digitize(v0, VB) - 1; ia = np.digitize(al, AB) - 1
    cell = np.full((len(VB) - 1, len(AB) - 1), np.nan); n = np.zeros_like(cell)
    for i in range(len(VB) - 1):
        for j in range(len(AB) - 1):
            m = (iv == i) & (ia == j); n[i, j] = m.sum()
            if m.sum() >= 50:
                cell[i, j] = np.sqrt(np.mean(e2[m]))
    r["cells"] = cell.tolist(); r["cell_n"] = n.tolist()
    w = np.array(W if W is not None else n, float)
    r["obj"] = float((np.where(np.isnan(cell), 0.5, cell) * w).sum() / w.sum())
    # wobble (solo, empty track), slip
    t0 = int(round(5 / dt)); wn = int(round(0.5 / dt))
    def wob(cols):
        s = st[:T][:, cols]
        yaw = np.unwrap(s[:, :, 2], axis=0); rr = np.diff(yaw, axis=0) / dt
        hf = rr - uniform_filter1d(rr, wn, axis=0, mode="nearest")
        mov = s[1:, :, 3] > 2.0
        x = hf[t0:][mov[t0:]]
        return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")
    r["wobble_solo"] = wob(np.nonzero(solo)[0]); r["wobble_all"] = wob(np.arange(B))
    vx = st[:T, :, 3]; mv = vx > 2.0
    sl = np.degrees(np.abs(np.arctan2(vy[:T][mv], vx[mv])))
    r["slip_p90"] = float(np.percentile(sl, 90))
    r["v_mean"] = float(vx.mean())
    lp = laps(z["lap"], dt)
    lm = np.array([np.median(x) if x else np.nan for x in lp])
    r["lap_solo"] = float(np.nanmedian(lm[solo])) if np.isfinite(lm[solo]).any() else float("nan")
    r["lap_solo_per"] = lm[solo].tolist()
    h = cnt[-1]
    for nm in ("hard_wall", "hard_prop"):
        r[nm] = float(h[:, c.index(nm)].mean())
        r[nm + "_solo"] = float(h[solo, c.index(nm)].mean())
        r[nm + "_box"] = float(h[box & ~opp, c.index(nm)].mean())
    r["hw_solo_per"] = h[solo, c.index("hard_wall")].tolist()
    r["hw_box_per"] = h[box & ~opp, c.index("hard_wall")].tolist()
    r["hp_box_per"] = h[box & ~opp, c.index("hard_prop")].tolist()
    r["n_solo"] = int(solo.sum()); r["n_box"] = int((box & ~opp).sum()); r["secs"] = T * dt
    return r


if __name__ == "__main__":
    import sys, json
    for a in sys.argv[1:]:
        d = score(a)
        print(a.rsplit("/", 1)[-1], json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
                                               if not isinstance(v, list)}))
