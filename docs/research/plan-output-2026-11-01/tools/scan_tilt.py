"""실제 차가 서 있을 때의 스캔으로 LiDAR 장착 기울기 재기: 지도(bag의 /map)와 PF 자세로 빔마다 '벽까지 거리'를 구하고,
측정 거리와 비교. 센서가 아래로 eps 기울면 그 방향 빔은 벽 전에 h/eps 거리에서 바닥에 맞음.
바닥에 맞은 빔이 없으면: 벽에 맞은 빔마다 eps(a) < h / R(a) -> (pitch, roll) 위쪽 경계.
인자: npz [...]"""
import sys
import numpy as np

H = 0.110
ALL_A, ALL_L = [], []
for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True)
    o = z["odom"]; pf = z["pf"]; sc = z["scan"].astype(np.float32); st = z["scan_t"]
    meta = z["scan_meta"]; a = np.arange(int(meta[7])) * meta[2] + meta[0]
    mp = z["map"]; mi = z["map_info"]          # (H, W) int8; resolution, origin x, origin y, ?
    res, ox, oy = float(mi[0]), float(mi[1]), float(mi[2])
    tfs = z["tf_static"]
    # stationary: odom |vx| < 0.01 for the surrounding 1 s
    v_at = np.interp(st, o[:, 0], np.abs(o[:, 5]))
    still = np.array([np.all(np.abs(np.interp(np.linspace(t - 0.5, t + 0.5, 11), o[:, 0], o[:, 5])) < 0.01) for t in st])
    if still.sum() < 20:
        print(fn.split("/")[-1], "정지 스캔", still.sum(), "개 - 건너뜀"); continue
    idx = np.nonzero(still)[0]
    # group into stationary windows (gaps > 0.3 s split)
    splits = np.nonzero(np.diff(st[idx]) > 0.3)[0] + 1
    for grp in np.split(idx, splits):
        if len(grp) < 20: continue
        med = np.nanmedian(np.where(np.isfinite(sc[grp]) & (sc[grp] > 0.03) & (sc[grp] < 30), sc[grp], np.nan), 0)
        t = st[grp].mean()
        x = np.interp(t, pf[:, 0], pf[:, 2]); y = np.interp(t, pf[:, 0], pf[:, 3])
        yaw = np.interp(t, pf[:, 0], np.unwrap(pf[:, 4]))
        lx = x + 0.297 * np.cos(yaw); ly = y + 0.297 * np.sin(yaw)
        occ = (mp >= 50) | (mp < 0)            # occupied or unknown
        # expected range: march every 1 cm
        ang = yaw + a
        d = np.arange(0.05, 30.0, 0.01)
        px = lx + d[None, :] * np.cos(ang)[:, None]; py = ly + d[None, :] * np.sin(ang)[:, None]
        j = np.floor((px - ox) / res).astype(int); i = np.floor((py - oy) / res).astype(int)
        inb = (i >= 0) & (i < occ.shape[0]) & (j >= 0) & (j < occ.shape[1])
        hit = np.where(inb, occ[np.clip(i, 0, occ.shape[0] - 1), np.clip(j, 0, occ.shape[1] - 1)], True)
        R = d[np.argmax(hit, 1)]
        ok = np.isfinite(med)
        short = ok & (med < R - 0.30) & (med > 0.4)
        wall = ok & (np.abs(med - R) < 0.15)
        print(f"{fn.split('/')[-1][:40]} 정지 {len(grp)}개 스캔 @({x:.2f},{y:.2f}) | 빔 {ok.sum()} 돌아옴, 벽 거리와 맞음 {wall.sum()}, "
              f"벽보다 30 cm 넘게 짧음 {short.sum()} (바닥·사람·물체 후보), 안 돌아옴 {(~ok).sum()}")
        # upper bound on the downward tilt per direction: a wall hit at R means eps(a) < H / R
        A = np.stack([np.cos(a[wall]), np.sin(a[wall])], 1)     # eps(a) = pitch*cos(a) + roll*sin(a) (sensor frame, + = down)
        lim = H / R[wall]
        ALL_A.append(A); ALL_L.append(lim)
        # largest |tilt| consistent with every wall beam, by direction
        best = []
        for phi in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            u = np.array([np.cos(phi), np.sin(phi)])
            proj = A @ u
            m = proj > 0.2
            if m.any():
                best.append((np.degrees(np.min(lim[m] / proj[m])), np.degrees(phi)))
        best = np.array(best)
        k = np.argmax(best[:, 0])
        print(f"   벽에 맞은 빔이 허락하는 아래쪽 기울기: 방향마다 최대 {best[:, 0].min():.2f}°-{best[:, 0].max():.2f}° (가장 느슨한 방향 {best[k, 1]:.0f}°)")
        if short.any():
            ss = np.nonzero(short)[0]
            runs = np.split(ss, np.nonzero(np.diff(ss) > 3)[0] + 1)
            for r in runs[:6]:
                if len(r) < 4: continue
                print(f"   짧은 빔 묶음: 각도 {np.degrees(a[r[0]]):+.1f}..{np.degrees(a[r[-1]]):+.1f}° 측정 {np.median(med[r]):.2f} m vs 벽 {np.median(R[r]):.2f} m "
                      f"(바닥이면 아래로 {np.degrees(np.arctan(H / np.median(med[r]))):.2f}°)")

A = np.concatenate(ALL_A); L = np.concatenate(ALL_L)
best = []
for phi in np.linspace(0, 2 * np.pi, 360, endpoint=False):
    u = np.array([np.cos(phi), np.sin(phi)])
    proj = A @ u
    m = proj > 0.3
    best.append(np.degrees(np.min(L[m] / proj[m])) if m.any() else np.inf)
best = np.array(best)
print(f"모든 정지 구간을 합친 경계 (벽에 맞은 빔 {len(L)}개): 어느 방향으로든 아래쪽 기울기 최대 {best.max():.2f}° "
      f"(가장 단단한 방향 {best.min():.2f}°, 중앙값 {np.median(best):.2f}°)")
