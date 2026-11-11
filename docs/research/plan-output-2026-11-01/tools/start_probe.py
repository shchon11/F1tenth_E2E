"""출발 직후와 세게 닿은 뒤를 기록(drive_record.py / unstick_probe.py npz)에서 셈 (11-12: 출발 3 s 안의 세게 닿음이 그 뒤보다 많은가,
무엇 때문인가). 출발 뒤 0-3 / 3-6 / 6-40 s의 세게 닿음/분(벽·상자·상대 차), 출발 자리별(차 앞 벽까지 거리, 앞 상자, 상대 차까지 거리)
3 s 안에 세게 닿은 비율, 벽·상자에 세게 닿은 뒤 후진했는지와 3 m 넘게 벗어났는지(닿은 뒤 차 앞이 얼마나 막혔는지로 나눔).
인자: 기록 npz [기록 npz ...]"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rec_lib as RL
for fn in sys.argv[1:]:
    R = RL.Rec(fn); dt, T, B = R.dt, R.T, R.B
    xy = R.st[:T, :, :2].astype(float); v = R.st[:T, :, 3].astype(float); yaw = R.st[:T, :, 2].astype(float)
    hard = R.d[:, :, 3:6]
    print("====", R.name, f"{B}대 x {T*dt:.0f} s")
    # hard contacts by time since the start
    for a, b_ in ((0, 3), (3, 6), (6, 40)):
        h = hard[int(a/dt):int(b_/dt)].sum((0, 1)); mins = B * (b_ - a) / 60
        print(f"  출발 뒤 {a}~{b_} s: 세게 닿음 벽 {h[0]/mins:.2f} 상자 {h[1]/mins:.2f} 상대 차 {h[2]/mins:.2f} /분  (건수 {h.astype(int)})")
    # start state
    tr = R.tid[0].astype(int)
    ray0 = np.array([R.ray(tr[b], xy[0, b, 0], xy[0, b, 1], yaw[0, b]) for b in range(B)])
    wd0 = np.array([R.wall_dist(tr[b], xy[0, b][None])[0] for b in range(B)])
    v0 = v[0]
    # nearest box ahead within +-0.4 m of the heading line
    nb = np.full(B, 99.0)
    for b in range(B):
        m = R.sid[0, b] >= 0
        if m.any():
            dxy = R.pose[0, b, m, :2] - xy[0, b]
            fx = dxy[:, 0] * np.cos(yaw[0, b]) + dxy[:, 1] * np.sin(yaw[0, b]); fy = -dxy[:, 0] * np.sin(yaw[0, b]) + dxy[:, 1] * np.cos(yaw[0, b])
            ok = (fx > 0) & (np.abs(fy) < 0.45)
            if ok.any():
                nb[b] = fx[ok].min()
    od = np.where(R.on[0], np.linalg.norm(R.op[0, :, :2] - xy[0], axis=1), 99.0)
    hw3 = hard[:30, :, 0].sum(0) > 0; hb3 = hard[:30, :, 1].sum(0) > 0; hc3 = hard[:30, :, 2].sum(0) > 0
    print(f"  출발 속도 중앙 {np.median(v0):.2f} (10~90 % {np.percentile(v0,10):.2f}~{np.percentile(v0,90):.2f}) m/s | 차 앞 벽까지 중앙 {np.median(ray0):.1f} m, 2 m 안 {np.mean(ray0<2)*100:.0f} %, 1 m 안 {np.mean(ray0<1)*100:.0f} % | 옆 벽까지 중앙 {np.median(wd0):.2f} m")
    print(f"  출발 3 s 안에 세게 닿은 차: 벽 {hw3.sum()} · 상자 {hb3.sum()} · 상대 차 {hc3.sum()} / {B}")
    for lab, m in (("차 앞 벽 1.5 m 안", ray0 < 1.5), ("1.5~3 m", (ray0 >= 1.5) & (ray0 < 3)), ("3 m 넘게", ray0 >= 3)):
        print(f"    {lab}: {m.sum()}대, 3 s 안 벽에 세게 {hw3[m].sum()} ({hw3[m].mean()*100:.1f} %)")
    for lab, m in (("차 앞 상자 2 m 안", nb < 2), ("2~4 m", (nb >= 2) & (nb < 4)), ("없음/4 m 넘게", nb >= 4)):
        print(f"    {lab}: {m.sum()}대, 3 s 안 상자에 세게 {hb3[m].sum()} ({hb3[m].mean()*100:.1f} %)")
    for lab, m in (("상대 차 1.5 m 안", od < 1.5), ("1.5~4 m", (od >= 1.5) & (od < 4)), ("4 m 넘게/없음", od >= 4)):
        print(f"    {lab}: {m.sum()}대, 3 s 안 상대 차에 세게 {hc3[m].sum()} ({hc3[m].mean()*100:.1f} %)")
    for b in np.nonzero(hw3)[0]:
        t0 = int(np.nonzero(hard[:30, b, 0])[0][0])
        vv = v[:t0 + 1, b]
        print(f"    벽 건 차 {b}: 닿은 때 {t0*dt:.1f} s, 출발 속도 {v0[b]:.2f}, 차 앞 벽 {ray0[b]:.1f} m, 옆 벽 {wd0[b]:.2f} m, 닿기 직전 속도 {vv[-1]:.2f}, 최고 {vv.max():.2f}, 상대 차 {od[b]:.1f} m, 앞 상자 {nb[b]:.1f} m, 그 뒤 3 m 벗어남 {'예' if (np.linalg.norm(xy[t0:, b]-xy[t0, b],axis=1)>=3).any() else '아니오'}")
    # recovery: what separates the ones that get away
    print("  [벽·상자에 세게 닿은 뒤 벗어났나]")
    rows = []
    for b in range(B):
        ev = np.nonzero(hard[:, b, :2].sum(1) > 0.5)[0]; last = -99
        for t0 in ev:
            if t0 - last < 30 or t0 < 3 or T - t0 < 80:
                continue
            last = t0
            disp = np.linalg.norm(xy[t0:, b] - xy[t0, b], axis=1)
            away = (disp >= 3.0).any()
            ta = int(np.argmax(disp >= 3.0)) if away else T - t0
            rev1 = (v[t0:t0 + 15, b] < -0.2).any()
            ray_a = R.ray(int(R.tid[t0, b]), xy[t0 + 5, b, 0], xy[t0 + 5, b, 1], yaw[t0 + 5, b])
            fwd = ((R.pl[t0 + 5:t0 + 15, b, 9, 0] - xy[t0 + 5:t0 + 15, b, 0]) * np.cos(yaw[t0 + 5:t0 + 15, b]) + (R.pl[t0 + 5:t0 + 15, b, 9, 1] - xy[t0 + 5:t0 + 15, b, 1]) * np.sin(yaw[t0 + 5:t0 + 15, b])).mean()
            rows.append((away, ta * dt, rev1, ray_a, fwd, v[t0 - 2, b], t0 * dt, hard[t0, b, :2].argmax()))
    A = np.array(rows, float)
    for lab, m in (("1.5 s 안에 후진함", A[:, 2] > 0.5), ("1.5 s 안에 후진 안 함", A[:, 2] < 0.5)):
        s = A[m]
        if len(s):
            print(f"    {lab}: {len(s)}건, 벗어남 {s[:,0].mean()*100:.0f} %, 벗어나는 데 중앙 {np.median(s[s[:,0]>0.5,1]) if (s[:,0]>0.5).any() else float('nan'):.1f} s, 못 벗어남 {int((s[:,0]<0.5).sum())}건 | 닿은 뒤 0.5 s에 차 앞 막힌 거리 중앙 {np.median(s[:,3]):.2f} m | 닿기 전 속도 중앙 {np.median(s[:,5]):.1f}")
    for lab, m in (("닿은 뒤 차 앞이 0.15 m 안(코가 막힘)", A[:, 3] < 0.15), ("0.15~0.5 m", (A[:, 3] >= 0.15) & (A[:, 3] < 0.5)), ("0.5 m 넘게", A[:, 3] >= 0.5)):
        s = A[m]
        if len(s):
            print(f"    {lab}: {len(s)}건, 1.5 s 안 후진 {s[:,2].mean()*100:.0f} %, 못 벗어남 {int((s[:,0]<0.5).sum())}건, plan의 1.0 s 점이 앞으로 중앙 {np.median(s[:,4]):+.2f} m")
