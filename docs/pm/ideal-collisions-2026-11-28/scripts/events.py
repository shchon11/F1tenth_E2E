"""이상 실행 기록에서 (1) 상자 통과(만남) 전부, (2) 세게 닿은 상자 사건 전부를 사건 단위로 잼. CPU만.

만남(encounter): 세션 b의 상자 j(살아 있을 때)에 차 중심이 2.0 m 안으로 들어온 연속 구간. 구간마다 촘촘한 궤적(결정당 10칸)으로
차체-상자 부호 있는 최소 여유, 그때 속도, 지나간 쪽(차 진행 방향 기준 상자가 왼쪽 +1 / 오른쪽 -1), 그 구간의 상자 접촉(약한 것
포함 / 세게)을 셈. 접촉은 그 결정에서 차체에 가장 가까운 상자에 붙임.

세게 닿은 사건(hard box onset)마다:
  - 가시성: 닿기 전 최대 4 s(또는 같은 상자의 앞 만남이 끝난 뒤부터) 결정마다 그 상자에 맞는 LiDAR 빔 수 (ev_lib.Caster).
    처음 보인 시각 = 닿을 때까지 이어서(1결정 끊김 허용) 빔 >= 2개인 구간의 시작. 그때 스캐너-상자 거리, 속도, 닿을 때까지 간 거리.
  - 물리적 회피 가능성 (이상 실행 모형, 작은 각 근사): 처음 보인 자세에서 닿은 자세까지 실제로 간 길을 기준선으로,
    닿은 자세에서 상자를 왼쪽/오른쪽으로 비키는 데 필요한 옆 이동 y_req (차 반폭 포함, 여유 0), 그쪽 벽까지 빈 폭 >= 차폭인지,
    그리고 그 거리 D 안에서 낼 수 있는 최대 옆 이동 y_max = int_0^D (D-s) dk(s) ds, dk = 곡률 한계(속도) -/+ 길의 곡률.
    두 가지: (a) 속도 그대로 조향만, (b) 처음부터 5 m/s^2 제동 + 조향(곡률 한계가 속도에 따라 커짐). 또 제동만으로 서는가.
    반응 지연 0 s(처음 보인 결정이 바로 반응)와 0.3 s 두 가지.
  - 계획: 처음 보인 때부터 닿을 때까지 결정마다 그 결정의 계획(10점)을 따라간 차체의 상자 여유와 계획이 상자 자리까지 닿는지.
  - 같은 상자의 앞 만남들(바퀴): 최소 여유, 접촉, 쪽.
  - 같은 만남 안에서 이미 세게 닿았나(끼여서 다시 박음).
  - 벽까지 빈 폭(상자 양옆), 닿을 때 속도.
인자: out.json rec.npz[:라벨] ...  (환경 변수 EV_MAXB = 처리할 세션 수, 시험용)"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import (Rec, Caster, body_poly, signed_clear, dense_traj, plan_poses, ray_free, kmax, load_limits, N_SUB)

LIM = load_limits()
NEAR = 2.0
MAXB = int(os.environ.get("EV_MAXB", "100000"))


def encounters(R, b, rm_k):
    """list of dicts for every (box, encounter) of session b before removal decision rm_k"""
    st = R.st.astype(float)
    out = []
    T = min(R.T, rm_k)
    polys0 = R.boxes(0, b)
    for j, P in enumerate(polys0):
        c = P.mean(0)
        dist = np.linalg.norm(st[:T, b, :2] - c[None], axis=1)
        near = dist < NEAR
        if not near.any():
            continue
        idx = np.nonzero(near)[0]
        # split into runs
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for r in runs:
            t0, t1 = max(0, r[0] - 1), min(T - 1, r[-1] + 1)
            tr, tt = dense_traj(st, t0, t1, b)
            body = body_poly(tr[:, 0], tr[:, 1], tr[:, 2], R.hl, R.hw)
            cl = signed_clear(body, P)
            i = int(np.argmin(cl))
            tm = tt[i]
            k = int(np.floor(tm))
            # side: box centre relative to the car heading at the closest moment
            hx, hy = np.cos(tr[i, 2]), np.sin(tr[i, 2])
            rx, ry = c[0] - tr[i, 0], c[1] - tr[i, 1]
            side = 1 if hx * ry - hy * rx > 0 else -1
            out.append(dict(b=int(b), j=int(j), t0=int(t0), t1=int(t1), tmin=float(tm), clear=float(cl[i]),
                            v=float(abs(st[min(k, R.T), b, 3])), side=side, cx=float(c[0]), cy=float(c[1]),
                            any=0, hard=0, hard_t=[], any_t=[]))
    return out


def attach_contacts(R, b, encs, rm_k):
    """box onsets (any col 1, hard col 4) of session b -> the encounter of the nearest box at that decision"""
    st = R.st.astype(float)
    for col, key in ((1, "any"), (4, "hard")):
        for t in np.nonzero(R.d[:rm_k, b, col] > 0)[0]:
            polys = R.boxes(t, b)
            if not polys:
                continue
            body = body_poly(*st[min(t + 1, R.T), b, :3], R.hl, R.hw)[None]
            j = int(np.argmin([signed_clear(body, P)[0] for P in polys]))
            # the encounter of box j containing t (or the nearest in time)
            cand = [e for e in encs if e["j"] == j]
            if not cand:
                continue
            e = min(cand, key=lambda e: 0 if e["t0"] <= t <= e["t1"] else min(abs(t - e["t0"]), abs(t - e["t1"])))
            e[key] += 1; e[key + "_t"].append(int(t))


def frenet_ok(tr, s, v0, D, yreq, room_ok, kpath, delay_s, brake):
    """max lateral shift reachable within D (small angle) toward each side; returns (ok, ymax_left, ymax_right)"""
    d0 = v0 * delay_s
    De = D - d0
    if De <= 0:
        return False, 0.0, 0.0
    m = (s >= d0) & (s <= D)
    ss = s[m] - d0
    kp = kpath[m]
    if brake:
        vv = np.sqrt(np.maximum(v0 ** 2 - 2 * LIM["a_brake"] * ss, 0.3 ** 2))
    else:
        vv = np.full_like(ss, v0)
    km = kmax(vv, LIM)
    # servo ramp: curvature rate limit ~ sv_max / wheelbase per s -> distance to reach full deviation
    w = np.clip(ss / max(v0 * 0.05, 1e-3), 0, 1)
    dkl = np.maximum(km - kp, 0) * w
    dkr = np.maximum(km + kp, 0) * w
    if len(ss) < 2:
        return False, 0.0, 0.0
    yl = np.trapz((De - ss) * dkl, ss)
    yr = np.trapz((De - ss) * dkr, ss)
    ok = (room_ok[0] and yl >= yreq[0]) or (room_ok[1] and yr >= yreq[1])
    if brake and v0 ** 2 / (2 * LIM["a_brake"]) <= De:
        ok = True
    return bool(ok), float(yl), float(yr)


def hard_events(R, C, b, encs, rm_k, mount_x):
    st = R.st.astype(float)
    out = []
    for e in encs:
        for n_h, t in enumerate(e["hard_t"]):
            j = e["j"]
            polys = R.boxes(t, b)
            P = polys[j]
            prev = [q for q in encs if q["j"] == j and q["t1"] < e["t0"]]
            # visibility back to max 4 s, not before the previous encounter of this box ended
            tlo = max(0, t - 40, (max(q["t1"] for q in prev) + 1) if prev else 0)
            nvis = np.zeros(t - tlo + 1, int); dsc = np.zeros(t - tlo + 1)
            for i, tt in enumerate(range(tlo, t + 1)):
                nvis[i], dsc[i] = C.beams_on(st[tt, b, :3], polys, j, mount_x)
            vis = nvis >= 2
            # the visible run ending at the hit (one-decision gaps allowed)
            i = len(vis) - 1
            first = None
            gap = 0
            while i >= 0:
                if vis[i]:
                    first = i; gap = 0
                else:
                    gap += 1
                    if gap > 1:
                        break
                i -= 1
            any_first = int(np.nonzero(vis)[0][0]) if vis.any() else None
            t_vis = tlo + first if first is not None else t
            # the driven path from t_vis to t (the hit decision): dense
            tr, ttd = dense_traj(st, t_vis, t, b)
            seg = np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            yaw_u = np.unwrap(tr[:, 2])
            kpath = np.gradient(yaw_u, s, edge_order=1) if len(s) > 2 and s[-1] > 0.05 else np.zeros_like(s)
            kpath = np.nan_to_num(np.convolve(kpath, np.ones(5) / 5, mode="same"))
            D = float(s[-1])
            v0 = float(abs(st[t_vis, b, 3]))
            # required shift at the hit pose (frame of the car at the hit decision)
            x, y, yaw = st[t, b, :3]
            c, s_ = np.cos(yaw), np.sin(yaw)
            loc = np.stack([(P[:, 0] - x) * c + (P[:, 1] - y) * s_, -(P[:, 0] - x) * s_ + (P[:, 1] - y) * c], 1)
            yreq_l = max(loc[:, 1].max() + R.hw, 0.0)      # move left so the car's right edge clears the box's left-most point
            yreq_r = max(R.hw - loc[:, 1].min(), 0.0)      # move right
            nl = np.array([-s_, c])
            pl_ = P[np.argmax(loc[:, 1])]; pr_ = P[np.argmin(loc[:, 1])]
            room_l = ray_free(C, pl_ + 0.01 * nl, nl); room_r = ray_free(C, pr_ - 0.01 * nl, -nl)
            room_ok = (room_l >= 2 * R.hw, room_r >= 2 * R.hw)
            feas = {}
            for dl in (0.0, 0.3):
                for br in (False, True):
                    feas[f"d{dl}_{'brake' if br else 'steer'}"] = frenet_ok(tr, s, v0, D, (yreq_l, yreq_r), room_ok, kpath, dl, br)
            # plans from t_vis to t
            pc = []
            for tt in range(max(t_vis, t - 30), t + 1):
                pp = plan_poses(st[tt, b, :3], R.pl[tt, b].astype(float))
                body = body_poly(pp[:, 0], pp[:, 1], pp[:, 2], R.hl, R.hw)
                cl = signed_clear(body, P)
                im = int(np.argmin(cl))
                reach = (cl[im] <= 0.0) or (im < len(cl) - 1)
                pc.append((int(tt), float(cl.min()), bool(reach)))
            # contiguous run of plans through the box ending at the hit
            thr = 0.0
            k = len(pc) - 1
            while k >= 0 and pc[k][1] <= thr:
                k -= 1
            plan_through_from = pc[k + 1][0] if k + 1 < len(pc) else None
            reach_from = next((q[0] for q in pc if q[2]), None)
            plan_clear_after_reach = [q for q in pc if reach_from is not None and q[0] >= reach_from and q[1] > 0.0]
            out.append(dict(
                b=int(b), t=int(t), j=int(j), enc_no=int(len(prev)), n_prev=len(prev),
                prev=[dict(clear=q["clear"], any=q["any"], hard=q["hard"], side=q["side"], v=q["v"]) for q in prev],
                rehit_in_enc=int(n_h > 0), enc_side=e["side"], enc_clear=e["clear"],
                v_hit=float(abs(st[t, b, 3])), v_1s=float(abs(st[max(0, t - 10), b, 3])),
                t_vis=int(t_vis), vis_s=float((t - t_vis) * R.dt), vis_any_s=float((t - (tlo + any_first)) * R.dt) if any_first is not None else 0.0,
                vis_window_s=float((t - tlo) * R.dt), seen=first is not None,
                d_vis=float(dsc[first]) if first is not None else float(dsc[-1]), D=D, v_vis=v0,
                nvis=nvis.tolist(), yreq_l=float(yreq_l), yreq_r=float(yreq_r), room_l=float(room_l), room_r=float(room_r),
                feas={k_: list(v_) for k_, v_ in feas.items()},
                plan=pc, plan_through_from=plan_through_from, reach_from=reach_from,
                n_plan_clear_after_reach=len(plan_clear_after_reach),
                plan_clear_last=pc[-2][1] if len(pc) >= 2 else pc[-1][1],
                box_c=P.mean(0).tolist(), wall_free=float(min(room_l, room_r)), wall_free_max=float(max(room_l, room_r))))
    return out


def main():
    outp = sys.argv[1]
    res = {}
    for arg in sys.argv[2:]:
        fn, lab = (arg.split(":", 1) + [os.path.basename(arg)])[:2]
        R = Rec(fn); C = Caster(R)
        rm_k = int(round(float(R.z["rm_t"]) / R.dt))
        mx = float(R.z["lidar_mount_x"])
        E, H = [], []
        for b in range(min(R.B, MAXB)):
            if not (R.sid[0, b] >= 0).any():
                continue
            enc = encounters(R, b, rm_k)
            attach_contacts(R, b, enc, rm_k)
            H += hard_events(R, C, b, enc, rm_k, mx)
            for e in enc:
                e.pop("hard_t"); e.pop("any_t")
            E += enc
        res[lab] = dict(file=fn, enc=E, hard=H, B=int(min(R.B, MAXB)), rm_k=rm_k, layout=(np.arange(R.B) % 64).tolist())
        print(f"{lab}: encounters {len(E)}, hard box events {len(H)}, any-contact encounters {sum(e['any'] > 0 for e in E)}", flush=True)
    json.dump(res, open(outp, "w"))


if __name__ == "__main__":
    main()
