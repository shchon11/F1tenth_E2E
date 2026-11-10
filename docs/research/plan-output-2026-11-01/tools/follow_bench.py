"""제어기만 따로, 정해진 기준 궤적으로 (11-10, 사용자 기준: 제어기는 주어진 궤적에 대한 추종 오차로 고름). policy 없이 고정된 궤적 하나
-- 직선 1 s, 왼쪽 원호(반지름 R, 일정 속도 v: 횡가속도 v^2/R), 직선 -- 을 운동학 자전거(무게중심, plan_steps와 같은 모델)로 만들고,
0.1 s마다 그 궤적의 앞 1 s를 plan으로 다시 줌(policy가 하는 것처럼; 기준은 고정). 차는 시뮬과 같은 물리(dynamics.step_dynamics 1 ms,
명령 지연 0.03 s, 서보, VESC 속도 고리, 바퀴 모델, 공칭 차). 제어기마다: 경로까지 옆 오차 RMS·최대, 실제 조향 0.1 s마다 바뀌는 크기와
방향 바로 뒤집힘, 요 각속도 떨림(1 s 평균을 뺀 나머지), 조향 끝에 붙은 비율, 걸음당 계산 시간.
인자: [v, 기본 5.0] [R, 기본 3.5] [제어기들, 기본 table,mpc] [plan 떨림 deg, 기본 0: 결정마다 plan 전체의 운동학 조향에 +-값을
번갈아 더해 다시 적분 -- policy의 plan이 결정마다 1-2 deg씩 오락가락하는 것처럼; 오차는 떨림 없는 기준 경로에 대해]"""
import sys, time, math
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import map_controller as M, dynamics as dyn
from f1sim.actuators import servo_target, vesc_accel
from f1sim.plan_follow import PlanFollower
from f1sim.plan_mpc import PlanMPC

torch.set_grad_enabled(False)
v0 = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
R = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
kinds = (sys.argv[3] if len(sys.argv) > 3 else "table,mpc").split(",")
jit = math.radians(float(sys.argv[4])) if len(sys.argv) > 4 else 0.0
lut, nom = M.config_lut()
P = {k: torch.tensor([float(v)]) for k, v in nom.items()}
L, lr = float(nom["lf"]) + float(nom["lr"]), float(nom["lr"])
h, dt = 0.025, 0.001
# the reference: kinematic bicycle about the CoG, steering ramped in / out over 0.2 s
T_total = 6.0
nT = int(T_total / h) + 80
t = np.arange(nT) * h
d_c = math.atan(L / R)                                            # the circle's kinematic steering (about the rear axle ~)
dk = np.clip((t - 1.0) / 0.2, 0, 1) * d_c - np.clip((t - 1.0 - 2.0 * math.pi * R / v0 * 0.5) / 0.2, 0, 1) * d_c
X = np.zeros(nT); Y = np.zeros(nT); PSI = np.zeros(nT)
for i in range(1, nT):
    beta = math.atan(lr / L * math.tan(dk[i - 1]))
    r = v0 * math.cos(beta) * math.tan(dk[i - 1]) / L
    X[i] = X[i - 1] + v0 * math.cos(PSI[i - 1] + beta) * h
    Y[i] = Y[i - 1] + v0 * math.sin(PSI[i - 1] + beta) * h
    PSI[i] = PSI[i - 1] + r * h
ref = {k: torch.tensor(a, dtype=torch.float32) for k, a in (("x", X), ("y", Y), ("psi", PSI), ("d", dk))}


def path_err(px, py):
    d2 = (X - px) ** 2 + (Y - py) ** 2
    i = int(np.argmin(d2))
    c = PSI[i] + math.atan(lr / L * math.tan(dk[i]))
    return -math.sin(c) * (px - X[i]) + math.cos(c) * (py - Y[i])


for kind in kinds:
    if kind == "table":
        fol = PlanFollower(1, "cpu", lut=lut, grip=M.GRIP_LEVELS.index(1.0), wheelbase=L, s_max=float(nom["s_max"]))
    elif kind == "nolut":
        fol = PlanFollower(1, "cpu", lut=None, wheelbase=L, s_max=float(nom["s_max"]))
    else:
        # "mpc" or "mpc:key=value,...": PlanMPC's arguments (w_lat, w_head, w_rate, w_yaw, horizon, sub)
        kw = {a.split("=")[0]: (int if a.split("=")[0] in ("horizon", "sub") else float)(a.split("=")[1])
              for a in kind.split(":")[1].split(";")} if ":" in kind else {}
        fol = PlanMPC(1, "cpu", nom, s_max=float(nom["s_max"]), **kw)
    st = torch.zeros(1, dyn.STATE_DIM); st[0, dyn.IVX] = v0; st[0, dyn.IOMEGA] = v0 / float(nom["r_w"])
    hist = torch.zeros(1, 8, 2); hist[:, :, 1] = v0                  # command history, newest first
    ax = torch.zeros(1)
    E, SD, YR, SAT, CT = [], [], [], [], []
    n_steps = int(5.0 / h)
    for k in range(n_steps):
        if k % 4 == 0:                                            # a new plan: the next 1 s of the reference
            sl = slice(k, k + 41)
            if jit > 0:
                # the reference's steering + an offset that flips sign every decision, integrated from the reference pose
                sg = jit * (1.0 if (k // 4) % 2 == 0 else -1.0)
                dj = dk[sl] + sg
                xs, ys, ps = [X[k]], [Y[k]], [PSI[k]]
                for i in range(40):
                    beta = math.atan(lr / L * math.tan(dj[i]))
                    rr = v0 * math.cos(beta) * math.tan(dj[i]) / L
                    xs.append(xs[-1] + v0 * math.cos(ps[-1] + beta) * h); ys.append(ys[-1] + v0 * math.sin(ps[-1] + beta) * h)
                    ps.append(ps[-1] + rr * h)
                tt = lambda a: torch.tensor(np.array(a), dtype=torch.float32)[None]
                fol.set_plan(slice(None), tt(xs), tt(ys), tt(ps), torch.full((1, 41), v0), tt(dj))
            else:
                fol.set_plan(slice(None), ref["x"][None, sl], ref["y"][None, sl], ref["psi"][None, sl],
                             torch.full((1, 41), v0), ref["d"][None, sl])
        t0 = time.perf_counter()
        cmd = fol.command(st)
        CT.append(time.perf_counter() - t0)
        hist = torch.cat([cmd[:, None, :], hist[:, :-1]], 1)
        for j in range(25):
            age = float(nom["cmd_delay"]) - (j + 1) * dt
            idx = max(0, min(hist.shape[1] - 1, math.ceil(age / h - 1e-9)))
            eff = hist[:, idx]
            v_fb = st[:, dyn.IOMEGA] * P["r_w"]
            a_cmd = vesc_accel(eff[:, 1], v_fb, P)
            st, ax, _ = dyn.step_dynamics(st, servo_target(eff[:, 0], P), a_cmd, ax, P, P["servo_tau"], dt, wheel=True)
        E.append(path_err(float(st[0, 0]), float(st[0, 1])))
        SD.append(math.degrees(float(st[0, dyn.ISTEER]))); YR.append(math.degrees(float(st[0, dyn.IR])))
        SAT.append(abs(float(cmd[0, 0])) > float(nom["s_max"]) - 0.01)
    E, SD, YR = np.array(E), np.array(SD), np.array(YR)
    dd = np.diff(SD[::4])
    zz = np.mean((dd[1:] * dd[:-1] < 0) & (np.abs(dd[1:]) > 0.5) & (np.abs(dd[:-1]) > 0.5))
    yrs = np.convolve(YR, np.ones(40) / 40, "same")
    print(f"{kind:>14} v {v0} R {R} (횡가속도 {v0 * v0 / R:.1f}, plan 떨림 {math.degrees(jit):.1f} deg): 옆 오차 RMS {np.sqrt(np.mean(E ** 2)) * 100:.1f} cm, 최대 "
          f"{np.abs(E).max() * 100:.1f} cm | 조향 0.1 s마다 {np.mean(np.abs(dd)):.2f} deg, 뒤집힘 {zz * 100:.0f} % | 요 각속도 떨림 "
          f"{np.std((YR - yrs)[20:-20]):.1f} deg/s | 조향 끝 {np.mean(SAT) * 100:.0f} % | 계산 {np.mean(CT) * 1000:.1f} ms/걸음", flush=True)
