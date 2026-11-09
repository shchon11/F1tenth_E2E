"""실측 명령을 시뮬 물리에 그대로 넣어 1초 구간마다 실차와 비교 (11-09, 사용자: "실측백 있잖아. 기존 측정했던거 믿지 말고 네가 새로
직접 다시 확인해").

구간: 대회장 bag의 주행 중(시작 속도 > 1 m/s) 1.0 s 구간, 0.25 s 간격. 구간 시작에서 차의 상태를 실측으로 맞춤 -- 속도 = 바퀴 속도(ERPM),
회전율 = 자이로, 옆 속도 0 -- 서보 각은 구간 0.5 s 전부터 같은 명령으로 서보만 돌려 만듦. 그 뒤 1 s 동안 bag의 /drive 명령(조향 [rad],
속도 [m/s])을 시뮬과 같은 방식으로 넣고(`cmd_delay` 뒤 적용, `servo_target`, `vesc_accel`, `dynamics.step_dynamics`, 1 kHz) 10 ms마다
회전율과 바퀴 속도를 실측과 비교.

`replay(W, over)`: over = 기준 차 파라미터 위에 덮어쓸 값들 {이름: 값 또는 (N,) 텐서}. 돌려주는 것: 회전율·속도 예측 (N, 100).
`metrics(W, pr)`: 회전율 오차 RMS(구간별 중앙), 1 s 뒤 방향 오차, 회전 크기 비(예측 / 실측, |r| > 0.5인 0.3-1.0 s 표본의 기울기),
속도 오차 RMS, 1 s 뒤 속도 오차."""
import os, sys, glob
import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2r_signals as G
from f1sim import dynamics as dyn
from f1sim import map_controller as M
from f1sim.actuators import servo_target, vesc_accel

HZ = 100
WIN = 1.0
WARM = 0.5


def build_windows(files, stride=0.25, v_min=1.0, device="cpu"):
    rows = {k: [] for k in ("cd", "cv", "r", "v", "ay", "bag", "t0")}
    for f in files:
        S = G.signals(G.load(f))
        n = len(S["t"])
        nw, nc = int(WIN * HZ), int((WIN + WARM) * HZ)
        step = int(stride * HZ)
        for i0 in range(int(WARM * HZ) + 1, n - nw - 1, step):
            v0 = S["v_wheel"][i0]
            if v0 < v_min or S["v_wheel"][i0:i0 + nw].min() < 0.3:
                continue
            j = i0 - int(WARM * HZ)
            rows["cd"].append(S["steer_cmd"][j:j + nc]); rows["cv"].append(S["speed_cmd"][j:j + nc])
            rows["r"].append(S["gz"][i0:i0 + nw + 1]); rows["v"].append(S["v_wheel"][i0:i0 + nw + 1])
            rows["ay"].append(S["ay"][i0:i0 + nw + 1])
            rows["bag"].append(os.path.basename(f)[:15]); rows["t0"].append(S["t"][i0])
    W = {k: torch.tensor(np.array(rows[k]), dtype=torch.float32, device=device) for k in ("cd", "cv", "r", "v", "ay")}
    W["bag"], W["t0"] = rows["bag"], np.array(rows["t0"])
    return W


def nominal(device="cpu"):
    P, wheel, dt = M.config_nominal()
    return {k: float(v) for k, v in P.items()}, wheel, dt


@torch.no_grad()
def replay(W, over=None, device=None):
    """the sim's physics driven by the recorded commands; (yaw rate, wheel speed) every 10 ms, (N, 101)"""
    dev = W["r"].device if device is None else device
    P0, wheel, dt = nominal()
    N = W["r"].shape[0]
    P = {}
    for k, v in P0.items():
        P[k] = torch.full((N,), float(v), device=dev)
    for k, v in (over or {}).items():
        P[k] = v.to(dev).float() if torch.is_tensor(v) else torch.full((N,), float(v), device=dev)
    sub = int(round(1.0 / HZ / dt))                      # physics steps per 10 ms
    nwarm = int(WARM * HZ)
    delay_steps = torch.round(P["cmd_delay"] / dt).long()     # (N,)
    cd, cv = W["cd"], W["cv"]
    ar = torch.arange(N, device=dev)

    def cmd_at(k_phys):
        """command in effect at physics step k_phys (counted from the warm-up start): the 10 ms sample sent cmd_delay before"""
        kk = ((k_phys - delay_steps).clamp(min=0) // sub).clamp(max=cd.shape[1] - 1)
        return cd[ar, kk], cv[ar, kk]

    # servo warm-up: the servo alone through the warm-up commands
    delta = torch.zeros(N, device=dev)
    for k in range(nwarm * sub):
        c_d, _ = cmd_at(torch.full((N,), k, device=dev))
        tgt = servo_target(c_d, P)
        rate = ((tgt - delta) / P["servo_tau"]).clamp(-P["sv_max"], P["sv_max"])
        delta = (delta + rate * dt).clamp(-P["s_max"], P["s_max"])
    st = torch.zeros(N, dyn.STATE_DIM, device=dev)
    st[:, dyn.IVX] = W["v"][:, 0]
    st[:, dyn.IR] = W["r"][:, 0]
    st[:, dyn.ISTEER] = delta
    st[:, dyn.IOMEGA] = W["v"][:, 0] / P["r_w"]
    ax = torch.zeros(N, device=dev)
    R, V = [st[:, dyn.IR].clone()], [st[:, dyn.IOMEGA] * P["r_w"]]
    k0 = nwarm * sub
    for i in range(int(WIN * HZ)):
        for j in range(sub):
            c_d, c_v = cmd_at(torch.full((N,), k0 + i * sub + j, device=dev))
            tgt = servo_target(c_d, P)
            v_fb = st[:, dyn.IOMEGA] * P["r_w"] if wheel else st[:, dyn.IVX]
            a_cmd = vesc_accel(c_v, v_fb, P)
            st, ax, _ = dyn.step_dynamics(st, tgt, a_cmd, ax, P, P["servo_tau"], dt, wheel=wheel)
        R.append(st[:, dyn.IR].clone()); V.append(st[:, dyn.IOMEGA] * P["r_w"])
    return torch.stack(R, 1), torch.stack(V, 1)


def metrics(W, pr):
    r_p, v_p = pr
    r_m, v_m = W["r"], W["v"]
    e_r = (r_p - r_m)[:, 1:]
    rms_r = e_r.pow(2).mean(1).sqrt()
    head = (e_r.sum(1) * (1.0 / HZ)).abs()                  # heading error after 1 s [rad]
    sl = slice(30, 101)
    rm, rp = r_m[:, sl].reshape(-1), r_p[:, sl].reshape(-1)
    big = rm.abs() > 0.5
    gain = float((rp[big] * rm[big]).sum() / (rm[big] * rm[big]).sum())
    e_v = (v_p - v_m)[:, 1:]
    return {"yaw_rms": float(rms_r.median()), "head_1s": float(head.median()), "yaw_gain": gain,
            "v_rms": float(e_v.pow(2).mean(1).sqrt().median()), "v_1s": float((v_p - v_m)[:, -1].median()),
            "v_1s_abs": float((v_p - v_m)[:, -1].abs().median())}


def comp_files():
    return [f for f in sorted(glob.glob("/data/sim2real_1109/*map08x23*.npz")) if "STATIONARY" not in f]


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = build_windows(comp_files(), device=dev)
    print(f"구간 {W['r'].shape[0]}개 (대회장 bag {len(set(W['bag']))}개)", flush=True)
    m = metrics(W, replay(W))
    print("지금 시뮬 기준 차:", {k: round(v, 4) for k, v in m.items()}, flush=True)
