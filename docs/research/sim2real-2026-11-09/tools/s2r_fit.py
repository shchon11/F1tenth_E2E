"""시뮬 파라미터를 실측 재생 오차로 맞춤 (11-09). s2r_replay의 1초 구간을 bag 단위로 둘로 나눠(맞춤용 / 확인용) 파라미터를 하나씩
격자로 바꾸며(나머지는 그때까지의 최선) 맞춤용 오차가 가장 작은 값을 고르고, 3바퀴 돎.
조향 쪽(steer_gain, steer_bias, servo_tau, cmd_delay, sv_max, B_f, B_r, mu_f_scale, Iz)은 회전율 오차로,
속도 쪽(motor_tau, a_max, v_switch, a_brake, speed_gain, motor_ramp)은 속도 오차로.
'확실히 정해지나'는 각 파라미터의 최선 근처 폭: 맞춤용 오차가 최선의 +3 % 안인 격자 값의 범위(넓으면 bag으로 못 정함).
인자: 출력 json"""
import sys, json, time
import numpy as np
import torch
import s2r_replay as RP

dev = "cuda" if torch.cuda.is_available() else "cpu"
files = RP.comp_files()
fit_f = files[0::2]; val_f = files[1::2]
Wf = RP.build_windows(fit_f, device=dev); Wv = RP.build_windows(val_f, device=dev)
print(f"맞춤용 구간 {Wf['r'].shape[0]} (bag {len(fit_f)}), 확인용 {Wv['r'].shape[0]} (bag {len(val_f)})", flush=True)


def tile(W, K):
    return {k: (v.repeat(K, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v) for k, v in W.items()}


def sweep(W, cur, name, values, key):
    K = len(values); N = W["r"].shape[0]
    WT = tile(W, K)
    over = {k: torch.full((N * K,), float(v), device=dev) for k, v in cur.items()}
    over[name] = torch.tensor(np.repeat(values, N), dtype=torch.float32, device=dev)
    r_p, v_p = RP.replay(WT, over)
    out = []
    for i in range(K):
        sl = slice(i * N, (i + 1) * N)
        m = RP.metrics({"r": W["r"], "v": W["v"]}, (r_p[sl], v_p[sl]))
        out.append(m[key])
    return np.array(out)


GRIDS_YAW = {
    "steer_gain": np.linspace(0.6, 1.3, 15), "steer_bias": np.linspace(-0.05, 0.05, 11),
    "servo_tau": np.array([0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.2]), "cmd_delay": np.array([0.0, 0.015, 0.03, 0.045, 0.06, 0.08, 0.1]),
    "sv_max": np.array([1.5, 2.0, 2.5, 3.2, 4.0, 6.0, 10.0]), "B_f": np.array([1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]),
    "B_r": np.array([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]), "mu_f_scale": np.array([0.7, 0.8, 0.92, 1.0, 1.1]),
    "Iz": np.array([0.025, 0.035, 0.047, 0.06, 0.08]),
}
GRIDS_V = {
    "speed_gain": np.linspace(0.94, 1.06, 7), "motor_tau": np.array([0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4]),
    "a_max": np.array([3.5, 4.5, 5.6, 6.5, 7.5, 9.0, 11.0]), "v_switch": np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]),
    "a_brake": np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 10.0]),
}
cur = {}
log = {"rounds": []}
t0 = time.time()
base_f = RP.metrics(Wf, RP.replay(Wf)); base_v = RP.metrics(Wv, RP.replay(Wv))
print("지금 시뮬: 맞춤용", {k: round(v, 4) for k, v in base_f.items()}, "| 확인용", {k: round(v, 4) for k, v in base_v.items()}, flush=True)
for rnd in range(3):
    for grids, key in ((GRIDS_YAW, "yaw_rms"), (GRIDS_V, "v_rms")):
        for name, vals in grids.items():
            L = sweep(Wf, cur, name, vals, key)
            i = int(np.argmin(L))
            ok = vals[L <= L[i] * 1.03]
            cur[name] = float(vals[i])
            log["rounds"].append({"round": rnd, "param": name, "values": vals.tolist(), "loss": L.tolist(), "best": cur[name],
                                  "flat_lo": float(ok.min()), "flat_hi": float(ok.max())})
            print(f"[{rnd}] {name:11s} 최선 {cur[name]:.4g} ({key} {L[i]:.4f}), +3 % 안 {ok.min():.4g}..{ok.max():.4g} | "
                  + " ".join(f"{v:.3g}:{l:.3f}" for v, l in zip(vals, L)) + f" | {time.time() - t0:.0f} s", flush=True)
mf = RP.metrics(Wf, RP.replay(Wf, cur)); mv = RP.metrics(Wv, RP.replay(Wv, cur))
print("맞춘 값:", cur, flush=True)
print("맞춘 뒤: 맞춤용", {k: round(v, 4) for k, v in mf.items()}, "| 확인용", {k: round(v, 4) for k, v in mv.items()}, flush=True)
log.update({"fit": cur, "before_fit": base_f, "before_val": base_v, "after_fit": mf, "after_val": mv, "fit_bags": fit_f, "val_bags": val_f})
json.dump(log, open(sys.argv[1], "w"), indent=1)
