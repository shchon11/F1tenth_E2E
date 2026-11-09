"""실측 npz (bag_extract.py) -> 같은 시간 격자(100 Hz)의 신호들. 다른 분석 도구들이 공통으로 씀.
여기서 정하는 것(모두 bag에서 직접 확인, 11-09):
  * 조향 명령 [rad], 속도 명령 [m/s] = VESC가 실제 받은 /commands/servo/position, /commands/motor/speed를 차의 변환식으로
    되돌린 값 (servo = 0.62875 x 조향 + 0.39527, ERPM = 4204 x 속도: 대회장 bag에서 /drive와 시간 맞춰 맞춤, 잔차 중앙 0.0002).
    /drive와 다른 때(조이스틱 전환)가 주행의 1-2 %
  * 바퀴 속도 [m/s] = /sensors/core ERPM / ERPM_PER_MPS
  * 자이로 = /sensors/imu/raw angular_velocity [rad/s] (정지 구간 중앙값을 바이어스로 뺌), 가속도 = linear_acceleration x 9.81
  * 위치 = /pf/pose/odom (파티클 필터)"""
import numpy as np

HZ = 100.0
ERPM_PER_MPS = 4204.0
SERVO_GAIN, SERVO_OFFSET = 0.62875, 0.39527     # servo = gain x steering [rad] + offset (the competition car, fitted 11-09)


def load(path):
    return np.load(path, allow_pickle=True)


def grid(Z, hz=HZ):
    t_end = min(Z["core"][-1, 0], Z["imu_raw"][-1, 0], Z["drive"][-1, 0] if "drive" in Z else 1e9)
    t_beg = max(Z["core"][0, 0], Z["imu_raw"][0, 0])
    return np.arange(t_beg, t_end, 1.0 / hz)


def hold(t_src, x_src, t):
    """the latest sample at or before t (sample and hold: a command stays until the next one)"""
    i = np.clip(np.searchsorted(t_src, t, side="right") - 1, 0, len(t_src) - 1)
    return x_src[i]


def signals(Z, hz=HZ, erpm_per_mps=ERPM_PER_MPS):
    t = grid(Z, hz)
    S = {"t": t}
    if "drive" in Z:
        S["drive_steer"] = hold(Z["drive"][:, 0], Z["drive"][:, 3], t)
        S["drive_speed"] = hold(Z["drive"][:, 0], Z["drive"][:, 2], t)
    # what the VESC was actually sent (after the joystick mux): 1-2 % of the driving differs from /drive, at the
    # hand-overs -- the commands the car carried out, back in rad and m/s through its own mapping
    if "erpm_cmd" in Z:
        S["erpm_cmd"] = hold(Z["erpm_cmd"][:, 0], Z["erpm_cmd"][:, 1], t)
        S["speed_cmd"] = S["erpm_cmd"] / erpm_per_mps
    if "servo_cmd" in Z:
        S["servo_cmd"] = hold(Z["servo_cmd"][:, 0], Z["servo_cmd"][:, 1], t)
        S["steer_cmd"] = (S["servo_cmd"] - SERVO_OFFSET) / SERVO_GAIN
    for k_cmd, k_drv in (("steer_cmd", "drive_steer"), ("speed_cmd", "drive_speed")):
        if k_cmd not in S and k_drv in S:
            S[k_cmd] = S[k_drv]
    c = Z["core"]
    S["v_wheel"] = np.interp(t, c[:, 0], c[:, 2]) / erpm_per_mps
    S["i_motor"] = np.interp(t, c[:, 0], c[:, 3])
    S["duty"] = np.interp(t, c[:, 0], c[:, 5])
    S["v_batt"] = np.interp(t, c[:, 0], c[:, 6])
    im = Z["imu_raw"]
    for j, k in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
        S[k] = np.interp(t, im[:, 0], im[:, 2 + j]) * (9.81 if k[0] == "a" else 1.0)
    # gyro / accel bias from standstill (wheel still for >= 1 s, away from its edges)
    still = np.abs(S["v_wheel"]) < 0.02
    k = int(hz)
    run = np.convolve(still.astype(float), np.ones(k), "same") >= k - 0.5
    S["still"] = run
    S["bias"] = {}
    if run.sum() > hz:
        for kk in ("gx", "gy", "gz", "ax", "ay"):
            b = float(np.median(S[kk][run])); S["bias"][kk] = b; S[kk] = S[kk] - b
    if "pf" in Z:
        p = Z["pf"]
        S["px"] = np.interp(t, p[:, 0], p[:, 2]); S["py"] = np.interp(t, p[:, 0], p[:, 3])
        S["pyaw"] = np.interp(t, p[:, 0], np.unwrap(p[:, 4]))
    if "odom" in Z:
        o = Z["odom"]
        S["odom_v"] = np.interp(t, o[:, 0], o[:, 5]); S["odom_wz"] = np.interp(t, o[:, 0], o[:, 7])
    if "imu_filt" in Z:
        S["imu_filt"] = np.interp(t, Z["imu_filt"][:, 0], Z["imu_filt"][:, 1])
    return S


def smooth(x, n):
    """centred moving average over n samples"""
    if n <= 1:
        return x
    k = np.ones(n) / n
    y = np.convolve(x, k, "same")
    return y


def deriv(x, hz=HZ, n=5):
    """centred difference over +-n samples"""
    d = np.zeros_like(x)
    d[n:-n] = (x[2 * n:] - x[:-2 * n]) * hz / (2 * n)
    return d
