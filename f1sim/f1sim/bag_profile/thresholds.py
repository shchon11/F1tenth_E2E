"""Every threshold the bag_profile tool judges by, in one place.

`docs/bag_recording_guide.md` quotes these numbers as "(도구 문턱값)". The summary printed by the tool ends with
`table_md()`, so the numbers a run used are listed in its own output; change one here and the next summary shows it.
Nothing else in the package may hold a literal that decides whether a value counts as measured.

Layout: `_t(name, value, unit, meaning)` returns the value and registers the row; the rows keep definition order.
"""
from __future__ import annotations

from typing import List, Tuple

TABLE: List[Tuple[str, object, str, str]] = []


def _t(name: str, value, unit: str, meaning: str):
    TABLE.append((name, value, unit, meaning))
    return value


# ---- test segment marks ----
SEGMENT_DEDUP_S = _t("SEGMENT_DEDUP_S", 2.0, "s", "같은 구간 표시가 이 시간 안에 되풀이되면 한 번으로 셈 (ros2 topic pub 을 3번 내는 관례)")

# ---- servo map, clamp window ----
SERVO_PAIR_TOL_S = _t("SERVO_PAIR_TOL_S", 0.01, "s", "Ackermann 명령과 서보 명령을 한 쌍으로 보는 시간차")
SERVO_FIT_MIN_PAIRS = _t("SERVO_FIT_MIN_PAIRS", 20, "쌍", "서보 변환식을 맞추는 데 필요한 최소 쌍 수")
SERVO_FIT_MIN_STEER_STD = _t("SERVO_FIT_MIN_STEER_STD", 0.02, "rad", "서보 변환식을 맞추는 데 필요한 조향 표준편차")
SERVO_FIT_INLIER_TOL = _t("SERVO_FIT_INLIER_TOL", 0.01, "서보 값", "변환식에서 이만큼 안이면 정상값으로 셈")
CLAMP_PAIR_TOL_S = _t("CLAMP_PAIR_TOL_S", 0.02, "s", "서보 명령과 적용 위치를 짝짓는 시간차")
CLAMP_EDGE_MIN_CMDS = _t("CLAMP_EDGE_MIN_CMDS", 3, "개", "창 끝이 '관측됨'이려면 그 밖으로 나간 명령 수")
CLAMP_EDGE_MARGIN = _t("CLAMP_EDGE_MARGIN", 1e-3, "서보 값", "창 밖으로 치는 최소 거리")

# ---- ERPM, standstill, units ----
ERPM_MIN_SPEED_CMD = _t("ERPM_MIN_SPEED_CMD", 0.5, "m/s", "ERPM 환산에 쓰는 속도 지령 하한")
ERPM_MIN_PAIRS = _t("ERPM_MIN_PAIRS", 10, "쌍", "ERPM 환산에 필요한 지령-회전수 쌍 수")
ERPM_ODOM_MIN_V = _t("ERPM_ODOM_MIN_V", 1.0, "m/s", "/odom 으로 환산을 대조할 때의 속도 하한")
ERPM_ODOM_MIN_SAMPLES = _t("ERPM_ODOM_MIN_SAMPLES", 50, "개", "/odom 대조에 필요한 샘플 수")
STILL_V = _t("STILL_V", 0.02, "m/s", "정지로 보는 바퀴 속도 상한")
STILL_SECONDS = _t("STILL_SECONDS", 1.0, "s", "정지 구간의 최소 길이 (바이어스, 가속도 단위)")
UNIT_MIN_V = _t("UNIT_MIN_V", 1.0, "m/s", "자이로 단위·축 부호 판정에 쓰는 속도 하한")
UNIT_MIN_SAMPLES = _t("UNIT_MIN_SAMPLES", 200, "개", "자이로 단위 판정에 필요한 샘플 수 (2 s)")
GYRO_DEG_P99 = _t("GYRO_DEG_P99", 15.0, "raw", "기준이 없을 때 |gz| p99 가 이보다 크면 deg/s 로 추정")

# ---- direct measurements ----
DELAY_MIN_V = _t("DELAY_MIN_V", 1.5, "m/s", "겉보기 지연에 쓰는 속도 하한")
DELAY_MIN_SEG_S = _t("DELAY_MIN_SEG_S", 3.0, "s", "겉보기 지연에 쓰는 움직임 구간의 최소 길이")
DELAY_MAX_LAG_S = _t("DELAY_MAX_LAG_S", 0.4, "s", "겉보기 지연 교차상관 탐색 한계")
TRIM_MAX_STEER = _t("TRIM_MAX_STEER", 0.08, "rad", "트림 판정에 쓰는 '거의 직진' 조향 한계")
TRIM_MIN_V = _t("TRIM_MIN_V", 1.5, "m/s", "트림 판정 속도 하한")
TRIM_MAX_DR = _t("TRIM_MAX_DR", 1.0, "rad/s^2", "트림 판정에서 요레이트 변화율 상한")
TRIM_MIN_SAMPLES = _t("TRIM_MIN_SAMPLES", 200, "개", "트림에 필요한 샘플 수 (2 s)")
CUR_LIMIT_FRAC = _t("CUR_LIMIT_FRAC", 0.9, "비율", "전류가 한계의 이 비율 이상(회생은 이하)이면 한계 구간으로 셈")
ACCEL_MIN_RUN_S = _t("ACCEL_MIN_RUN_S", 0.2, "s", "한계 전류 구간의 최소 길이")
ACCEL_TRIM_S = _t("ACCEL_TRIM_S", 0.05, "s", "한계 구간 양 끝에서 버리는 길이 (필터 영향)")
ACCEL_BIN_MIN_SAMPLES = _t("ACCEL_BIN_MIN_SAMPLES", 10, "개", "속도 구간 한 칸이 표에 들어가는 최소 샘플 수")
ACCEL_MIN_BINS = _t("ACCEL_MIN_BINS", 3, "칸", "a_max 가 '측정됨'이려면 필요한 속도 구간 수 (0.3~3, 3~5, 5~7, 7~9, 9~20 m/s 중)")
CUR_PLATEAU_BAND = _t("CUR_PLATEAU_BAND", 0.02, "비율", "전류 평탄 구간: 극값의 이 비율 안")
CUR_PLATEAU_MIN_S = _t("CUR_PLATEAU_MIN_S", 0.2, "s", "전류 평탄 구간의 최소 연속 길이")
CUR_PLATEAU_MIN_RUNS = _t("CUR_PLATEAU_MIN_RUNS", 2, "개", "전류 한계가 '측정됨'이려면 필요한 서로 다른 평탄 구간 수")
CUR_PLATEAU_SEP_S = _t("CUR_PLATEAU_SEP_S", 0.5, "s", "서로 다른 평탄 구간으로 세려면 떨어져 있어야 하는 시간")
LAT_HOLD_S = _t("LAT_HOLD_S", 0.3, "s", "그립 하한: 이만큼 유지된 횡가속만 셈")
LAT_NOSLIP_DEG = _t("LAT_NOSLIP_DEG", 5.0, "deg", "그립 하한: 유지 구간 안 차체 미끄럼 변화 상한")
SLIDE_DEG = _t("SLIDE_DEG", 30.0, "deg", "미끄러진 사건: 1 s 안 차체 미끄럼 변화")
SLIDE_WINDOW_S = _t("SLIDE_WINDOW_S", 1.0, "s", "미끄러진 사건을 보는 창")
SLIDE_MIN_V = _t("SLIDE_MIN_V", 2.0, "m/s", "미끄러진 사건의 속도 하한")
IMPACT_ACC = _t("IMPACT_ACC", 15.0, "m/s^2", "0.05 s 평균 |a| 가 이보다 크면 충격 (재생 구간과 미끄럼 사건에서 뺌)")
T2_V_MIN = _t("T2_V_MIN", 0.7, "m/s", "저속 정상 선회 속도 하한")
T2_V_MAX = _t("T2_V_MAX", 1.8, "m/s", "저속 정상 선회 속도 상한")
T2_MIN_STEER = _t("T2_MIN_STEER", 0.08, "rad", "저속 정상 선회 최소 조향")
T2_MAX_DR = _t("T2_MAX_DR", 0.5, "rad/s^2", "저속 정상 선회 요레이트 변화율 상한")
T2_MIN_SECONDS = _t("T2_MIN_SECONDS", 10.0, "s", "저속 정상 선회 샘플 합계 최소 (연속일 필요 없음)")
LIDAR_STILL_V = _t("LIDAR_STILL_V", 0.01, "m/s", "LiDAR 정지 잡음: 바퀴 속도 상한")
LIDAR_STILL_GYRO = _t("LIDAR_STILL_GYRO", 0.03, "rad/s", "LiDAR 정지 잡음: 자이로 크기 상한")
LIDAR_STILL_BLOCK_S = _t("LIDAR_STILL_BLOCK_S", 1.0, "s", "LiDAR 정지 잡음 블록 길이")
LIDAR_GAP_BEAMS = _t("LIDAR_GAP_BEAMS", 5, "빔", "누락으로 보는 끊김의 최대 빔 수")

# ---- replay windows ----
WIN_STRIDE_S = _t("WIN_STRIDE_S", 0.25, "s", "재생 구간 시작 간격")
WIN_MIN_V_START = _t("WIN_MIN_V_START", 1.0, "m/s", "재생 구간 시작 속도 하한")
WIN_MIN_V_ALL = _t("WIN_MIN_V_ALL", 0.3, "m/s", "재생 구간 내내 속도 하한")
WIN_MAX_CMD_GAP_S = _t("WIN_MAX_CMD_GAP_S", 0.1, "s", "재생 구간 안 명령 끊김 상한")
REPLAY_MIN_WINDOWS = _t("REPLAY_MIN_WINDOWS", 30, "개", "응답·타이어 맞춤에 필요한 재생 구간 수")

# ---- coast-down (T4) ----
COAST_MAX_CURRENT = _t("COAST_MAX_CURRENT", 0.5, "A", "코스트다운: |모터 전류|가 이 이하면 '전류 0'")
COAST_MIN_V = _t("COAST_MIN_V", 0.5, "m/s", "코스트다운에 쓰는 속도 하한")
COAST_MIN_RUN_S = _t("COAST_MIN_RUN_S", 1.0, "s", "코스트다운 한 구간의 최소 길이")
COAST_MIN_TOTAL_S = _t("COAST_MIN_TOTAL_S", 3.0, "s", "코스트다운 구간 합계 최소")
COAST_MIN_SEGMENTS = _t("COAST_MIN_SEGMENTS", 2, "개", "코스트다운 구간 최소 개수 (오차 추정)")
COAST_MIN_V_SPAN = _t("COAST_MIN_V_SPAN", 2.0, "m/s", "코스트다운 자료의 속도 폭 최소 (c_roll 과 c_drag 를 가르려면)")

# ---- steering step / chirp (T5) ----
CHIRP_MIN_V = _t("CHIRP_MIN_V", 1.5, "m/s", "처프 분석 속도 하한")
CHIRP_WIN_S = _t("CHIRP_WIN_S", 5.0, "s", "처프 구간을 찾는 창 길이")
CHIRP_HP_RMS = _t("CHIRP_HP_RMS", 0.02, "rad", "자동 인식: 1 Hz 이상 조향 성분의 rms 하한")
CHIRP_MIN_TOTAL_S = _t("CHIRP_MIN_TOTAL_S", 10.0, "s", "처프 분석에 필요한 구간 합계")
CHIRP_BAND_LO_HZ = _t("CHIRP_BAND_LO_HZ", 0.5, "Hz", "주파수 응답 맞춤 대역 하한")
CHIRP_BAND_HI_HZ = _t("CHIRP_BAND_HI_HZ", 4.0, "Hz", "주파수 응답 맞춤 대역 상한")
CHIRP_MIN_COHERENCE = _t("CHIRP_MIN_COHERENCE", 0.6, "-", "대역 안에서 이 일관성 이상인 주파수만 맞춤에 씀")
CHIRP_MIN_BINS = _t("CHIRP_MIN_BINS", 6, "개", "맞춤에 쓸 주파수 점의 최소 개수")
CHIRP_BW_FRAC = _t("CHIRP_BW_FRAC", 0.1, "비율", "입력 조향 성분의 대역폭: 대역 최대 파워의 이 비율 이상인 가장 높은 주파수")
CHIRP_SEP_MAX_CI_S = _t("CHIRP_SEP_MAX_CI_S", 0.03, "s", "순수 지연의 90 % 구간 폭이 이보다 좁아야 '분리됨'")
CHIRP_MIN_BANDWIDTH_HZ = _t("CHIRP_MIN_BANDWIDTH_HZ", 2.0, "Hz", "입력 조향 성분이 이 주파수까지는 있어야 분리 가능")

# ---- setup.yaml comparison and car-profile fingerprint ----
SETUP_TOL_ERPM = _t("SETUP_TOL_ERPM", 20.0, "ERPM/(m/s)", "setup.yaml 대조: ERPM 환산 허용 차")
SETUP_TOL_SERVO_GAIN = _t("SETUP_TOL_SERVO_GAIN", 0.02, "서보/rad", "setup.yaml 대조: 서보 이득 허용 차")
SETUP_TOL_SERVO_OFFSET = _t("SETUP_TOL_SERVO_OFFSET", 0.01, "서보", "setup.yaml 대조: 서보 오프셋 허용 차")
SETUP_TOL_SERVO_RANGE = _t("SETUP_TOL_SERVO_RANGE", 0.01, "서보", "setup.yaml 대조: 서보 창 끝 허용 차")
SETUP_TOL_CURRENT_A = _t("SETUP_TOL_CURRENT_A", 3.0, "A", "setup.yaml 대조: 전류 한계 허용 차")
SETUP_TOL_TF_M = _t("SETUP_TOL_TF_M", 0.01, "m", "setup.yaml 대조: LiDAR tf 허용 차")
FP_TOL_REGEN_A = _t("FP_TOL_REGEN_A", 8.0, "A", "--car 대조: 회생 전류 허용 차")
FP_TOL_DELAY_S = _t("FP_TOL_DELAY_S", 0.03, "s", "--car 대조: 겉보기 지연 허용 차")


def table_md() -> str:
    """The thresholds as a markdown table (appended to every summary)."""
    rows = ["| 이름 | 값 | 단위 | 뜻 |", "|---|---|---|---|"]
    for n, v, u, m in TABLE:
        rows.append(f"| {n} | {v} | {u} | {m} |")
    return "\n".join(rows)
