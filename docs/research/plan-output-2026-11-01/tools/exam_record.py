"""시험 기록 (exam1, 11-12). 학습 표(plan_validate ideal -> plan_table)는 학습 세계 그대로라 "학습이 움직이나"만 말해 줌(11-11: 같은
트랙 변형·같은 상자 은행·plan대로 정확히·40 s·세션 단위 점수 없음·기준 행 없음). 이 도구는 조건을 한 번 정해 얼려 둔 시험:
  - 트랙: 실제 Korea 2025 지도 하나 (뒤튼 변형·크기 변형 없음)
  - 실행: 물리 전부 + 제어기 (map = MAP이 plan을 따라감, 실차 방식 / track = 콘솔 기본인 시간 추종기 / ideal = plan대로 정확히, 비교용)
  - 상자: 학습에 쓰인 적 없는 은행(prop_banks_exam1, 11-12 새로 뽑음), 개수는 인자 (max2 = 대회처럼 2개 / uniform = 1개~전부, 학습 분포)
  - 세션: 90 s 통째(약 10바퀴), 상자가 있는 세션은 50 s에 모두 치움 (대회 형식: 고정 배치 -> 치우면 가장 빠른 길로)
  - 네 경우(혼자·상대 차 x 상자 유무)는 학습 세계와 같은 반반 추첨, 상대 차는 학습 세계의 종류·속도 배율 0.4~1.2 그대로,
    2 s 동안 0.5 m도 못 간 상대 차는 다시 출발 (모든 checkpoint에 똑같이)
  - 고정 시드, policy 평균 행동, LiDAR 자리·타이어 한계 여유는 checkpoint가 학습한 값
기록 형식은 drive_record.py와 같아(장면 도구가 그대로 읽음) 거기에 두 차의 '트랙 따라 남은 거리'(보상의 벽 기준 거리장, 중심선 없음)와
한 바퀴 길이를 더함 -> 누가 더 멀리 갔나·앞지름을 exam_table.py가 셈.
인자: checkpoint 출력npz [제어기 map|track|ideal, 기본 map] [환경 수, 기본 1024] [상자 개수 max2|uniform, 기본 max2] [초 90] [치우는 시각 50]
F1SIM_EXAM_SEED: 두 번째 시드(같은 조건에서 숫자가 얼마나 흔들리나).
F1SIM_EXAM_SLIP_FRAME=oracle (11-12, 원인 검증용이지 시험 점수가 아님): plan의 모형은 미끄러지지 않는 차(차체 방향에서 조금 안쪽으로 나아감)인데
실제 차는 코너에서 차체 방향보다 바깥쪽으로 나아감 -> policy가 보는 스캔·가속도계와 plan의 출발 방향을 '미끄러지지 않는 차라면 지금 진행
방향으로 나아가고 있을 차체 방향'으로 돌려 줌 (돌리는 각 = 실제 진행 방향(시뮬레이터의 참 옆속도: 실차에는 없는 값) − 모형의 진행 방향).
MAP이 받는 구간별 곡률·속도는 그대로. 기록에 그 각(dlt, 결정마다)을 남김.
F1SIM_EXAM_SLIP_FRAME=table + F1SIM_EXAM_SLIP_TABLE=표npz: 같은 일을 참 옆속도 없이, 차가 스스로 아는 값(속도 x 회전 속도 = 옆가속도, 속도)으로
표(ay, v, beta[rad])에서 읽은 미끄럼 각으로 함 (속도와 회전 속도는 시뮬레이터의 참값: 센서 잡음은 안 넣음). 참 옆속도(vy)는 기록에만 남김.
F1SIM_EXAM_SPEED_LEAD=초 (제어기 쪽 진단, 시험 점수가 아님): MAP이 속도 명령으로 plan의 첫 0.1 s 구간 속도 대신 그 초만큼 뒤의 속도를 냄
(속도 루프가 명령을 시간 상수 0.2 s로 따라가서, 첫 구간 속도를 그대로 주면 plan이 낸 가속의 절반도 안 나옴).
F1SIM_EXAM_SAVE_MEM=1 (11-12, 사용자: "기억이 반복이 되던데 그 반복되는거에서 특정 구간을 뭔가 이용할 수는 없나"): 결정마다 policy가 읽은
특징(스캔 인코더 출력 + 정규화한 상태 입력; 기억이 읽는 것)과 그 결정 뒤의 기억 상태를 `출력_mem.npz`(feat, mem; float16)에 따로 남김.
F1SIM_EXAM_SAVE_SCAN=1 (11-13, policy가 상자를 보고도 왜 그대로 가는지 policy 안에서 보는 분석용): 결정마다 policy가 읽은 가장 새 스캔
(range_max로 나눈 값, float16; 입력은 이것과 앞 두 결정의 스캔)과 빔마다 맞은 것(lidar.HIT_*)을 `출력_scan.npz`(scan, typ, range_max, angles)에 남김.
주행·기록은 그대로.
F1SIM_EXAM_TRACK_MPC=1 (11-24): 제어기 "track"에서 시간 추종기 대신 모형 예측 추종기(plan_mpc.PlanMPC, 기본값)로 몲.
F1SIM_EXAM_SAVE_PLAN=1 (11-24, 제어기 튜닝 방 map_room.py용): 결정마다 차 상태 전부와 MAP이 읽는 plan(구간 곡률·속도), plan을 만든 행동·출발 속도·조향을
`출력_plan.npz`에 따로 남김. 주행은 그대로.
F1SIM_EXAM_SAVE_VALUE=1 (11-13, 상자에 닿기 전 critic은 언제부터 그 상자를 값에 치는지 보는 분석용): 결정마다 critic이 읽은
참값 입력(학습과 같은 것: 상자 칸은 트랙을 따라 --box-range 안)과 critic의 값을 `출력_priv.npz`(priv, value, box_cols)에 남김.
F1SIM_EXAM_OPP_SLOT=JSON (11-12, 시험 점수가 아니라 상대 차 하나를 따로 보는 기록): 상대 차 자리 설정에 덮어씀(state_rl.build_env의
`--opp-slot`과 같음). 예: AuRoRus 포팅본만 상대로 `{"kind": "aurorus", "kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0,
"speed_scale": [1.0, 1.0]}` (그 차는 자기 선 속도의 0.8 x speed_scale로 달리고, 자기 LiDAR로 본 점만으로 비켜 감). 기록의 opp_slot에 남김.
F1SIM_EXAM_START_REST=1 (11-12, 대회 규칙의 출발을 따로 보는 기록): 모든 차가 선 채로(0 m/s) 출발. 규칙(RoboRacer 2025)은 1대1을 같은 출발선에서
30 cm 떨어져 나란히 세움 -> `F1SIM_EXAM_OPP_SLOT='{"spawn": "alongside"}'`(상대 차 종류·속도는 그대로, 배치만 나란히)와 같이 씀.
기록의 start_rest에 남김.
F1SIM_EXAM_HALF_LAP=1 (11-12, 사용자가 나간 대회의 1대1 출발): 상대 차를 반 바퀴 떨어진 반대쪽에 세움(spawn_gap = 트랙 길이의 절반).
`F1SIM_EXAM_OPP_SLOT='{"spawn": "ahead"}'` + F1SIM_EXAM_START_REST=1과 같이 씀. 기록의 half_lap에 남김.
상자 개수 "race4" (11-24, exam4 = 경주 시험, 사용자와 정한 설계): 기준 policy(F1SIM_EXAM_REF, 기본 scan31b u500 -- 시험 보는 checkpoint가
아님)의 얼린 복사본이 상대 차를 몲(학습의 SelfOpponents 그대로: plan대로 정확히, 느리게 하는 배율 = 참 속도/f에서 plan, 속도 x f).
조건 = {반 바퀴 떨어져 출발, 5 m 뒤에서 출발} x {상대 페이스 0.6, 0.8, 0.95, 제 페이스} x {줄 위 상자 3개(exam3 은행), 상자 없음} = 16가지,
조건마다 같은 수의 세션(F1SIM_EXAM4_PER, 기본 8 -> 128 세션 = 환경 256), 경주 i는 조건 i % 16. 90 s, 상자는 50 s에 치움.
기록에 조건(c_start 0 반 바퀴/1 5 m 뒤, c_pace, c_box), 우리 책임 차 접촉(cfault: env 규칙 car_contact_charge -- 뒤 차, 또는 차 길이
안에 나란히)을 더함; 표는 exam_race_table.py.
F1SIM_EXAM_PRED_ORACLE=1 / F1SIM_EXAM_PRED_PROBE=1 (11-24, 사용자: "예측이 도움이 됐나" -- 입력을 0으로 지우는 건 학습 분포 밖이라 거절, 이 둘만
OK): race4 + 상대 차 예측 머리가 있는 checkpoint(plan_rl --pred-opp)만. 둘 다 결정마다 머리의 예측(우리 차 틀, m)·P(보임), 상대 차 복사본의
자기 plan으로 만든 오라클(같은 형식: 0.1..1.0 s 뒤 10점, 페이스 f인 복사본은 plan의 길을 f배 빠르기로 -> t초 뒤 = plan의 f t초 자리),
'있음' 규칙(학습 타깃과 같음: 10 m(--pred-range) 안 + 1 s 안에 LiDAR 빔에 맞음), 두 차 거리를 `출력_pred.npz`에 남김.
  ORACLE: policy가 읽는 예측 입력을 매 결정 오라클로 바꿔 몲(P(보임) = 규칙대로 1 / 0, 점 x P -- 머리 출력은 logit이고 몸통은 sigmoid를 읽음).
    복사본의 plan을 우리 결정보다 먼저 만듦(복사본은 우리 plan을 안 읽어 값은 같음). 나머지는 보통 exam4 그대로.
  PROBE: 주행은 그대로(기록이 보통 exam4와 비트까지 같아야 함). 결정마다 같은 입력으로 policy를 더 돌려 예측만 바꿨을 때 plan이 얼마나
    움직이나: 예측 점 0.5 m 왼쪽 / 오른쪽 / 앞 / 뒤, P(보임) = 없음; 기준 = 우리 속도 입력(마지막 명령 속도 + 지난 plan 속도 10개) +0.5 m/s,
    지난 plan 9점 0.5 m 왼쪽. 같은 입력으로 다시 돈 plan이 몬 plan과 같은지도 남김(base_err). 분석은 pred_check.py.
  SAVE (11-25, LiDAR 표현 시험 lidar_repr_test.py): 주행은 그대로, 머리의 예측·P(보임)·'있음'·거리·오라클만 남김 (교란 추가 계산 없음).
F1SIM_EXAM4_PACES="a,b,c,d" (11-25, 표현 시험용 기록이지 exam4 점수가 아님): race4의 상대 페이스 4가지를 바꿈 (기본 0.6,0.8,0.95,1.0).
F1SIM_EXAM_SAVE_SCAN=1이면 `_scan.npz`에 결정마다 우리 차의 참 속도·옆속도·회전 속도(ego: vx, vy, r)도 남김 (11-25).
F1SIM_EXAM_SAVE_BOXVIS=1 (11-25, 상자 기억 시험): 결정마다 상자 칸마다 트랙 따라 앞 거리(plan_rl.box_along, 학습의 벽 기준 거리장 -- 타깃
정의용이지 입력이 아님)·살아 있음·지금 LiDAR에 보이나(plan_rl.boxes_in_view: 거리·시야각·벽에 안 가림)를 `_boxvis.npz`에 남김.
F1SIM_EXAM_ALL_BOXES=1 (11-25, line3): 모든 세션에 은행 배치를 차례로 깜 (기본은 리셋 추첨이 상자를 준 절반만)."""
import os, sys, json, argparse, hashlib, time
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
os.environ.pop("F1SIM_WORLD_TRACKS", None)
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")

EXAM = "exam1"
BANK = "/home/shchon11/f1sim_runs/prop_banks_exam1"
SEED = int(os.environ.get("F1SIM_EXAM_SEED", 20261112))
OPP_RESTART_S = 2.0
SLIP = os.environ.get("F1SIM_EXAM_SLIP_FRAME", "")
assert SLIP in ("", "oracle", "table"), SLIP
LEAD = float(os.environ.get("F1SIM_EXAM_SPEED_LEAD") or 0.0)
SAVE_MEM = os.environ.get("F1SIM_EXAM_SAVE_MEM", "") == "1"
SAVE_SCAN = os.environ.get("F1SIM_EXAM_SAVE_SCAN", "") == "1"   # the newest scan the policy read + what each beam hit
SAVE_VALUE = os.environ.get("F1SIM_EXAM_SAVE_VALUE", "") == "1"  # the critic's privileged input and its value
# (11-24, the controller tuning room: tools/map_room.py) "1": every decision's full car state (x, y, yaw, vx, vy, r, steering,
# wheel speed), the plan as MAP reads it (10 segments' curvature and speed, plan_segments) and as it was drawn (the action in
# [-1, 1] and the start speed / steering `plan_steps.roll` took) -> `<output>_plan.npz` (float32); driving unchanged
SAVE_PLAN = os.environ.get("F1SIM_EXAM_SAVE_PLAN", "") == "1"
# (11-24) mode "track" with the model-predictive follower (plan_mpc.PlanMPC, defaults) instead of PlanFollower: "1"
TRACK_MPC = os.environ.get("F1SIM_EXAM_TRACK_MPC", "") == "1"
PLN = {"state": [], "kap": [], "vseg": [], "act": [], "v0": [], "d0": []}
OPP_SLOT = os.environ.get("F1SIM_EXAM_OPP_SLOT", "")                # "" = the training world's opponents
REST = os.environ.get("F1SIM_EXAM_START_REST", "") == "1"        # every car starts standing (the rule's grid)
HALF = os.environ.get("F1SIM_EXAM_HALF_LAP", "") == "1"          # the opponent starts half a lap away (the user's events)
# (11-16, the user: "gru 네트워크가 ... 상대차에 관해서 도움을 줄 수도 있나? ... 근데 이거 측정이 거의 불가할 것 같은데") the memory wiped
# before the decision whenever the opponent is within R m ("near:R"), or at random on a share P of the decisions with NO car
# within R ("rand:P:R", the control: what wiping costs anyway); the wiped rows go to the record ("wiped")
WIPE = os.environ.get("F1SIM_EXAM_WIPE", "")
# (11-16) wiping is confounded: a random 5 % wipe alone cost 85 m / 90 s on the empty track (the memory carries the car's own
# pace). FREEZE keeps the memory as it was (not updated) instead: "near:R" while the opponent is within R m; the control
# "rand:P:L:R" freezes L decisions from a random start (chance P a decision) in races WITHOUT an opponent -- the cost of a
# stale memory alone. The frozen rows go to the record ("frozen")
FREEZE = os.environ.get("F1SIM_EXAM_FREEZE", "")
# (11-19) "1": the box memory (`Actor2.box_memory`, plan_rl --box-memory) cleared before every decision -- the policy drives
# with the box GRU reading only the scan of the moment, nothing remembered of the boxes (does the policy lean on it?)
WIPE_BOX = os.environ.get("F1SIM_EXAM_WIPE_BOX", "") == "1"
# (11-20, the user: "한번 편향이 잡히면 거기서 엎치락뒤치락") "1": drive SAMPLED plans with the policy's own exploration noise, as in
# training (on the empty track the training noise is drawn afresh every decision: --noise-corr-box/-car hold it only near a box
# or a car) -- how far from its line does training ever take the car?
SAMPLE = os.environ.get("F1SIM_EXAM_SAMPLE", "") == "1"
# (11-20) with SAMPLE, as plan_rl's exploration: only the first step's turn and speed get noise (its rl_mask), the turn's noise
# held AR(1) with this rho (plan_rl --noise-corr; 0 = a fresh draw every decision) and its std floored at this many (plan_rl's
# log_std_floor; 0 = the policy's own): how far from its line can exploration take the car?
NOISE_RHO = float(os.environ.get("F1SIM_EXAM_NOISE_RHO", "0") or 0)
STEER_STD_FLOOR = float(os.environ.get("F1SIM_EXAM_STEER_STD_FLOOR", "0") or 0)
# (11-21) with SAMPLE: the first step's SPEED noise held as AR(1) with this rho (0 = fresh, as training) -- how far the approach
# speed to a blind box swings between passes when the speed noise is held 1-2 s (training's fresh noise: 0.20 m/s, needed ~2 m/s)
SPEED_RHO = float(os.environ.get("F1SIM_EXAM_SPEED_RHO", "0") or 0)
# (11-21, measurement for the redesign: can a speed cap alone avoid the blind-corner boxes the policy hits every lap?) "C,D": while a
# box is present within D m ahead along the track, the plan's speeds are capped at C m/s (braking to it at most a_brake a step).
# An oracle: it knows every box from the start (perfect memory, the first pass too)
CAP = [float(x) for x in os.environ.get("F1SIM_EXAM_CAP", "").split(",") if x.strip()]
# (11-21, the user: "미리 알고 미리 라인 다르게" -- a known box is not blind any more) "oracle": every box the car has passed
# once in this session is drawn into the policy's LiDAR input at its true place, as if no wall hid it (min with the real
# scan); the walls, the contacts and the boxes themselves are untouched. Measures whether the avoid-when-visible skill, fed
# what a memory knows, takes a different line before the box shows. Passing = the car's distance to go crosses the box's
VIRT = os.environ.get("F1SIM_EXAM_VIRTUAL_BOXES", "")
# (11-22, plan_rl --line-shifts) "K": a line-level policy drives level K the whole session (0 = its own line) -- run once with
# 0 and once with K on the same sessions; how far apart the two driven lines are is how much a picked level moves the car
LINE_LEVEL = os.environ.get("F1SIM_EXAM_LINE_LEVEL", "")
# (11-21, measurement for "미리 라인 다르게": does trying a different line before a KNOWN box avoid it at speed?) "RHO,STD,D0,D1"
# with SAMPLE: only while a box the car has passed once this session lies D0..D1 m ahead along the track, the first step's
# turn gets noise held as AR(1) (RHO) at least STD wide; everywhere else, and the speed always, the policy's mean
TARGET = [float(x) for x in os.environ.get("F1SIM_EXAM_TARGET_NOISE", "").split(",") if x.strip()]
# (11-16) opponents crash ~0.55 times a minute per box present (they meet boxes on their own line): "1" keeps every box out of
# the racing-line corridor the opponents drive (EnvConfig.procedural_raceline_corridor "on"; the training worlds run "off")
CORRIDOR = os.environ.get("F1SIM_EXAM_CORRIDOR", "") == "1"
# (11-24) a pred_opp checkpoint in race4: the other car's prediction fed from the copy's own plan (ORACLE), or extra forward
# passes with the prediction perturbed, driving unchanged (PROBE); see the docstring
PRED_ORACLE = os.environ.get("F1SIM_EXAM_PRED_ORACLE", "") == "1"
PRED_PROBE = os.environ.get("F1SIM_EXAM_PRED_PROBE", "") == "1"
# (11-25) the head's prediction, 'there' and the oracle recorded, driving unchanged and no extra passes (lidar_repr_test.py)
PRED_SAVE = os.environ.get("F1SIM_EXAM_PRED_SAVE", "") == "1"
assert (PRED_ORACLE + PRED_PROBE + PRED_SAVE) <= 1, "one of F1SIM_EXAM_PRED_ORACLE / _PROBE / _SAVE"
# (11-25) the box memory probe: every box slot's along-track distance ahead, live, in view now (boxes_in_view)
SAVE_BOXVIS = os.environ.get("F1SIM_EXAM_SAVE_BOXVIS", "") == "1"
ALL_BOXES = os.environ.get("F1SIM_EXAM_ALL_BOXES", "") == "1"   # line3: a bank layout in every session

torch.set_grad_enabled(False)
ck_path, out_npz = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "map"
n_env = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
count = sys.argv[5] if len(sys.argv) > 5 else "max2"
secs = float(sys.argv[6]) if len(sys.argv) > 6 else 90.0
rm_t = float(sys.argv[7]) if len(sys.argv) > 7 else 50.0
assert mode in ("map", "track", "ideal"), mode
dev = torch.device(os.environ.get("F1SIM_EXAM_DEVICE", "cuda"))      # "cpu": a handful of sessions for a quick look
ck = torch.load(ck_path, map_location=dev, weights_only=False)
assert ck.get("format") == "plan_steps2", ck.get("format")
# the checkpoint's own scanner place and tyre-limit margin, set before anything reads them
from f1sim.learn.plan_policy import lidar_mount_of
_mx = lidar_mount_of(ck)
if _mx > 0:
    os.environ["F1SIM_LIDAR_MOUNT_X"] = str(_mx)
else:
    os.environ.pop("F1SIM_LIDAR_MOUNT_X", None)
_mg = (ck.get("args") or {}).get("plan_lat_margin")
if _mg is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(_mg)
else:
    os.environ.pop("F1SIM_PLAN_LAT_MARGIN", None)
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import PlanReward
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, plan_world, plan_segments, PT_IDX
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import map_controller as M
from f1sim import dynamics as dyn
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.lidar import HIT_CAR
from f1sim import props as _props

from world_of import world_path, exam_bank, map_name
WORLD = world_path(ck)                                     # the world the checkpoint was trained in (11-14)
if os.environ.get("F1SIM_EXAM_WORLD"):
    # (11-23) another map's exam for this checkpoint: the references (qual26r, ...) trained on the old map, examined on the new
    # bag-scan map (`configs/plan_world_korea26scan_qual.json`) so a new run there has its baseline
    WORLD = os.path.abspath(os.environ["F1SIM_EXAM_WORLD"])
BANK = exam_bank(WORLD)
# (11-18, the user: "시험에서는 막 쭈르륵 장애물 깔지 말고, 빈 트랙에서 가는 길에 장애물을 그 근처에서 뽑히게", "맵당 한 대략 3개 씩?",
# "좀 더 균일하고 라인쪽에 밀집되게", then "ㄱㅊ네" to the preview) count "line3" = exam2: the exam2 bank (exam_line_bank.py: 3 boxes a
# layout on the line three good checkpoints drive on the empty track, the layouts together covering the lap evenly), and every box
# race its own layout, in order -- drawn at random, 56 races from 56 layouts would repeat some and miss others
LINE3 = count == "line3"
if LINE3:
    # (11-23) a world names its own line exam ("exam3" = exam2's design on the bag-scan map with real-size boxes; its
    # "exam_bank" is that bank itself, so the exam1 -> exam2 rename below does nothing there)
    EXAM = str(json.load(open(WORLD)).get("exam_name") or "exam2")
    BANK = BANK.replace("prop_banks_exam1", "prop_banks_exam2")
    assert os.path.isdir(BANK), BANK
if LINE3 and json.load(open(WORLD)).get("solo_world"):
    # (11-24) a race world's line exam (exam3) is its solo world's, exactly (the seat empty, the same grid and draws as the
    # qualifier's exam3): the race world names it
    WORLD = os.path.join(os.path.dirname(WORLD), str(json.load(open(WORLD))["solo_world"]))
    assert os.path.isfile(WORLD), WORLD
RACE4 = count == "race4"
if RACE4:
    # (11-24) exam4: the race exam on the world's own map; boxes = the world's line exam bank (exam3's), written per race below
    EXAM = str(json.load(open(WORLD)).get("race_exam") or "exam4")
    REF = os.path.expanduser(os.environ.get("F1SIM_EXAM_REF", "~/f1sim_runs/exam4_ref/plan_u500.pt"))
    PER = int(os.environ.get("F1SIM_EXAM4_PER", "8"))
    C_PACES = (0.6, 0.8, 0.95, 1.0)
    if os.environ.get("F1SIM_EXAM4_PACES"):                 # (11-25) another set of 4 paces (not exam4's score)
        C_PACES = tuple(float(x) for x in os.environ["F1SIM_EXAM4_PACES"].split(","))
        assert len(C_PACES) == 4, C_PACES
    assert os.path.isdir(BANK), BANK
a = argparse.Namespace(**json.load(open(WORLD)))
if RACE4:
    # every session a 1v1: the seat filled, its scripted kind (ftg, no race line) only stands in -- the copy drives it
    a.stage, a.opp_present, a.spawn_half_lap = "race", 1.0, 0.0
    a.opp_slot = json.dumps({"kind": "ftg", "kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0,
                             "speed_scale": [1.0, 1.0], "spawn": "ahead", "present": 1.0})
if getattr(a, "stage", "race") != "race":
    # (11-17) a single-seat world (the qualifier, stage "props"): this record reads every car's opponent seat, so the exam runs
    # the same world as two-seat races with the seat left empty -- the ego meets and sees exactly what it trained in
    a.stage, a.opp_present = "race", 0.0
    if getattr(a, "empty_seat", None):
        # (11-23) the world names the empty seat's kind: one that drives without a race line ("ftg"), so the exam builds none
        a.opp_slot = json.dumps({"kind": str(a.empty_seat), "kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0})
a.envs, a.seed, a.device = n_env, SEED, dev.type
a.episode_s = secs + 5.0                                   # one episode per car: no respawn inside the session
a.warps, a.track_scales = 0, None                          # the real map only
a.prop_bank, a.prop_count = BANK, ("fixed" if (LINE3 or RACE4) else count)
if OPP_SLOT:
    a.opp_slot = OPP_SLOT                                   # merged over the training world's slot (`state_rl.build_env`)
if mode == "map":
    a.plan_format, a.repeat = "tw", 4
torch.manual_seed(SEED)
_extra = dict({"spawn_speed_max": 0.0} if REST else {}, **({"procedural_raceline_corridor": "on"} if CORRIDOR else {}))
env = SR.build_env(a, dev, extra=_extra or None)
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
if mode == "ideal":
    env.enable_ideal()
elif RACE4:
    env.enable_ideal(learner=False)                         # only the copy drives exactly; MAP / the follower drives ours
if LEAD > 0:
    assert mode == "map", mode
    env.tracker.tw_speed_lead = LEAD
if HALF:
    h = 0.5 * float(env.sim.track.length.min())                 # the real map only: one lap length
    env.ecfg.spawn_gap = (h, h)                                 # read when the reset lays the race out
    print(f"반 바퀴 출발: 트랙 길이 {[round(float(x), 1) for x in env.sim.track.length]} m -> 간격 {h:.1f} m", flush=True)
prepare_graph_runtime(env, log=lambda _t: None)
if RACE4:
    # conditions per race (i % 16): start (0 half a lap apart, 1 we 5 m behind), the copy's pace, boxes on the line or none
    G4 = env.B // env.M
    ci = torch.arange(G4, device=dev) % 16
    c_start, c_pace_i, c_box = ci % 2, (ci // 2) % 4, (ci // 8 == 0)
    c_pace = torch.tensor(C_PACES, device=dev)[c_pace_i]
    h4 = 0.5 * float(env.sim.track.length.min())
    env.spawn_gap_race = torch.where(c_start == 0, torch.full((G4,), h4, device=dev), torch.full((G4,), 5.0, device=dev))
    print(f"[exam4] 경주 {G4}개 = 조건 16가지 x {G4 // 16}: 출발(반 바퀴 {h4:.1f} m / 5 m 뒤) x 상대 페이스 {C_PACES} x 상자(줄 위 3개 / 없음)", flush=True)
env.reset(seed=SEED)
lid = env.learner_ids; Bl = int(lid.numel())
follower = None
if mode == "track":
    # the console's follower with its default gains (`plan_policy.PlanRunner`, mode "track")
    from f1sim.plan_follow import PlanFollower, to_car
    follower = PlanFollower(Bl, dev, k_v=2.0, k_lat=1.0, k_head=1.0, s_max=float(env.s_max), lut=env.tracker.map.lut,
                            grip=M.GRIP_LEVELS.index(1.0), wheelbase=float(env.tracker.nominal["lf"]) + float(env.tracker.nominal["lr"]))
    if TRACK_MPC:
        # (11-24, the controller room's reference row) the model-predictive follower with its defaults (`plan_policy.PlanRunner`'s
        # F1SIM_TRACK_MPC=1: horizon 20, weights 400/100/2500/0, k_v 2)
        from f1sim.plan_mpc import PlanMPC
        follower = PlanMPC(Bl, dev, env.tracker.nominal, s_max=float(env.s_max), lut=env.tracker.map.lut, grip=M.GRIP_LEVELS.index(1.0))
    env.ext_mask[lid] = True; env._ext_ids.update(int(i) for i in lid.tolist())
actor = load_actor2(ck, dev)
if STEER_STD_FLOOR > 0:                     # the first step's turn: exploration at least this wide (plan_rl's log_std_floor)
    actor.log_std_floor[0] = float(np.log(STEER_STD_FLOOR))
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)


class SlipObs(PlanObs2):
    """`PlanObs2` from the frame turned by `turn` (B_l,) [rad] about the CoG: the accelerometer's x / y in that frame (the scan
    is turned at the scanner, `sim.P["mount_yaw"]`; the gyro's z and the last plan in its own frame do not change)"""
    turn = None

    def state(self) -> torch.Tensor:
        s = super().state()
        if self.turn is None:
            return s
        c, s_ = torch.cos(self.turn), torch.sin(self.turn)
        o = s.clone()
        o[:, 3] = s[:, 3] * c + s[:, 4] * s_
        o[:, 4] = -s[:, 3] * s_ + s[:, 4] * c
        return o


if SLIP == "table":
    _t = np.load(os.environ["F1SIM_EXAM_SLIP_TABLE"])
    TAB_A, TAB_V, TAB_B = (torch.tensor(_t[k], dtype=torch.float32, device=dev) for k in ("ay", "v", "beta"))
    assert TAB_B.shape == (len(TAB_A), len(TAB_V)) and float(TAB_B.abs().max()) < 0.5


def table_slip(ay: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[rad] slip angle read from the table (rows = lateral acceleration, columns = speed), bilinear, held at the table's edges"""
    def cell(x, g):
        x = x.clamp(g[0], g[-1]); i = torch.bucketize(x, g).clamp(1, len(g) - 1)
        return i, (x - g[i - 1]) / (g[i] - g[i - 1])
    ia, wa = cell(ay, TAB_A); iv, wv = cell(v, TAB_V)
    return ((1 - wa) * (1 - wv) * TAB_B[ia - 1, iv - 1] + wa * (1 - wv) * TAB_B[ia, iv - 1]
            + (1 - wa) * wv * TAB_B[ia - 1, iv] + wa * wv * TAB_B[ia, iv])


def slip_turn(d_next: torch.Tensor) -> torch.Tensor:
    """[rad] (B_l,) the car's true direction of travel relative to its body (atan2(vy, vx)) minus the plan model's (a
    kinematic bicycle about the CoG at the next plan's first steering angle `d_next`); fades to 0 below 1.5 m/s, 0 backward"""
    s = env.sim.state[lid]
    vx, vy = s[:, dyn.IVX], s[:, dyn.IVY]
    if SLIP == "table":                                         # from what the car knows: speed and yaw rate
        b_real = table_slip(vx * s[:, dyn.IR], vx)
    else:
        b_real = torch.atan2(vy, vx.clamp_min(1e-3))
    b_kin = torch.atan(lim.lr / lim.wheelbase * torch.tan(d_next))
    w = (vx - 0.5).clamp(0.0, 1.0)
    return (w * (b_real - b_kin)).clamp(-0.35, 0.35)


obs = (SlipObs if SLIP else PlanObs2)(env, lim, pr.goal)
if getattr(actor, "pace_levels", 0):                         # (11-21) a pace policy: its state inputs exist before the first read
    obs.pace_on(actor.pace_levels, actor.pace_hold)
# (11-22) a rolling-plan policy (plan_rl --plan-roll) is examined the way it was trained: every decision the committed plan
# keeps its next 0.9 s and the policy's last step is appended; a whole new plan at a start, after a contact, or when something
# out of view (a box: nor passed) when a part was added now blocks the committed path (`plan_rl.roll_blocked`)
ROLL = (bool(ck["args"].get("plan_roll", False)) or bool(ck["args"].get("roll_grow", False))
        or os.environ.get("F1SIM_EXAM_ROLL") == "1")   # =1: any v2 checkpoint, rolled
# (11-22) --roll-grow: the steps a rolling decision kept when this checkpoint was saved (--plan-roll: 9)
ROLL_C = int(ck.get("roll_c", PS.N_STEP - 1)) if ck["args"].get("roll_grow", False) else PS.N_STEP - 1
ROLL_C = int(os.environ.get("F1SIM_EXAM_ROLL_C", ROLL_C))     # (11-23) drive a rolling checkpoint with another commitment
# (11-23) --roll-replan-seen: a whole new plan also when the committed path runs into a wall / box in view or known now
REPLAN_SEEN = bool(ck["args"].get("roll_replan_seen", False)) or os.environ.get("F1SIM_EXAM_REPLAN_SEEN") == "1"
if ROLL:
    from f1sim.learn.plan_rl import roll_blocked, boxes_in_view, box_along
    from f1sim.learn.plan_policy import roll_shift_c
    r_u = torch.zeros(Bl, PS.ACT_DIM, device=dev); r_p = None; r_full = torch.ones(Bl, dtype=torch.bool, device=dev)
    r_w = torch.zeros(Bl, PS.N_STEP, 3, device=dev); r_t = torch.zeros(Bl, PS.N_STEP, dtype=torch.long, device=dev)
    SEENB = None; SCANB = torch.zeros(11, Bl, 3, device=dev); t_dec = 0
    sb_seen = sb_ds = sb_pos = None
    h_mx = float(_mx) if _mx else 0.297; h_rmax = float(env.cfg.lidar.range_max)
    print("[exam] 한 칸씩 이어 붙이는 plan (출발·접촉·안 보이던 게 길을 막을 때만 통째로 새로 그림)", flush=True)
po = env.procedural
if LINE3:
    # the races the reset gave boxes are exam1's (the draw takes the same random numbers whatever the bank holds); each gets bank
    # layout 0, 1, 2, ... in learner order, written into every car of its race
    box_l = lid if ALL_BOXES else lid[(po.p_sid[lid] >= 0).any(1)]
    n_lay = int(po.b_n[0])
    if box_l.numel() > n_lay:
        print(f"[exam2] 상자 경주 {box_l.numel()}개 > 은행 배치 {n_lay}개: 배치를 되풀이해 씀", flush=True)
    k_l = torch.arange(box_l.numel(), device=dev) % n_lay
    t_l = env.sim.tid[box_l]
    sid_l, pose_l = po.b_sid[t_l, k_l], po.b_pose[t_l, k_l]
    mates = env.sim.other_idx[box_l]
    rows_ = torch.cat([box_l[:, None], mates], 1)
    src_ = torch.arange(box_l.numel(), device=dev)[:, None].expand_as(rows_)
    ok_ = rows_ >= 0
    dst_, src_ = rows_[ok_], src_[ok_]
    po.last_live.index_copy_(0, dst_, torch.zeros(box_l.numel(), po.P, dtype=torch.bool, device=dev)[src_])
    po._write(dst_, src_, pose_l[..., :2], pose_l[..., 2], sid_l.clamp_min(0), sid_l >= 0, P=po.C, Q=1)
    _nk = (po.p_sid[box_l] >= 0).sum(1)
    print(f"[exam2] 상자 경주 {box_l.numel()}개에 은행 배치 0..{box_l.numel() - 1}을 차례로: 경주마다 상자 {int(_nk.min())}-{int(_nk.max())}개", flush=True)
if RACE4:
    # the box races get the line bank's layouts in order, the others none; every car of a race the same
    n_lay = int(po.b_n[0])
    rl_ = torch.arange(G4, device=dev) * env.M                      # the learner row of each race
    kb_ = torch.cumsum(c_box.long(), 0) - 1
    k_l = torch.where(c_box, kb_ % n_lay, torch.zeros_like(kb_))
    t_l = env.sim.tid[rl_]
    sid_l, pose_l = po.b_sid[t_l, k_l], po.b_pose[t_l, k_l]
    live_l = (sid_l >= 0) & c_box[:, None]
    rows_ = torch.cat([rl_[:, None], env.sim.other_idx[rl_]], 1)
    src_ = torch.arange(G4, device=dev)[:, None].expand_as(rows_)
    dst_, src_ = rows_.reshape(-1), src_.reshape(-1)
    po.last_live.index_copy_(0, dst_, torch.zeros(G4, po.P, dtype=torch.bool, device=dev)[src_])
    po._write(dst_, src_, pose_l[..., :2], pose_l[..., 2], sid_l.clamp_min(0), live_l, P=po.C, Q=1)
    _nk = (po.p_sid[rl_] >= 0).sum(1)
    print(f"[exam4] 상자 경주 {int(c_box.sum())}개에 은행 배치 0..{int(c_box.sum()) - 1}: 경주마다 상자 {int(_nk[c_box].min())}-{int(_nk[c_box].max())}개, "
          f"상자 없는 경주 {int((~c_box).sum())}개 (상자 {int(_nk[~c_box].max())}개)", flush=True)
remover = SR.PropRemoval(env, 1.0, rm_t, rm_t, seed=SEED) if po is not None else None
KNOWN = bool(VIRT) or bool(TARGET)
if KNOWN:
    v_seen = torch.zeros(Bl, po.C, dtype=torch.bool, device=dev)    # passed at least once this session
    v_prev = torch.full((Bl, po.C), float("nan"), device=dev)       # last decision's distance along the track to the box
    v_zone = torch.zeros(Bl, dtype=torch.bool, device=dev)          # TARGET: a known box D0..D1 m ahead
    t_stat = {"zone": 0.0, "n": 0.0}
if VIRT:
    assert VIRT == "oracle" and po is not None, VIRT
    _fp = [torch.as_tensor(np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint), dtype=torch.float32)
           for sh in po.shapes]
    _V = max(int(f.shape[0]) for f in _fp)
    FPV = torch.zeros(len(_fp), _V, 2, device=dev)                 # each shape's footprint, the last vertex repeated to _V
    for _k, _f in enumerate(_fp):
        FPV[_k, :_f.shape[0]] = _f.to(dev); FPV[_k, _f.shape[0]:] = _f[-1].to(dev)
    _ang = env.sim.lidar.angles.float()
    _mx = float(os.environ.get("F1SIM_LIDAR_MOUNT_X") or 0.297)
    _orig_scan = obs._scan
    v_stat = {"drawn": 0.0, "n": 0.0}

    def _virt_scan():
        sc_ = _orig_scan()                                           # (Bl, beams) normalized
        sid_ = po.p_sid[lid]; ok_ = (sid_ >= 0) & v_seen
        if not bool(ok_.any()):
            return sc_
        st_ = env.sim.state[lid].float()
        sx = st_[:, 0] + _mx * torch.cos(st_[:, 2]); sy = st_[:, 1] + _mx * torch.sin(st_[:, 2])
        a_ = st_[:, 2:3] + _ang[None, :]                             # (Bl, N) world beam directions
        dx, dy = torch.cos(a_), torch.sin(a_)
        pose_ = po.p_poses[lid].float()                              # (Bl, C, 3)
        fp_ = FPV[sid_.clamp_min(0)]                                 # (Bl, C, V, 2) box frame
        c_, s_ = torch.cos(pose_[..., 2])[..., None], torch.sin(pose_[..., 2])[..., None]
        wx = pose_[..., 0:1] + c_ * fp_[..., 0] - s_ * fp_[..., 1]; wy = pose_[..., 1:2] + s_ * fp_[..., 0] + c_ * fp_[..., 1]
        ax, ay = wx, wy; bx = torch.roll(wx, -1, -1); by = torch.roll(wy, -1, -1)          # edges A -> B, (Bl, C, V)
        ex, ey = bx - ax, by - ay
        # ray S + t d against every edge: t = cross(A - S, E) / cross(d, E), u = cross(A - S, d) / cross(d, E)
        asx = (ax - sx[:, None, None])[:, None]; asy = (ay - sy[:, None, None])[:, None]        # (Bl, 1, C, V)
        dxx = dx[:, :, None, None]; dyy = dy[:, :, None, None]                                  # (Bl, N, 1, 1)
        den = dxx * ey[:, None] - dyy * ex[:, None]
        t_ = (asx * ey[:, None] - asy * ex[:, None]) / den
        u_ = (asx * dyy - asy * dxx) / den
        hit = (den.abs() > 1e-9) & (t_ > 0) & (u_ >= 0) & (u_ <= 1) & ok_[:, None, :, None]
        rng_ = torch.where(hit, t_, torch.full_like(t_, float("inf"))).amin((2, 3))            # (Bl, N) [m]
        v_ = (rng_ / obs.range).clamp(0.0, 1.0)
        out_ = torch.minimum(sc_, v_)
        v_stat["drawn"] += float((out_ < sc_ - 1e-6).float().sum()); v_stat["n"] += float(sc_.numel())
        return out_
    obs._scan = _virt_scan
restarter = SR.OppRestart(env, OPP_RESTART_S, PS.DT) if env.M > 1 else None
so = None
if RACE4:
    # the other car: a frozen copy of the FIXED reference policy, at the race's pace (plan_rl.SelfOpponents, as in training)
    from f1sim.learn.plan_rl import SelfOpponents
    ref_ck = torch.load(REF, map_location=dev, weights_only=False)
    assert abs(lidar_mount_of(ref_ck) - _mx) < 1e-6, "the reference policy's LiDAR place differs from the examined one's"
    so = SelfOpponents(env, lim, 1.0, seed=SEED, max_pool=1, p_newest=1.0, pace=(0.6, 0.95, 0.0))
    so.add(load_actor2(ref_ck, dev), int(ref_ck.get("update") or 0))
    so.pace[:] = c_pace
    so.on[:] = True
    R_c = {"c_start": c_start.cpu().numpy(), "c_pace": c_pace.cpu().numpy(), "c_box": c_box.cpu().numpy()}
    print(f"[exam4] 상대 차 = {REF}의 얼린 복사본 (update {ref_ck.get('update')})", flush=True)
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
PRED = PRED_ORACLE or PRED_PROBE or PRED_SAVE
if PRED:
    assert RACE4 and so is not None and getattr(actor, "pred_opp", False), "PRED_ORACLE / PRED_PROBE: race4 and a pred_opp checkpoint"
    assert getattr(actor, "branches", 1) == 1 and not getattr(actor, "pace_levels", 0) and not ROLL and not SLIP
    from f1sim.learn.plan_policy import PRED_N, PRED_SCALE, pred_points
    from f1sim.learn.plan_reward import SEEN_S
    PRED_RANGE = float(ck["args"].get("pred_range") or 10.0)
    _orow = torch.full((env.B,), -1, dtype=torch.long, device=dev)
    _orow[so.rows] = torch.arange(so.rows.numel(), device=dev)
    O_OF = _orow[others]                                    # each learner's other car among the copy's rows
    assert bool((O_OF >= 0).all())
    PRD = {k: [] for k in ("there", "dist", "head", "head_p", "orc")}
    _kk = torch.arange(1, PRED_N + 1, device=dev, dtype=torch.float32) * PS.SUB      # 0.1 .. 1.0 s on the dense grid

    def opp_oracle(st_me: torch.Tensor) -> torch.Tensor:
        """the copy's own plan of this decision (`so.plan` done) -> where it will be 0.1 .. 1.0 s on, (B_l, PRED_N, 2) [m] in
        our car's frame at the decision (as plan_rl's --pred-opp targets). A copy at pace f drives its plan's path f times
        as fast (`SelfOpponents.PACE_KEYS`): t s on = the plan's pose at f t (linear between the dense poses)"""
        sub = {k_: so.p[k_][O_OF] for k_ in ("x", "y", "psi")}
        xw, yw, _ = plan_world(sub, env.sim.state[so.rows[O_OF]])
        tau = so.pace[O_OF].float()[:, None] * _kk[None, :]
        i0 = tau.floor().long().clamp(0, xw.shape[1] - 2)
        w = (tau - i0.float()).clamp(0.0, 1.0)
        px = xw.gather(1, i0) * (1.0 - w) + xw.gather(1, i0 + 1) * w
        py = yw.gather(1, i0) * (1.0 - w) + yw.gather(1, i0 + 1) * w
        dx, dy = px - st_me[:, 0:1], py - st_me[:, 1:2]
        c_, s_ = torch.cos(st_me[:, dyn.IYAW])[:, None], torch.sin(st_me[:, dyn.IYAW])[:, None]
        return torch.stack([dx * c_ + dy * s_, -dx * s_ + dy * c_], -1)

    def feed_of(pts_m: torch.Tensor, p_vis: torch.Tensor) -> torch.Tensor:
        """what the trunk reads (`Actor2.pred_feed`'s layout): P(visible), then the points / PRED_SCALE x P(visible)"""
        return torch.cat([p_vis[:, None], (pts_m / PRED_SCALE).reshape(pts_m.shape[0], -1) * p_vis[:, None]], 1)

if PRED_PROBE:
    # the perturbations (name, kind, value): the prediction's points shifted [m] (x ahead, y left, our car's frame), the
    # prediction's P(visible) set to 0; references on our own inputs (STATE_DIM2's layout: imu 0-5, last steering 6, last
    # speed 7, the last plan's points 2..10 8-25 (x, y), its speeds 26-35)
    PROBES = [("pred_left", "shift", (0.0, 0.5)), ("pred_right", "shift", (0.0, -0.5)), ("pred_ahead", "shift", (0.5, 0.0)),
              ("pred_behind", "shift", (-0.5, 0.0)), ("pred_gone", "gone", None),
              ("own_speed+0.5", "state", ([7] + list(range(26, 36)), 0.5)), ("own_lastplan_left0.5", "state", (list(range(9, 26, 2)), 0.5))]
    PCOLS = ("mean_disp", "disp_05", "disp_10", "dv_abs", "dy_10", "dx_10", "dv_mean")
    PRD.update(probe=[], base_err=[])

    def plan_from(scans: torch.Tensor, s_in: torch.Tensor, h_in: torch.Tensor, cmd_d: torch.Tensor, pin_of=None):
        """one forward pass of the actor as `plan_v2` runs it (mean plan), touching neither `obs` nor the driving state;
        `pin_of(q)` replaces what the trunk reads of the prediction q"""
        f_ = actor.features(scans, s_in)
        h1 = actor.mem_cell(f_, h_in)
        q_ = actor.pred_q(scans, f_, h1)
        pin = actor.pred_feed(q_) if pin_of is None else pin_of(q_)
        m_, _, vh = actor.heads(actor.mem_trunk(f_, h1, pin))
        return PS.roll(torch.tanh(m_), vh.clamp(-PS.V_REV, PS.V_FWD), cmd_d, lim), q_

    def probe_all(scans, s_in, h_in, cmd_d, p_drv) -> torch.Tensor:
        """(B_l, len(PROBES), len(PCOLS)): how far each perturbation moves the plan from the unperturbed one"""
        pb, qb = plan_from(scans, s_in, h_in, cmd_d)
        PRD["base_err"].append(torch.stack([(pb["pts"] - p_drv["pts"]).abs().amax(), (pb["pv"] - p_drv["pv"]).abs().amax()]).float().cpu())
        out_ = []
        for _nm, kind, val in PROBES:
            if kind == "shift":
                d_ = torch.tensor(val, device=dev) / PRED_SCALE
                pin_of = lambda q, d_=d_: type(actor).pred_feed(torch.cat([(q[:, :2 * PRED_N].view(-1, PRED_N, 2) + d_).reshape(q.shape[0], -1), q[:, -1:]], 1))
                pk, _ = plan_from(scans, s_in, h_in, cmd_d, pin_of)
            elif kind == "gone":
                pk, _ = plan_from(scans, s_in, h_in, cmd_d, lambda q: torch.zeros(q.shape[0], 2 * PRED_N + 1, device=dev))
            else:
                ix_, dv_ = val
                s2 = s_in.clone(); s2[:, ix_] = s2[:, ix_] + dv_
                pk, _ = plan_from(scans, s2, h_in, cmd_d)
            dp = pk["pts"] - pb["pts"]; dd = dp.norm(dim=-1); dv = pk["pv"] - pb["pv"]
            out_.append(torch.stack([dd.mean(1), dd[:, 4], dd[:, 9], dv.abs().mean(1), dp[:, 9, 1], dp[:, 9, 0], dv.mean(1)], 1))
        return torch.stack(out_, 1)

COLS =("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")
I_CNT = [CC.index(k) for k in COLS]
i_hk, i_hx, i_hy, i_hs = (CC.index(k) for k in ("held_kind", "held_x", "held_y", "held_severity"))
KN = list(getattr(env, "alt_teacher_kinds", ())) + [str(json.loads(OPP_SLOT).get("kind", "raceline")) if OPP_SLOT else "raceline"]
ii = PT_IDX.to(dev)
T = int(round(secs / PS.DT))
R = {k: [] for k in ("st", "op", "on", "pl", "rs", "cnt", "ocnt", "held", "sid", "pose", "tid", "kind", "scale", "lap", "ors",
                     "rem", "orem")}
if RACE4:
    R["cfault"] = []                                        # car contact onsets this decision that the env's rule puts on us
TWO = getattr(actor, "branches", 1) == 2
if TWO:
    # a two-plan policy (11-13): which plan was driven, both logits, and the plan NOT driven (world points, as "pl")
    R["br"] = []; R["logit"] = []; R["pl_other"] = []
    obs.prev_branch = torch.zeros(Bl, device=dev)           # the state input exists from the first decision on
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
turn = torch.zeros(Bl, device=dev)                          # the frame the newest scan was taken in (SLIP)
if SLIP:
    R["dlt"] = []; R["vy"] = []
else:
    R["vy"] = []                                            # 11-16: the side speed every decision -> slip angle (sliding)
    yaw0 = env.sim.P["mount_yaw"][lid].clone()
    # the scanner does turn with P["mount_yaw"] (a compiled scan path could have kept its angles): a scan taken 0.2 rad round
    # must be the scan before, 0.2 rad of beams along (one control step of a standing car, before the session's first decision)
    if mode == "map":
        env.ext_plan_kv = (kk, vv)                          # a standing plan
    _s0 = obs._scan().clone(); env.sim.P["mount_yaw"][lid] = yaw0 + 0.2; env.step(act0)
    _s1 = obs._scan().clone(); env.sim.P["mount_yaw"][lid] = yaw0
    _n = int(round(0.2 / float(env.sim.lidar.angles[1] - env.sim.lidar.angles[0])))
    _e_turn = float((_s1[:, :-_n] - _s0[:, _n:]).abs().median()); _e_same = float((_s1 - _s0).abs().median())
    print(f"스캐너 돌림 확인: 0.2 rad({_n} 빔) 돌린 스캔과 원래 스캔을 {_n} 빔 민 것의 차 중앙 {_e_turn:.4f}, 안 민 것과의 차 {_e_same:.4f} (거리/최대거리)", flush=True)
    assert _e_turn < 0.25 * max(_e_same, 1e-4), "the scan did not turn with P['mount_yaw']"
    env.step(act0); obs.reset_rows(torch.arange(Bl, device=dev))


def snap():
    """the world as the decision about to be made sees it"""
    R["st"].append(env.sim.state[lid, :4].float().cpu()); R["op"].append(env.sim.state[others, :4].float().cpu())
    R["on"].append((~absent[others]).cpu())
    if po is None:
        R["sid"].append(torch.full((Bl, 1), -1, dtype=torch.int16)); R["pose"].append(torch.zeros(Bl, 1, 3))
    else:
        R["sid"].append(po.p_sid[eid].to(torch.int16).cpu()); R["pose"].append(po.p_poses[eid].float().cpu())
    R["tid"].append(env.sim.tid[lid].to(torch.int16).cpu())
    code = env.slot_kind_code[others] if hasattr(env, "slot_kind_code") else torch.full((Bl,), -1, device=dev, dtype=torch.long)
    R["kind"].append(torch.where(code < 0, torch.full_like(code, len(KN) - 1), code).to(torch.int8).cpu())
    sc = env.teacher.speed_scale[others] if torch.is_tensor(getattr(env.teacher, "speed_scale", None)) else torch.ones(Bl, device=dev)
    R["scale"].append(sc.float().cpu())
    # the walls-only distance to go of both cars (NaN in a wall / off the grid)
    R["rem"].append(pr.goal.remaining(env.sim.state[lid, :2].float(), env.sim.tid[lid]).float().cpu())
    R["orem"].append(pr.goal.remaining(env.sim.state[others, :2].float(), env.sim.tid[others]).float().cpu())


MEM = {"feat": [], "mem": []}
if SAVE_MEM:
    assert getattr(actor, "memory", 0), "이 checkpoint에는 기억이 없습니다"
SCN = {"scan": [], "typ": [], "ego": []}
BXV = {"ds": [], "live": [], "view": []}
if SAVE_BOXVIS:
    assert po is not None, "F1SIM_EXAM_SAVE_BOXVIS: a world with boxes"
    from f1sim.learn.plan_rl import box_along as _box_along, boxes_in_view as _boxes_in_view
    _bv_mx = float(_mx) if _mx else 0.297; _bv_rmax = float(env.cfg.lidar.range_max)
VAL = {"priv": [], "value": []}
if SAVE_VALUE:
    from f1sim.learn.plan_policy import Critic
    from f1sim.learn.plan_rl import track_boxes, PRIV_V2_PROP_SLOTS
    _br = float(ck["args"].get("box_range") or 0.0)
    assert not ck["args"].get("hold_p"), "a hold-trained critic also reads the decisions left on the plan in hand"
    critic = Critic(obs.beams, int(ck["priv_dim"]), state_dim=int(obs.state().shape[1])).to(dev)
    critic.load_state_dict(ck["critic"]); critic.eval()
    _P0 = _w = 0
    if _br > 0:                                             # plan_rl's --box-range: the box block's columns, refilled along the track
        _pv0, _pb0 = obs.priv(), env._prop_block(env.sim.state)[lid].float()
        _w = _pb0.shape[1]
        _hit = [j for j in range(_pv0.shape[1] - _w + 1) if torch.allclose(_pv0[:, j:j + _w].float(), _pb0, atol=1e-5)]
        assert len(_hit) == 1 and float(_pb0.abs().sum()) > 0.0, f"the critic's box block not found unambiguously ({_hit})"
        _P0 = _hit[0]

    def priv_now():
        pv = obs.priv()
        if _br > 0:
            pv[:, _P0:_P0 + _w] = track_boxes(env, pr.goal, lid, PRIV_V2_PROP_SLOTS, _br)
        return pv
t0 = time.time()
for kd in range(T):
    snap()
    st = env.sim.state[lid].clone()
    if not SLIP:
        R["vy"].append(st[:, dyn.IVY].float().cpu())
    if SLIP:
        R["dlt"].append(turn.float().cpu()); R["vy"].append(st[:, dyn.IVY].float().cpu())
        st[:, dyn.IYAW] = st[:, dyn.IYAW] + turn            # the plan starts in the frame its scan was taken in
        obs.turn = turn
    if SAVE_MEM:
        MEM["feat"].append(torch.cat([actor.enc(obs.scans), actor.norm(obs.state())], 1).half().cpu())
    if SAVE_SCAN:
        SCN["scan"].append(obs.scans[:, 0].half().cpu()); SCN["typ"].append(env.last_result.scan_type[lid].to(torch.uint8).cpu())
        SCN["ego"].append(st[:, [dyn.IVX, dyn.IVY, dyn.IR]].float().cpu())
    if SAVE_BOXVIS:
        _ds, _lv, _ps = _box_along(env, pr.goal, lid)
        BXV["ds"].append(_ds.half().cpu()); BXV["live"].append(_lv.cpu())
        BXV["view"].append(_boxes_in_view(env, lid, _ps, _lv, _bv_mx, _bv_rmax).cpu())
    if SAVE_VALUE:
        _pv = priv_now()
        VAL["priv"].append(_pv.half().cpu()); VAL["value"].append(critic(obs.scans, obs.state(), _pv).float().cpu())
    if WIPE:
        _w = WIPE.split(":")
        _near = (~absent[others]) & ((env.sim.state[lid, :2] - env.sim.state[others, :2]).norm(dim=1) < float(_w[-1]))
        _wipe = _near if _w[0] == "near" else (~_near) & (torch.rand(Bl, device=dev) < float(_w[1]))
        if getattr(obs, "mem_h", None) is not None:
            obs.mem_h[_wipe] = 0.0
        R.setdefault("wiped", []).append(_wipe.cpu())
    if WIPE_BOX and getattr(actor, "box_memory", 0) and getattr(obs, "mem_h", None) is not None:
        obs.mem_h[:, actor.memory_main:] = 0.0
    if FREEZE:
        _f = FREEZE.split(":")
        _near = (~absent[others]) & ((env.sim.state[lid, :2] - env.sim.state[others, :2]).norm(dim=1) < float(_f[-1]))
        if _f[0] == "near":
            _fz = _near
        else:
            if kd == 0:
                fz_left = torch.zeros(Bl, dtype=torch.long, device=dev)
            _st = absent[others] & (fz_left == 0) & (torch.rand(Bl, device=dev) < float(_f[1]))
            fz_left[_st] = int(_f[2])
            _fz = fz_left > 0
            fz_left = (fz_left - 1).clamp_min(0)
        h_prev = obs.mem_h.clone() if getattr(obs, "mem_h", None) is not None else None
        R.setdefault("frozen", []).append(_fz.cpu())
    cmd_d0 = obs.cmd_d.clone()
    _eps = None
    if SAMPLE:
        if "_eps_n" not in globals() or _eps_n.shape[0] != obs.scans.shape[0]:
            _eps_n = torch.randn(obs.scans.shape[0], PS.ACT_DIM, device=obs.scans.device)
        _rs = env_restarted if "env_restarted" in globals() and env_restarted is not None else None
        _fresh = torch.randn_like(_eps_n)
        _sp = SPEED_RHO * _eps_n[:, PS.N_STEP] + (1.0 - SPEED_RHO ** 2) ** 0.5 * _fresh[:, PS.N_STEP]
        _eps_n[:, 0] = NOISE_RHO * _eps_n[:, 0] + (1.0 - NOISE_RHO ** 2) ** 0.5 * _fresh[:, 0]
        _eps_n[:, 1:] = _fresh[:, 1:]
        _eps_n[:, PS.N_STEP] = _sp
        if _rs is not None:
            _eps_n[_rs] = torch.randn_like(_eps_n[_rs])
        _eps = torch.zeros_like(_eps_n); _eps[:, 0] = _eps_n[:, 0]; _eps[:, PS.N_STEP] = _eps_n[:, PS.N_STEP]
        if TARGET:
            # held turn noise only in the zone, at least TARGET[1] wide (the policy's own std scaled up); no speed noise
            if "_tn" not in globals():
                _tn = torch.randn(obs.scans.shape[0], device=obs.scans.device)
            _tn = TARGET[0] * _tn + (1.0 - TARGET[0] ** 2) ** 0.5 * torch.randn_like(_tn)
            _sd0 = float(torch.maximum(actor.log_std, actor.log_std_floor)[0].exp())
            _eps = torch.zeros_like(_eps_n)
            _eps[:, 0] = torch.where(v_zone, _tn * max(1.0, TARGET[1] / _sd0), torch.zeros_like(_tn))
    _cap = None
    if CAP and po is not None:                                  # F1SIM_EXAM_CAP: a box present within D m ahead -> the cap
        _tid = env.sim.tid[lid]
        _dc = pr.goal.remaining(env.sim.state[lid, :2].float(), _tid)                                   # (Bl,)
        _pb = po.p_poses[lid][..., :2].float()                                                        # (Bl, P, 2)
        _db = pr.goal.remaining(_pb.reshape(-1, 2), _tid[:, None].expand(-1, _pb.shape[1]).reshape(-1)).view(_pb.shape[:2])
        _L = pr.goal.lap[_tid][:, None]
        _ah = torch.remainder(_dc[:, None] - _db, _L)
        _ok = (po.p_sid[lid] >= 0) & torch.isfinite(_ah) & (_ah < CAP[1])
        _cap = torch.where(_ok.any(1), torch.full_like(_dc, CAP[0]), torch.full_like(_dc, float("inf")))
    if ROLL:                                                    # the known boxes and the scanner now; is the committed path blocked
        _ds, _live, _pos = box_along(env, pr.goal, lid)
        if sb_seen is None:
            sb_seen = torch.zeros_like(_live); sb_ds = torch.full_like(_ds, float("nan")); sb_pos = _pos.clone()
        _moved = ((_pos - sb_pos).abs().sum(-1) > 0.05) | ~_live
        _passed = _live & ~_moved & (sb_ds > 0.0) & (sb_ds <= 3.0) & (_ds <= 0.0) & (_ds > -3.0)
        _view = boxes_in_view(env, lid, _pos, _live, h_mx, h_rmax)
        sb_seen = (sb_seen & ~_moved) | _passed | _view
        sb_ds = torch.where(_live, _ds, torch.full_like(_ds, float("nan"))); sb_pos = _pos
        _sn = env.sim.state[lid, :3].float()
        SCANB[t_dec % 11] = torch.stack([_sn[:, 0] + h_mx * torch.cos(_sn[:, 2]), _sn[:, 1] + h_mx * torch.sin(_sn[:, 2]), _sn[:, 2]], 1)
        if SEENB is None:
            SEENB = torch.zeros(11, *sb_seen.shape, dtype=torch.bool, device=dev)
        SEENB[t_dec % 11] = sb_seen
        _blk = torch.zeros(Bl, dtype=torch.bool, device=dev)
        if r_p is not None and bool((~r_full).any()):
            _ix = r_t[:, :ROLL_C + 1] % 11; _ar = torch.arange(Bl, device=dev)[:, None]     # the committed points only
            _kn = ((sb_seen & ~_moved)[:, None, :].expand(-1, ROLL_C + 1, -1) if REPLAN_SEEN
                   else SEENB[_ix, _ar] & ~_moved[:, None, :])
            _bw, _bb = roll_blocked(env, lid, pr, r_w[:, :ROLL_C + 1], _kn, SCANB[_ix, _ar], _view, h_mx, h_rmax,
                                    any_seen=REPLAN_SEEN)
            _blk = (_bw | _bb) & ~r_full
            r_full = r_full | _blk
        R.setdefault("newp", []).append(r_full.to(torch.int8).cpu()); R.setdefault("blocked", []).append(_blk.to(torch.int8).cpu())
    if PRED:
        # the other car "there" by the rule plan_rl's --pred-opp target uses (not absent, within --pred-range, a beam on a car
        # within SEEN_S)
        _dist = (env.sim.state[others, :2] - st[:, :2]).norm(dim=1)
        _there = (~absent[others]) & (_dist <= PRED_RANGE) & (pr.seen_t <= SEEN_S)
        PRD["there"].append(_there.cpu()); PRD["dist"].append(_dist.float().cpu())
    if PRED_ORACLE:
        so.plan()                                               # the copy's plan first (it does not read ours): the oracle
        _orc = opp_oracle(st)
        PRD["orc"].append(_orc.float().cpu())
        _pin = feed_of(_orc, _there.float())
        actor.pred_feed = lambda q_, _p=_pin: _p.to(q_.dtype)    # the trunk reads the oracle (the head still runs: pred_out)
    if PRED_PROBE:
        _scans0, _s0 = obs.scans.clone(), obs.state().clone()
        _h0 = (obs.mem_h.clone() if getattr(obs, "mem_h", None) is not None
               else torch.zeros(Bl, actor.memory, device=dev))
    u_, mean, _, v_hat, p = (plan_v2(actor, obs, lim, sample=SAMPLE, eps=_eps, cap=_cap) if (_eps is not None or _cap is not None)
                             else plan_v2(actor, obs, lim, sample=SAMPLE))
    if os.environ.get("F1SIM_EXAM_SAVE_ACT") == "1":            # (11-22) every output the policy gave (pre-tanh), driven or not
        R.setdefault("uall", []).append(u_.float().cpu())
    if PRED:                                                    # the head's own answer this decision (computed in either mode)
        _hp, _hv = pred_points(actor.pred_out.float())
        PRD["head"].append(_hp.cpu()); PRD["head_p"].append(_hv.cpu())
    if ROLL:                                                    # the committed 0.9 s rolled on, the policy's last step appended
        u_x = u_ if r_p is None else torch.where(r_full[:, None], u_, roll_shift_c(r_u, u_, ROLL_C))
        v_s = v_hat.detach().clamp(-PS.V_REV, PS.V_FWD); d_s = cmd_d0
        if r_p is not None:
            v_s = torch.where(r_full, v_s, r_p["v"][:, PS.SUB]); d_s = torch.where(r_full, d_s, r_p["delta"][:, PS.SUB])
        p = PS.roll(torch.tanh(u_x), v_s, d_s, lim)
        r_t = torch.where(r_full[:, None], torch.full_like(r_t, t_dec),
                          torch.cat([r_t[:, 1:ROLL_C + 1], torch.full_like(r_t[:, ROLL_C:], t_dec)], 1))
        r_u, r_p = u_x, p
        if os.environ.get("F1SIM_EXAM_SAVE_ACT") == "1":            # (11-22) what a rolling policy appends: the plan's speeds
            R.setdefault("plv", []).append(p["pv"].float().cpu()); R.setdefault("ulast", []).append(u_[:, [PS.N_STEP - 1, 2 * PS.N_STEP - 1]].float().cpu())
        _xr, _yr, _pr = plan_world(p, env.sim.state[lid]); _ir = PT_IDX.to(dev)
        r_w = torch.stack([_xr[:, _ir], _yr[:, _ir], _pr[:, _ir]], -1)
    if LINE_LEVEL != "" and getattr(obs, "pace_idx", None) is not None:   # F1SIM_EXAM_LINE_LEVEL: the level held fixed
        obs.pace_idx[:] = int(LINE_LEVEL)
    if getattr(obs, "pace_idx", None) is not None:              # (11-21) a pace policy: the level in force each decision
        R.setdefault("pace", []).append(obs.pace_idx.to(torch.int8).cpu())
    if FREEZE and h_prev is not None:
        obs.mem_h = torch.where(_fz[:, None], h_prev, obs.mem_h)
    if SAVE_MEM:
        MEM["mem"].append(obs.mem_h.half().cpu())
    x, y, psi = plan_world(p, st)
    if TWO:
        br = actor.branch_out
        p_o = PS.roll(torch.tanh(actor.means_out[torch.arange(Bl, device=dev), 1 - br]),
                      v_hat.detach().clamp(-PS.V_REV, PS.V_FWD), cmd_d0, lim)
        xo, yo, _ = plan_world(p_o, st)
        R["br"].append(br.to(torch.int8).cpu()); R["logit"].append(actor.logits_out.float().cpu())
        R["pl_other"].append(torch.stack([xo[:, ii], yo[:, ii]], -1).cpu())
    pc = pr.plan_cost(x, y, psi, p["v"])
    if os.environ.get("F1SIM_EXAM_SAVE_ACT") == "1":            # (11-22) where the plan driven first meets a wall / anything
        c_, s_ = torch.cos(psi)[..., None], torch.sin(psi)[..., None]; P_ = pr.corners
        pw_ = torch.stack([x[..., None] + P_[:, 0] * c_ - P_[:, 1] * s_, y[..., None] + P_[:, 0] * s_ + P_[:, 1] * c_], -1)
        wh_ = (env.sim.track.sample_edt(pw_, env.sim.tid[lid][:, None, None].expand(pw_.shape[:-1])) <= 0.0).any(-1); wh_[:, 0] = False
        R.setdefault("pwall", []).append(torch.where(wh_.any(1), wh_.float().argmax(1), torch.full_like(wh_[:, 0], -1, dtype=torch.long)).to(torch.int16).cpu())
        R.setdefault("pany", []).append(torch.where(pc["hit"], pc["first"], torch.full_like(pc["first"], -1)).to(torch.int16).cpu())
    pts = torch.stack([x[:, ii], y[:, ii]], -1)
    if SAVE_PLAN:
        _k, _v = plan_segments(p, lim)
        PLN["state"].append(st.float().cpu()); PLN["kap"].append(_k.float().cpu()); PLN["vseg"].append(_v.float().cpu())
        PLN["act"].append(torch.tanh(u_).float().cpu()); PLN["v0"].append(p["v"][:, 0].float().cpu()); PLN["d0"].append(p["delta"][:, 0].float().cpu())
    if mode == "map":
        k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    elif mode == "track":
        follower.set_plan(slice(None), x, y, psi, p["v"], p["delta"])
    else:
        env.sim.ideal_vcap[lid] = 1e3
    restarted = torch.zeros(Bl, dtype=torch.bool, device=dev)
    if so is not None:
        if not PRED_ORACLE:                                     # (ORACLE: made before our decision)
            so.plan()
        cf_ = torch.zeros(Bl, device=dev)
    if PRED_PROBE or PRED_SAVE:                                 # driving above is done; these passes touch nothing it reads
        PRD["orc"].append(opp_oracle(st).float().cpu())
        PRD["probe"].append(probe_all(_scans0, _s0, _h0, cmd_d0, p).cpu())
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        if follower is not None:
            env.ext_cmd[lid] = to_car(env, lid, follower.command(env.sim.state[lid]))
        if SLIP:
            turn = slip_turn(p["delta"][:, PS.SUB]); env.sim.P["mount_yaw"][lid] = yaw0 + turn
        if so is not None:
            so.command(j)
        _, _, te, trc, _ = env.step(act0)
        restarted |= (te | trc)[lid]
        if so is not None:
            so.stepped(te | trc)
            on_ = env.last_result.contact_onsets
            if on_ is not None and bool(on_[lid, 2].any()):
                cf_ += (on_[lid, 2] & pr.at_fault()).float()
        pr.control_step((env.last_result.scan_type[lid] == HIT_CAR).any(1))
    out = pr.decision(pts, pc["price"], restarted)
    if ROLL:                                                    # off the plan (held / pushed), a restart or a contact: a whole plan
        r_full = restarted | ((env.sim.state[lid, :2].float() - r_w[:, 0, :2]).norm(dim=-1) > 0.01) | (out["contact"] < 0)
        if ROLL_C == 0:                                         # --roll-grow at 0 kept: every decision a whole new plan
            r_full = torch.ones_like(r_full)
        t_dec += 1
    if KNOWN:
        _tid = env.sim.tid[lid]
        _dc = pr.goal.remaining(env.sim.state[lid, :2].float(), _tid)
        _pb = po.p_poses[lid][..., :2].float()
        _db = pr.goal.remaining(_pb.reshape(-1, 2), _tid[:, None].expand(-1, _pb.shape[1]).reshape(-1)).view(_pb.shape[:2])
        _L = pr.goal.lap[_tid][:, None]
        _ah = torch.remainder(_dc[:, None] - _db, _L)               # along the track to the box ahead [m]
        v_seen |= (v_prev < 3.0) & (_ah > _L - 3.0) & (po.p_sid[lid] >= 0)     # it was just ahead, now just behind
        v_prev = _ah
        if TARGET:
            v_zone = (v_seen & (po.p_sid[lid] >= 0) & (_ah >= TARGET[2]) & (_ah <= TARGET[3])).any(1)
            t_stat["zone"] += float(v_zone.float().sum()); t_stat["n"] += float(Bl)
    obs.push_scan(); obs.set_plan(pts, p["pv"]); obs.set_plan2(p)
    if so is not None:
        so.after()
        so.on[:] = True                                     # a restarted race keeps its copy (and pace: not drawn again)
        so.pace[:] = c_pace
        R["cfault"].append(cf_.cpu())
    if bool(restarted.any()):
        rows = torch.nonzero(restarted)[:, 0]
        obs.reset_rows(rows)
        if hasattr(follower, "reset_rows"):
            follower.reset_rows(rows)
    if remover is not None:
        remover.step()
    ors = torch.zeros(env.B, dtype=torch.bool, device=dev)
    if restarter is not None:
        pre = env.sim.state[:, :2].clone()
        if restarter.step():
            ors = (env.sim.state[:, :2] - pre).norm(dim=1) > 1e-6          # the restarted cars are the ones that moved
    R["ors"].append(ors[others].cpu())
    R["pl"].append(pts.cpu()); R["rs"].append(restarted.cpu()); R["lap"].append(out["lap"].float().cpu())
    cf = env.sim.contact_frame()
    R["cnt"].append(cf[lid].float()[:, I_CNT].cpu()); R["ocnt"].append(cf[others].float()[:, I_CNT].cpu())
    R["held"].append(cf[lid].float()[:, [i_hk, i_hx, i_hy, i_hs]].cpu())
    if kd in (9, 99) or (kd + 1) % 300 == 0:
        print(f"  {(kd + 1) * PS.DT:5.1f} s / {secs:g} s ({time.time() - t0:.0f} s)", flush=True)
snap()
D = {k: torch.stack(v).numpy() for k, v in R.items()}
trk = env.sim.track
_bank = sorted(f for f in os.listdir(BANK) if f.endswith(".pt"))
_sha = hashlib.sha256(b"".join(open(os.path.join(BANK, f), "rb").read() for f in _bank)).hexdigest()[:12]
D.update(edt=trk.edt.float().cpu().numpy(), t_off=trk.t_off.cpu().numpy(), t_H=trk.t_H.cpu().numpy(), t_W=trk.t_W.cpu().numpy(),
         t_res=trk.t_res.cpu().numpy(), t_origin=trk.t_origin.cpu().numpy(),
         car=np.array([float(env.cfg.vehicle.length), float(env.cfg.vehicle.width)]), dt=np.array(PS.DT),
         kinds=np.array(KN), cols=np.array(COLS), name=np.array(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"),
         ep_s=np.array(secs + 5.0), level=env.level_seed[lid].cpu().numpy(), lapL=pr.goal.lap.float().cpu().numpy(),
         exam=np.array(EXAM + ("+mpc" if TRACK_MPC else "") + ("+slip_" + SLIP if SLIP else "") + (f"+lead{LEAD:.2f}" if LEAD > 0 else "")), speed_lead=np.array(float(getattr(env.tracker, "tw_speed_lead", 0.0) or 0.0) if mode == "map" else LEAD), mode=np.array(mode), count=np.array(count), rm_t=np.array(rm_t), seed=np.array(SEED),
         map=np.array(map_name(WORLD)), bank=np.array(f"{BANK} ({', '.join(_bank)}; sha256 {_sha})"), opp_restart_s=np.array(OPP_RESTART_S),
         lidar_mount_x=np.array(_mx), lat_margin=np.array(float(_mg) if _mg is not None else 1.0), opp_slot=np.array(OPP_SLOT), start_rest=np.array(REST), half_lap=np.array(HALF))
if RACE4:
    D.update(R_c, ref=np.array(REF), per=np.array(PER))
for k in sorted({int(s) for s in np.unique(D["sid"]) if s >= 0}):
    sh = po.shapes[k]
    D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
D["pred_mode"] = np.array("oracle" if PRED_ORACLE else ("probe" if PRED_PROBE else ("save" if PRED_SAVE else ""))); D["wipe"] = np.array(WIPE); D["wipe_box"] = np.array(WIPE_BOX); D["sample"] = np.array(SAMPLE); D["noise_rho"] = np.array(NOISE_RHO); D["steer_std_floor"] = np.array(STEER_STD_FLOOR); D["freeze"] = np.array(FREEZE); D["corridor"] = np.array(CORRIDOR)
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez_compressed(out_npz, **D)
if SAVE_MEM:
    np.savez(out_npz[:-4] + "_mem.npz", feat=torch.stack(MEM["feat"]).numpy(), mem=torch.stack(MEM["mem"]).numpy())
if SAVE_VALUE:
    np.savez_compressed(out_npz[:-4] + "_priv.npz", priv=torch.stack(VAL["priv"]).numpy(), value=torch.stack(VAL["value"]).numpy(),
                        box_cols=np.array([_P0, _w]), box_range=np.array(_br))
if SAVE_PLAN:
    np.savez_compressed(out_npz[:-4] + "_plan.npz", **{k: torch.stack(v).numpy() for k, v in PLN.items()})
if PRED:
    _pd = {k: torch.stack(v).numpy() for k, v in PRD.items() if v}
    _pd.update(mode=np.array("oracle" if PRED_ORACLE else ("probe" if PRED_PROBE else "save")), pred_range=np.array(PRED_RANGE), seen_s=np.array(SEEN_S),
               pace_of=so.pace[O_OF].float().cpu().numpy(), pred_scale=np.array(PRED_SCALE))
    if PRED_PROBE:
        _pd.update(probe_names=np.array([n_ for n_, _, _ in PROBES]), probe_cols=np.array(PCOLS))
    np.savez_compressed(out_npz[:-4] + "_pred.npz", **_pd)
    if PRED_PROBE:
        _be = _pd["base_err"].max(0)
        print(f"[pred probe] 같은 입력으로 다시 돈 plan과 몬 plan의 차 최대: 점 {_be[0]:.2e} m, 속도 {_be[1]:.2e} m/s", flush=True)
if SAVE_SCAN:
    np.savez_compressed(out_npz[:-4] + "_scan.npz", scan=torch.stack(SCN["scan"]).numpy(), typ=torch.stack(SCN["typ"]).numpy(),
                        ego=torch.stack(SCN["ego"]).numpy(),
                        range_max=np.array(obs.range), angles=env.sim.lidar.angles.float().cpu().numpy())
if SAVE_BOXVIS:
    np.savez_compressed(out_npz[:-4] + "_boxvis.npz", ds=torch.stack(BXV["ds"]).numpy(), live=torch.stack(BXV["live"]).numpy(),
                        view=torch.stack(BXV["view"]).numpy(), mount_x=np.array(_bv_mx), rmax=np.array(_bv_rmax))
c = D["cnt"][-1].sum(0)
print(f"[{EXAM}{' · 진행 방향 틀(' + SLIP + ')' if SLIP else ''} · {mode} · 상자 {count} · 시드 {SEED}] {D['name']}: 세션 {Bl}개 x {secs:.0f} s (상자는 {rm_t:g} s에 치움), 도중 다시 출발 "
      f"{int(D['rs'].sum())}번, 선 상대 차 다시 출발 {restarter.count if restarter is not None else 0}번 | 누적: "
      + " · ".join(f"{k} {int(v)}" for k, v in zip(COLS, c)) + f" | {time.time() - t0:.0f} s | 저장 {out_npz}", flush=True)
if VIRT:
    print(f"[virtual boxes] passed boxes drawn into the policy's scan: {100.0 * v_stat['drawn'] / max(v_stat['n'], 1.0):.2f} % of beams "
          f"shortened; boxes passed at least once at the end: {int(v_seen.sum())}", flush=True)
if TARGET:
    print(f"[target noise] decisions in the zone (a passed box {TARGET[2]:g}-{TARGET[3]:g} m ahead): "
          f"{100.0 * t_stat['zone'] / max(t_stat['n'], 1.0):.1f} %", flush=True)
