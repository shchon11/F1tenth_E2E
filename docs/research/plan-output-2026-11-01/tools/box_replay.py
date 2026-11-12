"""상자 앞에서 같은 차를 복제해 다시 달리기 (box_replay.py, 11-12).
물음: 지난 바퀴에 깨끗이 지난 상자를 이번에 세게 치는 일(box_side.py: 세게 닿은 통과는 지난 바퀴와 다른 쪽으로 간 통과, 1~2 s 전 자리는
4~5 cm 안에서 같음)은 (H1) 같은 접근에서 센서 잡음만으로 비키는 쪽이 갈리는 선택인가, (H0) 접근 순간의 상태·기억이 이미 정해 놓은 것인가.
방법: 한 세션 안에서 달리고 있는 차를 복제함. 세계를 K개씩 묶음(원본 1 + 복제 K-1). 정한 주기(기본 3 s)마다 묶음의 복제 세계들에 원본 세계의
행별 상태를 모두 덮어씀(차 상태·차마다 다른 매개변수·센서 필터 상태·접촉 상태·상자·policy 입력과 기억) -> 그 순간 원본과 복제는 같은 차·같은
세계이고, 그 뒤 달라지는 것은 행마다 따로 뽑히는 센서 잡음(LiDAR·IMU)뿐. 다음 주기까지 따로 달린 뒤 다시 덮어씀. 원본만 세션 내내 이어 달림.
  (시험 기록을 읽어 다시 차리는 방식은 버림: 가속도계 치우침·센서 장착 각·가감속 한계 같은 매개변수가 차마다 무작위인데 기록에는 없음.)
복제 갈래: A = 그대로(잡음만 다름) / B = 기억을 같은 차가 지난 바퀴 같은 자리에서 갖고 있던 것으로 바꿈(3 s 넘게 전, 0.5 m·0.6 rad 안;
  참 위치는 이 분석에서만 씀) / C = 기억을 0으로.
세계: 시험(exam1)과 같음 - 실제 Korea 2025 지도, 학습에 안 쓰인 상자 은행, plan대로 정확히 실행, policy 평균 행동; 상대 차 없음, 상자는 안 치움.
덮어쓸 상태 고르기: 환경에서 닿는 텐서 가운데 첫 차원이 차 수(B)인 것을 모두 훑어, 주행 중에 바뀌었거나 재시작에서 바뀐 것 + 접촉이 있어야만
  바뀌는 것(FORCE) - 행 번호나 역할을 담은 것(NEVER). 목록은 로그와 기록(copied)에 남김.
자기 점검 (BOX_REPLAY_SELFTEST=1): 센서 잡음 매개변수를 모두 0으로 두면 복제는 원본과 끝까지 같은 길을 가야 함. 주기마다 덮어쓰기 직전에
  원본과의 차이를 냄 (차이가 있으면 덮어쓰지 않은 상태가 남았다는 뜻). 끝에 통과/실패와 종료 코드.
(11-13) 복제 갈래 L / R = 복제 직후 처음 몇 결정 동안만 조향 명령에 왼쪽(+) / 오른쪽(-)으로 몇 도를 더함(그 뒤는 policy 그대로).
  물음: 상자에 세게 닿는 접근에서 policy는 왼쪽·오른쪽을 정하지 못해 가운데로 가는가(그러면 조금 밀어 준 쪽으로 계속 가서 피함),
  상자를 늦게 알아채는가(그러면 밀어 줘도 원래 길로 돌아가 똑같이 닿음).
인자: checkpoint 출력npz [묶음 수 21] [초 90] [주기 초 3.0] [상자 개수 max2|uniform]
환경 변수: F1SIM_EXAM_DEVICE(cuda|cpu), F1SIM_EXAM_SEED, BOX_REPLAY_ARMS="3,1,1"(A, B, C 복제 수, 다섯 개면 + L, R),
  BOX_REPLAY_NUDGE="2.0,3"(L/R: 조향에 더하는 각 [deg], 복제 뒤 결정 수)"""
import os, sys, json, argparse, time
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
os.environ.pop("F1SIM_WORLD_TRACKS", None)
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")

BANK = "/home/shchon11/f1sim_runs/prop_banks_exam1"
SEED = int(os.environ.get("F1SIM_EXAM_SEED", 20261112))
SELFTEST = os.environ.get("BOX_REPLAY_SELFTEST", "") == "1"
_arms = [int(x) for x in os.environ.get("BOX_REPLAY_ARMS", "3,1,1").split(",")] + [0, 0]
NA, NB, NC, NL, NR = _arms[:5]
K = 1 + NA + NB + NC + NL + NR
_nd = os.environ.get("BOX_REPLAY_NUDGE", "2.0,3").split(",")
NUDGE_RAD, NUDGE_N = float(_nd[0]) * np.pi / 180.0, int(_nd[1])
DET_N = 20                                # decisions driven to find what a step changes
B_BACK, B_POS, B_YAW, B_LAP = 30, 0.5, 0.6, 15   # arm B: >= 3 s back, within 0.5 m / 0.6 rad, the latest such visit

torch.set_grad_enabled(False)
ck_path, out_npz = sys.argv[1], sys.argv[2]
G = int(sys.argv[3]) if len(sys.argv) > 3 else 21
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
period_s = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
count = sys.argv[6] if len(sys.argv) > 6 else "max2"
dev = torch.device(os.environ.get("F1SIM_EXAM_DEVICE", "cuda"))
ck = torch.load(ck_path, map_location=dev, weights_only=False)
assert ck.get("format") == "plan_steps2", ck.get("format")
# the checkpoint's own scanner place and tyre-limit margin, set before anything reads them (exam_record.py's)
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
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, plan_world, PT_IDX
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import map_controller as M
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import props as _props

a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = 2 * G * K, SEED, dev.type
a.episode_s = secs + 5.0                                   # one episode per car: no respawn inside the session
a.warps, a.track_scales = 0, None                          # the real map only
a.prop_bank, a.prop_count = BANK, count
a.opp_present, a.prop_fraction = 0.0, 1.0                  # alone, every world with boxes
torch.manual_seed(SEED)
env = SR.build_env(a, dev)
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None)
actor = load_actor2(ck, dev)
assert getattr(actor, "memory", 0), "이 checkpoint에는 기억이 없습니다"
B = int(env.B)
lid = env.learner_ids; Bl = int(lid.numel())
assert Bl == G * K, (Bl, G, K)
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
po = env.procedural
ii = PT_IDX.to(dev)
COLS = ("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")
I_CNT = [CC.index(k) for k in COLS]
i_hk, i_hx, i_hy, i_hs = (CC.index(k) for k in ("held_kind", "held_x", "held_y", "held_severity"))
act0 = torch.zeros(B, env.act_dim, device=dev)
kL = torch.arange(Bl, device=dev)
grp, mem_j = kL // K, kL % K
arm = torch.where(mem_j == 0, 0, torch.where(mem_j <= NA, 1, torch.where(mem_j <= NA + NB, 2, torch.where(
    mem_j <= NA + NB + NC, 3, torch.where(mem_j <= NA + NB + NC + NL, 4, 5)))))
nudge_dir = torch.where(arm == 4, 1.0, torch.where(arm == 5, -1.0, 0.0))     # L: + (left), R: -
dk = torch.nonzero(mem_j != 0)[:, 0]; sk = grp[dk] * K     # replicate rows and their sources (into the learner rows)
src = torch.arange(G, device=dev) * K
if SELFTEST:
    arm = torch.where(mem_j == 0, 0, 1)                    # every replicate untouched

# ---- which per-car tensors make up a car's row ----
FORCE = set("""sim.collided sim.contact_counts sim.contact_held sim.contact_now sim.contact_pt_sev sim.impact_kind sim.kind_prev
sim.prop_touched sim.wall_engaged sim.wall_impact sim.wall_touched sim.car_collision sim.car_touched sim.car_impact_roll sim.ideal_vcap
sim.track.env_props.p_n sim.track.env_props.p_vel sim.track.env_props.p_zlo sim.track.env_props.last_kind sim.track.env_props.last_live
sim.track.env_props.last_s _was_touching _was_touching_wall _wall_pending _car_prev _car_resp _car_sev_ok _car_sev_prev ep_contacts ep_limit
last_result.collision last_result.contact_impacts last_result.contact_onsets last_result.wall_collision last_result.wall_impact
last_result.car_collision last_result.car_impact s_prev sector_idx sector_t0 sector_valid tracker._inner.backout_t tracker._inner.contact
tracker._inner.wedged_t tracker.map.i_gap lead_paid lead_steps _last_feat""".split())
NEVER_LEAF = set("""eid other_idx ideal_rows tid absent learner learner_ids race slot spawn_order_race teacher_race level_seed ext_mask
external_driven on_policy pool_driven teacher_driven hold_rows map_rows opp_driver slot_cap slot_cap_set slot_scale_hi slot_scale_lo
spawn_abreast _slot_mixed_rows""".split())
NEVER_PART = ("_alt_rows.", "alt_teachers.", "alt_teacher_mask.", "sim.params.P.")     # P's entries are rows of sim.params.buf


def pname(path):
    return ".".join(str(k) for _, k in path).replace("procedural.", "sim.track.env_props.")


def walk(root):
    """{path: axis} of every tensor reachable from `root` through f1sim objects, lists and dicts whose axis 0 (or 1) is the cars"""
    out = {}; seen = set()

    def rec(o, path, depth):
        if id(o) in seen:
            return
        seen.add(id(o))
        if torch.is_tensor(o):
            if o.ndim >= 1 and o.shape[0] == B and o.numel() > 0:
                out[path] = 0
            elif o.ndim >= 2 and o.shape[1] == B:
                out[path] = 1
            return
        if depth > 5 or isinstance(o, (torch.nn.Module, str, bytes, int, float, bool, type(None), np.ndarray)):
            return
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, path + (("k", k),), depth + 1)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                rec(v, path + (("i", i),), depth + 1)
        elif type(o).__module__.startswith("f1sim"):
            names = list(vars(o).keys()) if hasattr(o, "__dict__") else []
            names += [s for s in getattr(type(o), "__slots__", ()) if hasattr(o, s)]
            for k in names:
                rec(getattr(o, k), path + (("a", k),), depth + 1)
    rec(root, (), 0)
    return out


def resolve(root, path):
    o = root
    try:
        for kind, k in path:
            o = getattr(o, k) if kind == "a" else o[k]
    except (AttributeError, KeyError, IndexError, TypeError):
        return None
    return o if torch.is_tensor(o) else None


def shot():
    return {p: (resolve(env, p).clone(), ax) for p, ax in walk(env).items()}


def differs(x, y):
    if x.shape != y.shape or x.dtype != y.dtype:
        return True
    if x.is_floating_point():
        return bool(((x != y) & ~(x.isnan() & y.isnan())).any())
    return bool((x != y).any())


def decide(nudge=None):
    """one decision of every learner row, as exam_record.py's ideal mode -> (plan points, restarted). `nudge` (Bl,) [rad] is
    added to the plan's steering commands (and so to the commands the next decision reads as its last ones)"""
    st = env.sim.state[lid].clone()
    _, _, _, _, p = plan_v2(actor, obs, lim)
    if nudge is not None:
        p["cmd_d"] = p["cmd_d"] + nudge[:, None].to(p["cmd_d"].dtype)
    x, y, psi = plan_world(p, st)
    pts = torch.stack([x[:, ii], y[:, ii]], -1)
    env.sim.ideal_vcap[lid] = 1e3
    restarted = torch.zeros(Bl, dtype=torch.bool, device=dev)
    for j in range(PS.SUB):
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        _, _, te, trc, _ = env.step(act0)
        restarted |= (te | trc)[lid]
    obs.push_scan(); obs.set_plan(pts, p["pv"]); obs.set_plan2(p)
    if bool(restarted.any()):
        obs.reset_rows(torch.nonzero(restarted)[:, 0])
    return pts, restarted


QUIET = []


def quiet():
    """SELFTEST: every random sensor term off (in place: a restart draws them again)"""
    P = env.sim.params.P
    for n in P:
        if n in ("dropout_value", "road_tilt_v"):
            continue
        if (any(s in n for s in ("noise", "jitter", "dropout", "spike", "bias_walk", "shock_rate"))
                or n.startswith(("vib_accel", "vib_gyro")) or n == "road_tilt"):
            env.sim.params.set_all(n, 0.0)
            if n not in QUIET:
                QUIET.append(n)
    for n in ("car_porosity", "car_rear_porosity"):
        t = getattr(env.sim, n, None)
        if torch.is_tensor(t):
            t.zero_()


t_build = time.time()
env.reset(seed=SEED + 100003)
obs = PlanObs2(env, lim, None)
S0 = shot()
for _ in range(DET_N):
    decide()
S1 = shot()
env.reset(seed=SEED)
obs.reset_rows(kL)
S2 = shot()
assert bool(env.sim.absent[others].all()) and not bool(env.sim.absent[lid].any()), "상대 차가 있는 세계가 섞였습니다"
DYN, SKIP = [], []
for p in sorted(set(S0) | set(S1) | set(S2), key=pname):
    n = pname(p); leaf = n.split(".")[-1]
    ax = (S2.get(p) or S1.get(p) or S0.get(p))[1]
    never = leaf in NEVER_LEAF or any(s in n for s in NEVER_PART)
    late = p not in S0
    ch = ((p in S0 and p in S1 and differs(S0[p][0], S1[p][0])) or (p in S1 and p in S2 and differs(S1[p][0], S2[p][0])))
    t = resolve(env, p)
    take = (ch or n in FORCE or (late and t is not None and t.dtype != torch.long)) and not never
    if ch and never and "sim.params.P." not in n:              # P's entries go with sim.params.buf
        print(f"  주의: 바뀌는데 덮어쓰지 않음(행 번호·역할): {n}", flush=True)
    (DYN if take else SKIP).append((p, ax))
missing = sorted(FORCE - {pname(p) for p, _ in DYN})
del S0, S1, S2
print(f"B {B} · 학습 차 {Bl} = 묶음 {G} x (원본 1 + A {NA} + B {NB} + C {NC} + L {NL} + R {NR}, L/R {NUDGE_RAD * 180 / np.pi:g} deg x {NUDGE_N}결정) · 덮어쓸 텐서 {len(DYN)}개, 안 덮어쓰는 것 {len(SKIP)}개 "
      f"({time.time() - t_build:.0f} s)", flush=True)
print("  덮어씀: " + " ".join(pname(p) + ("[1]" if ax else "") for p, ax in DYN), flush=True)
print("  안 덮어씀: " + " ".join(pname(p) for p, _ in SKIP), flush=True)
if missing:
    print("  FORCE에 있는데 못 찾음: " + " ".join(missing), flush=True)
OBS_T = ("scans", "plan_xy", "plan_v", "rel", "rel_v", "cmd_d", "cmd_v", "mem_h")
row_s = torch.cat([lid[sk], others[sk]]); row_d = torch.cat([lid[dk], others[dk]])


def clone_rows():
    """every replicate world becomes its group's source world"""
    for p, ax in DYN:
        t = resolve(env, p)
        if t is None:
            continue
        if ax == 0:
            t[row_d] = t[row_s]
        else:
            t[:, row_d] = t[:, row_s]
    for n in OBS_T:
        t = getattr(obs, n, None)
        if t is not None:
            t[dk] = t[sk]


# ---- the session ----
T = int(round(secs / PS.DT)); PER = int(round(period_s / PS.DT))
R = {k: [] for k in ("st", "pl", "rs", "cnt", "inc", "held", "sid", "pose", "tid")}
SY = {k: [] for k in ("t", "div", "bvalid", "btau")}
H_xy = torch.zeros(T + 1, G, 2, device=dev); H_yaw = torch.zeros(T + 1, G, device=dev)
H_mem = torch.zeros(T + 1, G, int(actor.memory), device=dev)
last_rs = torch.full((G,), -1, dtype=torch.long, device=dev)          # the source's last restart (decision index)
win_rs = torch.zeros(Bl, dtype=torch.bool, device=dev)                # restarted since the last overwrite
worst = {"pos": 0.0, "yaw": 0.0, "v": 0.0, "mem": 0.0, "cnt": 0.0, "bad": 0, "n": 0}
prof = []                                                             # SELFTEST: (largest gap, rows with any gap) per decision since the overwrite


def snap():
    R["st"].append(env.sim.state[lid, :4].float().cpu())
    R["sid"].append(po.p_sid[eid].to(torch.int16).cpu()); R["pose"].append(po.p_poses[eid].float().cpu())
    R["tid"].append(env.sim.tid[lid].to(torch.int16).cpu())


def counts():
    return env.sim.contact_frame()[lid].float()[:, I_CNT]


if SELFTEST:
    quiet()
    print("  자기 점검: 0으로 둔 잡음 매개변수 " + " ".join(QUIET), flush=True)
t0 = time.time()
for kd in range(T):
    if kd % PER == 0:
        s_all = env.sim.state[lid]
        ok = ~(win_rs[dk] | win_rs[sk])
        div = torch.full((Bl,), float("nan"), device=dev)
        if kd > 0:
            d = s_all[dk] - s_all[sk]
            dpos = d[:, :2].norm(dim=1)
            div[dk] = torch.where(ok, dpos, torch.full_like(dpos, float("nan")))
            if bool(ok.any()):
                dyaw = ((d[:, 2] + np.pi) % (2 * np.pi) - np.pi).abs()
                dmem = (obs.mem_h[dk] - obs.mem_h[sk]).abs().amax(1)
                dcnt = (counts()[dk] - counts()[sk]).abs().amax(1)
                w = dict(pos=float(dpos[ok].max()), yaw=float(dyaw[ok].max()), v=float(d[:, 3][ok].abs().max()),
                         mem=float(dmem[ok].max()), cnt=float(dcnt[ok].max()), bad=int((dpos[ok] > 1e-3).sum()), n=int(ok.sum()))
                for k_ in ("pos", "yaw", "v", "mem", "cnt"):
                    worst[k_] = max(worst[k_], w[k_])
                worst["bad"] += w["bad"]; worst["n"] += w["n"]
                if SELFTEST or kd % (10 * PER) == 0:
                    print(f"  {kd * PS.DT:5.1f} s 덮어쓰기 전 복제-원본 차이(복제 {w['n']}대): 자리 최대 {w['pos']:.2e} m (중앙 {float(dpos[ok].median()):.2e}), "
                          f"방향 {w['yaw']:.2e} rad, 속도 {w['v']:.2e} m/s, 기억 {w['mem']:.2e}, 접촉 수 {w['cnt']:.0f}"
                          + (f", 1 mm 넘는 복제 {w['bad']}대" if SELFTEST else ""), flush=True)
                if SELFTEST and prof:
                    print("         덮어쓴 뒤 결정 수별 자리 차이 최대 [m] (차이 있는 복제 수): "
                          + " · ".join(f"{j + 1}: {prof[j][0]:.1e} ({prof[j][1]})" for j in (0, 1, 4, 9, 19, len(prof) - 1) if j < len(prof)), flush=True)
        clone_rows()
        win_rs[:] = False; prof = []
        bvalid = torch.zeros(G, dtype=torch.bool, device=dev); btau = torch.zeros(G, dtype=torch.long, device=dev)
        if not SELFTEST and getattr(obs, "mem_h", None) is not None:
            obs.mem_h[arm == 3] = 0.0
            n_c = kd - B_BACK + 1                                       # candidates: decisions 0 .. kd - B_BACK
            if NB > 0 and n_c > 0:
                xy_n, yaw_n = s_all[src, :2], s_all[src, 2]
                dd = (H_xy[:n_c] - xy_n[None]).norm(dim=2)
                dy = ((H_yaw[:n_c] - yaw_n[None] + np.pi) % (2 * np.pi) - np.pi).abs()
                idx = torch.arange(n_c, device=dev)[:, None]
                m = (dd < B_POS) & (dy < B_YAW) & (idx > last_rs[None])
                latest = torch.where(m, idx, torch.full_like(idx, -1)).amax(0)
                m &= idx >= (latest[None] - B_LAP)
                tau = torch.where(m, dd, torch.full_like(dd, float("inf"))).argmin(0)
                bvalid = m.any(0); btau = kd - tau
                rows_b = torch.nonzero((arm == 2) & bvalid[grp])[:, 0]
                obs.mem_h[rows_b] = H_mem[tau[grp[rows_b]], grp[rows_b]]
        SY["t"].append(kd); SY["div"].append(div.cpu()); SY["bvalid"].append(bvalid.cpu()); SY["btau"].append(btau.cpu())
    if SELFTEST:
        quiet()
    s_src = env.sim.state[lid[src]]
    H_xy[kd] = s_src[:, :2]; H_yaw[kd] = s_src[:, 2]
    if getattr(obs, "mem_h", None) is not None:
        H_mem[kd] = obs.mem_h[src]
    snap()
    c0 = counts()
    pts, restarted = decide(nudge_dir * NUDGE_RAD if (NL + NR > 0 and not SELFTEST and kd % PER < NUDGE_N) else None)
    c1 = counts()
    win_rs |= restarted
    if SELFTEST:
        s_now = env.sim.state[lid]
        dp = (s_now[dk, :2] - s_now[sk, :2]).norm(dim=1)[~(win_rs[dk] | win_rs[sk])]
        prof.append((float(dp.max()), int((dp > 0).sum())) if dp.numel() else (0.0, 0))
    last_rs[restarted[src]] = kd
    R["pl"].append(pts.cpu()); R["rs"].append(restarted.cpu())
    R["cnt"].append(c1.cpu()); R["inc"].append(torch.where(restarted[:, None], torch.zeros_like(c1), (c1 - c0).clamp_min(0)).cpu())
    cf = env.sim.contact_frame()
    R["held"].append(cf[lid].float()[:, [i_hk, i_hx, i_hy, i_hs]].cpu())
    if kd in (9, 99) or (kd + 1) % 300 == 0:
        print(f"  {(kd + 1) * PS.DT:5.1f} s / {secs:g} s ({time.time() - t0:.0f} s)", flush=True)
snap()
D = {k: torch.stack(v).numpy() for k, v in R.items()}
trk = env.sim.track
D.update(edt=trk.edt.float().cpu().numpy(), t_off=trk.t_off.cpu().numpy(), t_H=trk.t_H.cpu().numpy(), t_W=trk.t_W.cpu().numpy(),
         t_res=trk.t_res.cpu().numpy(), t_origin=trk.t_origin.cpu().numpy(),
         car=np.array([float(env.cfg.vehicle.length), float(env.cfg.vehicle.width)]), dt=np.array(PS.DT), cols=np.array(COLS),
         name=np.array(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"),
         grp=grp.cpu().numpy(), arm=arm.to(torch.int8).cpu().numpy(), period=np.array(PER),
         nudge_deg=np.array(NUDGE_RAD * 180.0 / np.pi), nudge_n=np.array(NUDGE_N), sync_t=np.array(SY["t"]),
         div=torch.stack(SY["div"]).numpy(), bvalid=torch.stack(SY["bvalid"]).numpy(), btau=torch.stack(SY["btau"]).numpy(),
         copied=np.array([pname(p) for p, _ in DYN]), selftest=np.array(SELFTEST), seed=np.array(SEED), count=np.array(count),
         bank=np.array(BANK), lidar_mount_x=np.array(_mx), lat_margin=np.array(float(_mg) if _mg is not None else 1.0))
for k in sorted({int(s) for s in np.unique(D["sid"]) if s >= 0}):
    sh = po.shapes[k]
    D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez_compressed(out_npz, **D)
c = D["inc"].sum((0, 1))
print(f"[복제 주행 · 상자 {count} · 시드 {SEED}{' · 자기 점검(잡음 0)' if SELFTEST else ''}] {D['name']}: 묶음 {G}개 x {secs:.0f} s, 주기 {period_s:g} s, "
      f"도중 다시 출발 {int(D['rs'].sum())}번 | 접촉 합: " + " · ".join(f"{k} {int(v)}" for k, v in zip(COLS, c))
      + f" | 복제-원본 자리 차이 최대 {worst['pos']:.2e} m | {time.time() - t0:.0f} s | 저장 {out_npz}", flush=True)
if SELFTEST:
    good = worst["n"] > 0 and worst["bad"] == 0 and worst["cnt"] == 0
    print(("자기 점검 통과" if good else "자기 점검 실패") + f": 복제 {worst['n']}대-주기 가운데 1 mm 넘게 벌어진 것 {worst['bad']}, "
          f"최대 자리 {worst['pos']:.2e} m · 방향 {worst['yaw']:.2e} rad · 속도 {worst['v']:.2e} m/s · 기억 {worst['mem']:.2e} · 접촉 수 차 {worst['cnt']:.0f}", flush=True)
    sys.exit(0 if good else 1)
