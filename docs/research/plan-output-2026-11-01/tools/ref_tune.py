"""정해진 궤적을 얼마나 정확히 따라가나로 MAP 튜닝 (11-15 21:45; 사용자 10-20 "제어기는 궤적과의 거리로 맞춤", 11-15 "Map 튜닝으로는
안돼?", "당연히 제어지연까지 같이 해놔야지").

policy가 고리 안에 있으면(시험) 오차를 차가 벗어난 만큼 policy가 다시 그린 plan과 재게 되고, 천천히 달리는 설정이 오차가 작아 1등이
됨(map_tune.py 첫 목표: 평균 3.1 m/s 설정이 1등, 벽은 더 많음). 그래서 기준 궤적을 고정: 지금 policy가 이상 실행으로 실제 달린 궤적
(시험 기록 exam1_*_ideal_*.npz, 0.1 s마다 x, y, 방향, 속도) -- 빠르고, 벽·섬 끝에 붙고, 상자를 비키는 그 policy의 운전 그대로.

차 256대(상자·상대 없음, 실제 지도)를 기록의 두 구간(3-18 s, 45-60 s) 시작 자세·속도에 놓고, 0.1 s마다 지금 자세에서 기준 궤적의
0.1..1.0 s 뒤 점으로 원호 10개를 이어 plan을 만들어(그 점의 기준 속도, 따라잡기 없음 -- tw_tracking.py와 같은 방식) MAP에 넘김
(시험의 MAP과 같은 길: env.ext_plan_kv). 잣대: 차에서 기준 경로까지 거리 RMSE(1 s 뒤부터, 기준 진행 위치 둘레 ±1 s 안의 점),
뒤처짐(끝에서 기준보다 몇 s 늦나), 벽 세게 닿은 차 수. 목표 = RMSE + 벽 닿은 차 비율 x 5 cm (tw_tune.py와 같음).

설정은 한 프로세스 안에서 바꿔 끼움(env.tracker.map.p, tw_speed_lead); 처음에 지금 설정을 두 번 돌려 같은 값이 나오는지 확인.
진화 탐색은 map_tune.py와 같은 공간(지연 보상·속도 앞당김 포함).
인자: 체크포인트 이상실행기록npz 출력jsonl [세대 수 10] [난수 시드]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["F1SIM_NO_STAGGER"] = "1"
os.environ.pop("F1SIM_WORLD_TRACKS", None)
torch.set_grad_enabled(False)
ck_path, ref_npz, out_jl = sys.argv[1], sys.argv[2], sys.argv[3]
GENS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
RSEED = int(sys.argv[5]) if len(sys.argv) > 5 else 1115
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of
_mx = lidar_mount_of(ck)
if _mx > 0:
    os.environ["F1SIM_LIDAR_MOUNT_X"] = str(_mx)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim import dynamics as dyn
from f1sim import map_controller as M
from f1sim import plan_tw
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from world_of import world_path

WINDOWS = ((3.0, 18.0), (45.0, 60.0))
z = np.load(ref_npz)
st_r = z["st"].astype(np.float32)                               # (T + 1, B0, 4) every 0.1 s
if os.environ.get("REF_SMOKE"):                                 # a quick check of the harness: a few cars, 4 s
    st_r = st_r[:, :int(os.environ["REF_SMOKE"])]
    WINDOWS = ((3.0, 7.0), (45.0, 49.0))
B0 = st_r.shape[1]
N_ENV = B0 * len(WINDOWS)

a = argparse.Namespace(**json.load(open(world_path(ck))))
a.envs, a.seed, a.device = 2 * N_ENV, 51, "cuda"                 # half the rows are opponent slots (absent here)
a.episode_s = 3600.0
a.warps, a.track_scales = 0, None
a.prop_fraction, a.removal_p, a.opp_present, a.spawn_behind_prop = 0.0, 0.0, 0.0, 0.0
a.prop_bank = None
a.plan_format, a.repeat = "tw", 4
env = SR.build_env(a, dev)
prepare_graph_runtime(env, log=lambda _t: None)
lid = env.learner_ids
assert int(lid.numel()) == N_ENV, (int(lid.numel()), N_ENV)
dt = float(env.sim.control_dt); R = 4
SUBR = int(round(0.1 / dt))                                     # control steps per 0.1 s
trk = env.tracker
P0 = trk.map.p
IW = CC.index("hard_wall")

# the references, densified to the control grid: (N_ENV, Td, 4) x, y, heading, speed
# each window starts at the first moment (within 3 s of its nominal start) where the reference runs nearly straight
# (|curvature| < 0.15 1/m) above 2 m/s: the car is put there at the reference's speed with no turn going on, so the
# start transient does not dominate (a mid-corner start with zero yaw rate put 19 % of the cars into a wall in 4 s)
hd_r = np.arctan2(np.gradient(st_r[..., 1], axis=0), np.gradient(st_r[..., 0], axis=0))
ds_r = np.linalg.norm(np.gradient(st_r[..., :2], axis=0), axis=-1)
kap_r = np.abs(np.angle(np.exp(1j * np.gradient(hd_r, axis=0)))) / np.maximum(ds_r, 1e-3)
refs = []
for (t0, t1) in WINDOWS:
    n = int(round((t1 - t0) / 0.1)) + 12                       # + 1.2 s: the plan reads 1.0 s past the end
    tq = np.arange(0, (n - 1) * SUBR + 1) / SUBR
    for b in range(B0):
        i0 = int(round(t0 / 0.1))
        for i in range(i0, i0 + 30):
            if kap_r[i, b] < 0.15 and st_r[i, b, 3] > 2.0:
                i0 = i; break
        seg = st_r[i0:i0 + n, b]
        xs = np.interp(tq, np.arange(n), seg[:, 0]); ys = np.interp(tq, np.arange(n), seg[:, 1])
        vs = np.interp(tq, np.arange(n), seg[:, 3])
        hd = np.arctan2(np.gradient(ys), np.gradient(xs))
        refs.append(np.stack([xs, ys, hd, vs], -1)[None])
REF = torch.tensor(np.concatenate(refs, 0), device=dev, dtype=torch.float32)      # (N_ENV, Td, 4)
Td = REF.shape[1]
DUR = WINDOWS[0][1] - WINDOWS[0][0]
NSTEP = int(round(DUR / dt))
ar = torch.arange(N_ENV, device=dev)


def plan_kv(prog, pose):
    """(kappa, v) (N, 10): arcs from the car's pose to the reference points 0.1..1.0 s after its progress point, at the
    reference's speed there (no catching up on time lost)"""
    x, y, p = pose[:, 0].clone(), pose[:, 1].clone(), pose[:, 2].clone()
    ks, vs = [], []
    for k in range(1, 11):
        i_t = (prog + k * SUBR).clamp(max=Td - 1)
        i_m = (prog + int(round((k - 0.5) * SUBR))).clamp(max=Td - 1)
        dx, dy = REF[ar, i_t, 0] - x, REF[ar, i_t, 1] - y
        c_, s_ = torch.cos(p), torch.sin(p)
        rx, ry = c_ * dx + s_ * dy, -s_ * dx + c_ * dy
        d2 = (rx * rx + ry * ry).clamp_min(1e-9)
        kap = (2 * ry / d2).clamp(-plan_tw.KAPPA_MAX, plan_tw.KAPPA_MAX)
        v = REF[ar, i_m, 3].clamp(0.0, plan_tw.V_FWD)
        ks.append(kap); vs.append(v)
        x, y, p = plan_tw._arc(x, y, p, kap, v * 0.1)
    return torch.stack(ks, 1), torch.stack(vs, 1)


def run(cfg):
    mp = {k: v for k, v in cfg.items() if k != "lead"}
    trk.map.p = M.MAPParams.from_dict({**{f: getattr(P0, f) for f in P0.__dataclass_fields__}, **mp})
    trk.tw_speed_lead = float(cfg.get("lead", 0.0))
    env.reset(seed=51)
    s = env.sim.state.clone()
    r0 = REF[:, 0]
    s[lid, :] = 0.0
    s[lid, dyn.IX], s[lid, dyn.IY], s[lid, dyn.IYAW], s[lid, dyn.IVX] = r0[:, 0], r0[:, 1], r0[:, 2], r0[:, 3]
    s[lid, dyn.IOMEGA] = r0[:, 3] / env.sim.P["r_w"][lid]
    env.sim.state.copy_(s)
    trk.map.reset(torch.ones(env.B, dtype=torch.bool, device=dev))
    kk = torch.zeros(env.B, 10, device=dev); vv = torch.zeros(env.B, 10, device=dev)
    act = torch.zeros(env.B, env.act_dim, device=dev)
    cf0 = env.sim.contact_frame()[lid, IW].float().clone()
    prog = torch.zeros(N_ENV, dtype=torch.long, device=dev)
    win = torch.arange(int(round(0.4 / dt)), device=dev)
    d_all, hit = [], torch.zeros(N_ENV, dtype=torch.bool, device=dev)
    for k in range(NSTEP):
        if k % R == 0:
            idx = (prog[:, None] + win[None, :]).clamp(max=Td - 1)
            dist = (REF[ar[:, None], idx, :2] - env.sim.state[lid, None, :2]).norm(dim=-1)
            prog = idx.gather(1, dist.argmin(1, keepdim=True))[:, 0]
            k_, v_ = plan_kv(prog, env.sim.state[lid, :3])
            kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
        env.step(act)
        if k >= int(round(1.0 / dt)) and k % 2 == 0:
            j = (prog[:, None] + torch.arange(-40, 41, device=dev)[None, :]).clamp(0, Td - 1)
            pts = REF[ar[:, None], j, :2]
            pos = env.sim.state[lid, None, :2]
            a_, b_ = pts[:, :-1], pts[:, 1:]
            ab = b_ - a_; ap = pos - a_
            t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0, 1)
            dd = (ap - t[..., None] * ab).norm(dim=-1).min(1).values
            d_all.append(torch.where(hit, torch.full_like(dd, float("nan")), dd))     # each car until its first hard wall hit
        hit |= (env.sim.contact_frame()[lid, IW].float() - cf0) > 0
    D = torch.stack(d_all, 1)                                    # (N, n), NaN after a car's first hard wall hit
    Dv = D[~torch.isnan(D)]
    lag = (k + 1) * dt - prog.float() * dt                       # [s] behind the reference at the end
    rmse = float(Dv.square().mean().sqrt())
    return dict(rmse=rmse, p90=float(Dv.quantile(0.9)), p99=float(Dv.quantile(0.99)),
                wall_cars=float(hit.float().mean()), lag=float(lag.median()),
                obj=rmse + 0.05 * float(hit.float().mean()))


NOW = dict(eta_ref="heading", l1_point="arc", steer_speed="planned", lut_interp="code", m_l1=0.47, q_l1=0.0, t_clip_min=0.35,
           v_steer_gain_max=1.281, acc_scaler_for_steer=0.886, dec_scaler_for_steer=0.809, delay_comp=0.0214,
           speed_lookahead_for_steer=0.06, lead=0.0)
PAPER = dict(NOW, eta_ref="velocity", l1_point="euclid", steer_speed="measured", lut_interp="linear", v_steer_gain_max=1.0,
             acc_scaler_for_steer=1.0, dec_scaler_for_steer=1.0)
SHORT = {"eta_ref": "heading", "l1_point": "arc", "steer_speed": "planned", "lut_interp": "linear", "m_l1": 0.37, "q_l1": -0.49,
         "t_clip_min": 0.78, "v_steer_gain_max": 1.47, "acc_scaler_for_steer": 1.0, "dec_scaler_for_steer": 1.0,
         "delay_comp": 0.009, "speed_lookahead_for_steer": 0.133, "lead": 0.141}
CAT = {"eta_ref": ("heading", "velocity"), "l1_point": ("arc", "euclid"), "steer_speed": ("planned", "measured"),
       "lut_interp": ("code", "linear")}
NUM = {"m_l1": (0.15, 0.7), "q_l1": (-0.6, 0.4), "t_clip_min": (0.3, 1.0), "v_steer_gain_max": (0.9, 1.6),
       "acc_scaler_for_steer": (0.7, 1.3), "dec_scaler_for_steer": (0.7, 1.3), "delay_comp": (0.0, 0.12),
       "speed_lookahead_for_steer": (0.0, 0.2), "lead": (0.0, 0.2)}
rng = np.random.default_rng(RSEED)


def mutate(c, sig):
    c = dict(c)
    for k, (lo, hi) in NUM.items():
        c[k] = float(np.clip(c.get(k, 0.5 * (lo + hi)) + rng.normal() * sig * (hi - lo), lo, hi))
    for k, opts in CAT.items():
        if rng.random() < 0.15:
            c[k] = opts[1 - opts.index(c[k])]
    return c


def rand():
    c = {k: str(rng.choice(v)) for k, v in CAT.items()}
    c.update({k: float(rng.uniform(lo, hi)) for k, (lo, hi) in NUM.items()})
    return c


done = []
if os.path.exists(out_jl):
    done = [json.loads(l) for l in open(out_jl)]


def record(cfg, tag=""):
    t0 = time.time()
    r = run(cfg)
    d = dict(i=len(done), tag=tag, cfg=cfg, **r)
    done.append(d)
    with open(out_jl, "a") as f:
        f.write(json.dumps(d) + "\n")
    print(f"[{time.strftime('%H:%M')}] #{d['i']:03d} {tag:<6} 궤적과 거리 RMSE {r['rmse'] * 100:5.2f} cm (90 % {r['p90'] * 100:5.1f}, 99 % "
          f"{r['p99'] * 100:5.1f}) 벽 닿은 차 {r['wall_cars'] * 100:4.1f} % 뒤처짐 {r['lag']:+.2f} s ({time.time() - t0:.0f} s) | {json.dumps(cfg)}",
          flush=True)
    return d


if not done:
    a1 = record(NOW, "지금")
    a2 = record(NOW, "지금2")
    assert abs(a1["rmse"] - a2["rmse"]) < 1e-3 * max(1.0, a1["rmse"] * 100), ("not repeatable", a1["rmse"], a2["rmse"])
    record(PAPER, "논문")
    assert abs(done[-1]["rmse"] - a1["rmse"]) > 1e-4, "swapping MAPParams had no effect"
    record(SHORT, "짧은L1")
    for _ in range(4):
        record(rand(), "무작위")
pool = [d for d in done if d["tag"] != "지금2"]
g0 = max(0, (len(pool) - 7) // 6)
for g in range(g0, GENS):
    ranked = sorted(pool, key=lambda d: d["obj"])
    sig = 0.25 * (0.75 ** g)
    for parent, n in ((ranked[0], 4), (ranked[1], 2)):
        for _ in range(n):
            pool.append(record(mutate(parent["cfg"], sig), f"세대{g + 1}"))
ranked = sorted(pool, key=lambda d: d["obj"])
print("\n상위 5:")
for d in ranked[:5]:
    print(f"  #{d['i']:03d} 목표 {d['obj'] * 100:.2f} cm, RMSE {d['rmse'] * 100:.2f}, 벽 차 {d['wall_cars'] * 100:.1f} %, 뒤처짐 {d['lag']:+.2f} s | {json.dumps(d['cfg'])}")
