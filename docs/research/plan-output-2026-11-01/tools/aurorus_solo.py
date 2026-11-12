"""AuRoRus 회피 주행 포팅본(f1sim.learn.aurorus_opponent)을 혼자 달리게 한 기록 (11-12): 학습에 상대 차로 넣기 전에 우리 policy와
같은 세계·같은 시드에서 견주려고 (drive_record.py에 같은 덮어쓰기를 주면 우리 policy의 같은 기록).
학습 세계(plan_rl.build)의 학습 차 자리에 포팅본을 앉힘: 자기 LiDAR 스캔만 읽고(상자의 참 위치는 안 씀; 벽은 지도), 조향·속도 명령은
학습 조건대로 정확히 실행(`env.ideal_cmd`). 상대 차는 없음(opp_present 0).
환경 변수: F1SIM_AUR_DRIVE=physics 조향·속도 명령을 시뮬레이터의 차(타이어·명령 지연·서보·모터)에 그대로 줌 (기본 exact = 정확 실행,
  타이어 한계 없음) · F1SIM_AUR_FLOOR=gate 자기 자이로·속도로 짐작한 차체 기울기로 바닥에 닿을 거리의 스캔 점을 버림 /
  oracle 시뮬레이터가 바닥이라고 아는 점을 버림(차는 가질 수 없는 정답, gate의 기준) / off 원본 그대로(기본) ·
  F1SIM_AUR_FLOOR_TOL gate의 여유 [rad] · F1SIM_AUR_CFG='{"follower": "map"}' 고른 경로를 MAP 제어기가 따라감(기본 = 원본의 겨눈 점 조향);
  F1SIM_AUR_MAP = MAP 설정 덮어쓰기 JSON (기본 = 이 지도의 튜닝값, baseline_settings의 raceline solo)
기록은 drive_record.py와 같은 npz 틀(st = 포팅본 차, pl = 차 자리 그대로) + 포팅본의 내부 값:
  pscale 선 속도의 몇 배로 달리나 · avoid 회피 중 · lost 선에서 1.5 m 넘게 벗어남 · nob 장애물로 본 점 수 · gain 속도 줄임 배율
  reach 멈춰야 할 때 남은 거리(inf = 안 멈춰도 됨) · tgt 겨눈 점 · stuck 서 있어서 다시 출발시킨 결정
  obs 장애물로 본 스캔 점(그 밖은 NaN) · path 고른 경로의 점(회피 중이 아니면 NaN) · pfree 벽·장애물에 안 닿는 경로가 있었나
  (obs / path / tgt / nob / gain / reach는 결정의 마지막 제어 주기 것)
  att 차체의 참 roll·pitch · tilt 포팅본이 짐작한 roll·pitch(gate일 때) · nfl gate가 바닥이라고 버린 점 수 ·
  nobf 장애물로 본 점 가운데 실제로는 바닥인 것 · yr 요 각속도 · vy 옆으로 미끄러지는 속도
  xt (T, 차, 4, 2) 제어 주기마다 [0.2 s 전, 0.5 s 전]에 받은 경로에서 지금 차까지의 옆 거리 [m] (그 사이 다시 출발했으면 NaN) ·
  xav (T, 차, 4, 2) 그 경로를 받을 때 회피 중이었나 (추종 오차는 경로와의 옆 거리로만 잼; 속도·접촉으로 재지 않음)
속도 배율은 차마다 고정(목록을 차 순서대로 돌아가며). 원본은 후진이 없어 막힌 차는 그대로 서 있음 -> 학습 세계가 상대 차에 하듯
(state_rl.OppRestart) stuck_s 동안 0.5 m도 못 가면 다시 출발시키고 그 횟수를 남김 (0 = 끔).
인자: 출력npz [환경 수, 기본 384] [초, 기본 40] [세계 설정 덮어쓰기 JSON] [속도 배율 목록 JSON, 기본 [0.6, 0.8, 1.0]] [stuck_s, 기본 2]"""
import os, sys, json, time
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR
from f1sim.learn import aurorus_opponent as AO
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import props as _props
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
out_npz = sys.argv[1]
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 384
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
OV = {"episode_s": secs + 5.0, "opp_present": 0.0}
if len(sys.argv) > 4:
    OV.update(json.loads(sys.argv[4]))
SCALES = [float(v) for v in json.loads(sys.argv[5])] if len(sys.argv) > 5 else [0.6, 0.8, 1.0]
stuck_s = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
SEED = 4242
SEED = int(os.environ.get("F1SIM_AUR_SEED", SEED))
# the port's own settings (`AurorusConfig` fields, JSON) and switches for checks: F1SIM_AUR_DESKEW=0 places every beam from
# the cycle's pose as the original does; F1SIM_AUR_SIM_NO_SWEEP=1 makes the simulator take a whole scan at one instant
AUR_CFG = json.loads(os.environ.get("F1SIM_AUR_CFG", "{}"))
DESKEW = os.environ.get("F1SIM_AUR_DESKEW", "1") != "0"
DRIVE = os.environ.get("F1SIM_AUR_DRIVE", "exact")
FLOOR = os.environ.get("F1SIM_AUR_FLOOR", "off")
FLOOR_TOL = float(os.environ.get("F1SIM_AUR_FLOOR_TOL", "0.03"))
MAP_OV = json.loads(os.environ.get("F1SIM_AUR_MAP", "{}"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env, lim, world = PR.build(PR.WORLD, n_env, SEED, dev, fmt="v2", overrides=OV)
if os.environ.get("F1SIM_AUR_SIM_NO_SWEEP", "0") == "1":
    env.cfg.lidar.motion_distortion = False
    assert env.sim.cfg.lidar.motion_distortion is False
lid = env.learner_ids; Bl = int(lid.numel())
if DRIVE == "physics":                                             # the learner's rows are driven exactly while this is set
    # in place, not `= None`: on a GPU the physics is a graph captured in `build` with exact execution on, and it keeps
    # reading this tensor (found 11-12: with `= None` the 128-car GPU records were exact execution under a physics label)
    env.ideal_cmd = None; env.sim.ideal_rows.zero_()
MAP_SET = None
if AUR_CFG.get("follower") == "map":
    MAP_SET = {**(AO.map_params_for(env) or {}), **MAP_OV}
opp = AO.AurorusOpponents(env, share=1.0, scale=(min(SCALES), max(SCALES)), seed=SEED, rows=lid, cfg=AO.AurorusConfig(**AUR_CFG),
                          deskew=DESKEW, drive=DRIVE, floor=FLOOR, floor_tol=FLOOR_TOL, map_params=MAP_SET)
opp.core.keep_way = True
from f1sim.learn.tracking_meter import path_distance
LAGS = (8, 20)                                                     # control cycles: 0.2 s, 0.5 s
NW = int(opp.core.wu.numel())
ring = torch.full((max(LAGS), Bl, NW, 2), float("nan"), device=dev)    # the path each car was handed, the last 0.5 s of cycles
ring_av = torch.zeros(max(LAGS), Bl, dtype=torch.bool, device=dev)
ring_i = 0
fixed = torch.tensor(SCALES, device=dev, dtype=opp.scale.dtype)[torch.arange(Bl, device=dev) % len(SCALES)]
opp.scale[:] = fixed
opp.core.chunk = 1 << 20                                           # one chunk: `last_plan` then holds every car that planned
NB, KP = int(opp.core.beams.numel()), int(opp.core.K)
pr = PlanReward(env, 0.995, dev)                                   # for the line crossings (laps) only
po = env.procedural
rm_p = float(getattr(world, "removal_p", 0.0))
remover = SR.PropRemoval(env, rm_p, *[float(v) for v in getattr(world, "removal_s", (10.0, 30.0))], seed=SEED) if rm_p > 0 else None
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
COLS = ("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")
I_CNT = [CC.index(k) for k in COLS]
i_hk, i_hx, i_hy, i_hs = (CC.index(k) for k in ("held_kind", "held_x", "held_y", "held_severity"))
KN = list(getattr(env, "alt_teacher_kinds", ())) + ["raceline"]
T = int(secs / PS.DT)
R = {k: [] for k in ("st", "op", "on", "pl", "rs", "cnt", "ocnt", "held", "sid", "pose", "tid", "kind", "scale", "lap", "ors", "newp",
                     "copy", "cap", "pscale", "avoid", "lost", "nob", "gain", "reach", "tgt", "stuck", "obs", "path", "pfree", "off", "kap", "vref",
                     "att", "tilt", "nfl", "nobf", "yr", "vy", "xt", "xav")}
act0 = torch.zeros(env.B, env.act_dim, device=dev)
K_ST = max(int(round(stuck_s / PS.DT)), 1)
hist = torch.zeros(K_ST, Bl, 2, device=dev)                        # where each car was at the last K_ST decisions (a ring)
age = torch.zeros(Bl, dtype=torch.long, device=dev)                # decisions in a row it has been driving
h_i, n_stuck = 0, 0
t_port = 0.0


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.time()



def snap():
    """the world as the decision about to be made sees it (drive_record.py's layout)"""
    R["st"].append(env.sim.state[lid, :4].float().cpu()); R["op"].append(env.sim.state[others, :4].float().cpu())
    R["on"].append((~absent[others]).cpu())
    if po is None:                                                 # a world without boxes (prop_fraction 0)
        R["sid"].append(torch.full((Bl, 1), -1, dtype=torch.int16)); R["pose"].append(torch.zeros(Bl, 1, 3))
    else:
        R["sid"].append(po.p_sid[eid].to(torch.int16).cpu()); R["pose"].append(po.p_poses[eid].float().cpu())
    R["tid"].append(env.sim.tid[lid].to(torch.int16).cpu())
    R["kind"].append(torch.full((Bl,), len(KN) - 1, dtype=torch.int8)); R["scale"].append(torch.ones(Bl))
    R["copy"].append(torch.full((Bl,), -1, dtype=torch.int8)); R["cap"].append(torch.full((Bl,), 1e3))
    R["pscale"].append(opp.scale.float().cpu())
    R["yr"].append(env.sim.state[lid, 5].float().cpu()); R["vy"].append(env.sim.state[lid, 4].float().cpu())


t0 = time.time()
for kd in range(T):
    snap()
    t1 = sync()
    opp.plan()
    t_port += sync() - t1
    restarted = torch.zeros(Bl, dtype=torch.bool, device=dev)
    xt_c = torch.full((PS.SUB, Bl, len(LAGS)), float("nan"), device=dev); xav_c = torch.zeros(PS.SUB, Bl, len(LAGS), dtype=torch.bool, device=dev)
    for j in range(PS.SUB):
        t1 = sync()
        opp.command(j)
        t_port += sync() - t1
        if j == PS.SUB - 1:                                            # what the last cycle's scan really hit
            typ = env.last_result.scan_type[lid][:, opp.core.beams]
        # the car now against the paths it was handed 0.2 s and 0.5 s ago
        now = env.sim.state[lid, :2].float()
        for q, lag in enumerate(LAGS):
            k_old = (ring_i - lag) % max(LAGS)
            old = ring[k_old]
            okp = ~torch.isnan(old[:, 0, 0])
            dxt = torch.full((Bl,), float("nan"), device=dev)
            if bool(okp.any()):
                dxt[okp] = path_distance(old[okp], now[okp])
            xt_c[j, :, q] = dxt; xav_c[j, :, q] = ring_av[k_old]
        Lc = opp.core.last
        ring[ring_i] = float("nan"); ring_av[ring_i] = False
        if Lc and Lc.get("way") is not None:
            ring[ring_i, Lc["idx"]] = Lc["way"].float(); ring_av[ring_i, Lc["idx"]] = opp.core.avoiding[Lc["idx"]]
        ring_i = (ring_i + 1) % max(LAGS)
        _, _, te, trc, _ = env.step(act0)
        done = te | trc
        restarted |= done[lid]
        if bool(done[lid].any()):                                      # a new start: the old paths are of the old place
            ring[:, done[lid]] = float("nan")
        opp.stepped(done)
        pr.control_step()
    # what the controller saw and chose in the decision's last cycle (cars it did not drive then: blank)
    L = opp.core.last
    tgt = torch.full((Bl, 2), float("nan"), device=dev); nob = torch.zeros(Bl, dtype=torch.int16, device=dev)
    gain = torch.ones(Bl, device=dev); reach = torch.full((Bl,), float("inf"), device=dev)
    if L:
        i = L["idx"]
        tgt[i] = L["target"].float(); nob[i] = L["n_obst"].to(torch.int16); gain[i] = L["gain"].float(); reach[i] = L["reach"].float()
    obs_p = torch.full((Bl, NB, 2), float("nan"), device=dev); path = torch.full((Bl, KP, 2), float("nan"), device=dev)
    pfree = torch.ones(Bl, dtype=torch.bool, device=dev)
    if L:
        obs_p[i] = torch.where(L["ok"][..., None], L["pts"].float(), torch.full_like(L["pts"].float(), float("nan")))
        pj = L["plan_rows"]
        if pj.numel():
            LP = opp.core.last_plan
            rr = torch.arange(int(pj.numel()), device=dev)
            Pc = LP["P"][rr, LP["chosen"]].float()                             # (R, K, 2) the chosen path
            n_c = LP["points"][LP["chosen"]]                                   # its number of points
            Pc = torch.where((torch.arange(KP, device=dev)[None] < n_c[:, None])[..., None], Pc, torch.full_like(Pc, float("nan")))
            path[i[pj]] = Pc; pfree[i[pj]] = LP["free"]
    R["obs"].append(obs_p.to(torch.float16).cpu()); R["path"].append(path.to(torch.float16).cpu()); R["pfree"].append(pfree.cpu())
    R["tgt"].append(tgt.cpu()); R["nob"].append(nob.cpu()); R["gain"].append(gain.cpu()); R["reach"].append(reach.cpu())
    nfl = torch.zeros(Bl, dtype=torch.int16, device=dev); nobf = torch.zeros(Bl, dtype=torch.int16, device=dev)
    if L:
        nobf[i] = (L["ok"] & (typ[i] == 3)).sum(1).to(torch.int16)
        if L["n_floor"] is not None:
            nfl[i] = L["n_floor"].to(torch.int16)
    R["nfl"].append(nfl.cpu()); R["nobf"].append(nobf.cpu())
    R["att"].append(env.sim.att[lid][:, [0, 2]].float().cpu())
    R["tilt"].append(opp.tilt.tilt.float().cpu() if opp.tilt is not None else torch.full((Bl, 2), float("nan")))
    # its own line: the car's sideways offset from it (left +), the line's curvature there (left +), the speed it asks for
    off = torch.full((Bl,), float("nan"), device=dev); kap = torch.zeros(Bl, device=dev); vref = torch.zeros(Bl, device=dev)
    if L:
        ln, t_i = opp.core.line, env.sim.tid[lid][i]
        h0 = ln.heading_at(t_i, L["s0"][:, None]).squeeze(1)
        off[i] = (-torch.sin(h0) * (L["pose"][:, 0] - L["npt"][:, 0]) + torch.cos(h0) * (L["pose"][:, 1] - L["npt"][:, 1])).float()
        kap[i] = ln.curvature_at(t_i, L["s0"][:, None]).squeeze(1).float(); vref[i] = L["ref"].float()
    R["off"].append(off.cpu()); R["kap"].append(kap.cpu()); R["vref"].append(vref.cpu())
    R["avoid"].append(opp.core.avoiding.clone().cpu()); R["lost"].append(opp.core.lost.clone().cpu())
    xy = env.sim.state[lid, :2].float()
    out = pr.decision(xy[:, None].expand(-1, PS.N_STEP, -1).contiguous(), None, restarted)
    opp.after()
    opp.scale[:] = fixed                                           # a restart draws a new multiple: each car keeps its own
    if remover is not None:
        remover.step()
    stuck = torch.zeros(Bl, dtype=torch.bool, device=dev)
    if stuck_s > 0:
        # `state_rl.OppRestart`'s rule: less than 0.5 m of net movement over stuck_s of driving
        age = torch.where(restarted, torch.zeros_like(age), age + 1)
        stuck = (age > K_ST) & ((xy - hist[h_i]).norm(dim=1) < 0.5)
        hist[h_i] = xy
        h_i = (h_i + 1) % K_ST
        if bool(stuck.any()):
            rows = torch.nonzero(stuck).flatten()
            env.restart_cars(lid[rows], far=True)
            pr.begin(rows); opp._reset(rows)
            ring[:, rows] = float("nan")
            age[rows] = 0
            n_stuck += int(rows.numel())
    R["stuck"].append(stuck.cpu())
    R["xt"].append(xt_c.permute(1, 0, 2).to(torch.float16).cpu()); R["xav"].append(xav_c.permute(1, 0, 2).cpu())
    R["rs"].append((restarted | stuck).cpu()); R["lap"].append(out["lap"].float().cpu())
    R["ors"].append(torch.zeros(Bl, dtype=torch.bool)); R["newp"].append(torch.ones(Bl, dtype=torch.bool))
    R["pl"].append(xy[:, None].expand(-1, PS.N_STEP, -1).cpu().clone())
    cf = env.sim.contact_frame()
    R["cnt"].append(cf[lid].float()[:, I_CNT].cpu()); R["ocnt"].append(cf[others].float()[:, I_CNT].cpu())
    R["held"].append(cf[lid].float()[:, [i_hk, i_hx, i_hy, i_hs]].cpu())
    if (kd + 1) % 100 == 0:
        print(f"  {kd + 1}/{T} 결정 ({time.time() - t0:.0f} s)", flush=True)
snap()
D = {k: torch.stack(v).numpy() for k, v in R.items()}
trk = env.sim.track
D.update(edt=trk.edt.float().cpu().numpy(), t_off=trk.t_off.cpu().numpy(), t_H=trk.t_H.cpu().numpy(), t_W=trk.t_W.cpu().numpy(),
         t_res=trk.t_res.cpu().numpy(), t_origin=trk.t_origin.cpu().numpy(),
         car=np.array([float(env.cfg.vehicle.length), float(env.cfg.vehicle.width)]), dt=np.array(PS.DT),
         kinds=np.array(KN), cols=np.array(COLS), name=np.array("aurorus 포팅본"), ep_s=np.array(float(OV["episode_s"])),
         level=env.level_seed[lid].cpu().numpy(), scales=np.array(SCALES), stuck_s=np.array(stuck_s),
         mount=np.array([float(env.cfg.lidar.mount_x), float(env.cfg.lidar.mount_y)]))
for k in sorted({int(s) for s in np.unique(D["sid"]) if s >= 0}):
    sh = po.shapes[k]
    D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
D["aur_cfg"] = np.array(json.dumps(AUR_CFG)); D["deskew"] = np.array(opp.core.sweep is not None)
D["sim_sweep"] = np.array(bool(env.cfg.lidar.motion_distortion))
D["drive"] = np.array(DRIVE); D["floor"] = np.array(FLOOR); D["floor_tol"] = np.array(FLOOR_TOL)
D["map_set"] = np.array(json.dumps(MAP_SET or {})); D["calibrated"] = np.array(True)     # physics drive through each car's calibration (11-12 20:00 on)
np.savez_compressed(out_npz, **D)
inc = np.clip(np.diff(np.concatenate([np.zeros((1,) + D["cnt"].shape[1:], np.float32), D["cnt"]], 0), axis=0), 0, None).sum((0, 1))
print(f"aurorus 포팅본: 차 {Bl}대 x {secs:.0f} s ({time.time() - t0:.0f} s 걸림, 제어 주기당 {1e3 * (time.time() - t0) / (T * PS.SUB):.1f} ms, 그중 포팅본 계산 {1e3 * t_port / (T * PS.SUB):.1f} ms), "
      f"주행 {'보통 물리' if DRIVE == 'physics' else '정확 실행'}, 바닥 점 {dict(off='그대로', gate=f'자기 센서로 거름(여유 {FLOOR_TOL:g})', oracle='정답으로 거름')[FLOOR]}, "
      f"스캐너 위치 무게중심 앞 {float(env.cfg.lidar.mount_x):.3f} m, 스캔 도는 동안의 움직임 보정 {'켬' if opp.core.sweep is not None else '끔'}"
      f"{', 시뮬레이터 스캔 왜곡 끔' if not env.cfg.lidar.motion_distortion else ''}{', 설정 ' + json.dumps(AUR_CFG) if AUR_CFG else ''} | 접촉 시작: "
      + " · ".join(f"{k} {int(v)}" for k, v in zip(COLS, inc)) + f" | 서 있어서 다시 출발 {n_stuck}번 | 회피 중이던 결정 "
      f"{100 * D['avoid'].mean():.0f} % · 선에서 벗어난 결정 {int(D['lost'].sum())} · 장애물로 본 점 {int(D['nob'].sum())}개 가운데 실제 바닥 {int(D['nobf'].sum())}개"
      f" · gate가 버린 점 {int(D['nfl'].sum())}개 | 저장 {out_npz}", flush=True)
