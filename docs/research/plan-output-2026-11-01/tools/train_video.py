"""학습 세계 영상 (11-11, 사용자: "영상도 학습환경 기준으로뽑아"). 표의 숫자와 같은 세계 -- plan_rl.build(학습 세계: 원래 트랙 +
변형 트랙, 학습 상자 배치, 상대 차 = 레이싱 라인 0.4-1.2배, 경주의 반은 상자를 10-30 s에 치움), 학습 조건(plan대로 정확히), policy의
평균 행동, 통째 에피소드, 고정 시드(checkpoint마다 같은 차·같은 배치·같은 상대). 출발 때 모습으로 차 넷을 고름(원래 트랙 먼저):
  1 혼자 · 빈 트랙   2 혼자 · 상자(치우는 경주 먼저)   3 상대 차 · 빈 트랙   4 상대 차 · 상자(치우는 경주 먼저)
각 `secs` 초를 한 mp4로(0.1 s마다 한 장, 20 fps = 실제의 2배 빠르기). 왼쪽 = 그 차의 트랙 전체, 오른쪽 = 차 주변 8 m.
인자: checkpoint 출력mp4 [초, 기본 40] [환경 수, 기본 256]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.viewer import recorder as REC
from f1sim import props as _props
from f1sim import plan_steps as PS
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

torch.set_grad_enabled(False)
ck_path, out_mp4 = sys.argv[1], sys.argv[2]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
n_env = int(sys.argv[4]) if len(sys.argv) > 4 else 256
SEED = 4242                                                      # plan_validate's seed
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:      # the tyre limit the checkpoint was trained with
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
env, lim, world = PR.build(PR.WORLD, n_env, SEED, dev, fmt="v2", overrides={"episode_s": secs + 5.0})
lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
po = env.procedural
rm_p = float(getattr(world, "removal_p", 0.0))
remover = SR.PropRemoval(env, rm_p, *[float(v) for v in getattr(world, "removal_s", (10.0, 30.0))], seed=SEED) if rm_p > 0 else None
if remover is not None:
    remover.step()                                               # draws each race's removal time (nothing is due at 0 s)
# the stuck-opponent restart the checkpoint was trained with (11-11, plan_rl --opp-restart-s; F1SIM_OPP_RESTART_S overrides)
_rs = os.environ.get("F1SIM_OPP_RESTART_S") or (ck.get("args") or {}).get("opp_restart_s") or 0.0
restarter = SR.OppRestart(env, float(_rs), PS.DT) if float(_rs) > 0 and env.M > 1 else None
M = int(env.M)
eid = env.sim.eid[lid]
others = env.sim.other_idx[lid][:, 0] if M > 1 else None
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)

# the four cars, from the start (the same for every checkpoint: fixed seed)
tid0 = env.sim.tid[lid].cpu().numpy()
props0 = (po.p_sid[eid] >= 0).any(-1).cpu().numpy() if po is not None else np.zeros(Bl, bool)
opp0 = (~absent[others]).cpu().numpy() if others is not None else np.zeros(Bl, bool)
rm0 = torch.isfinite(remover.t_rm[lid // M]).cpu().numpy() if remover is not None else np.zeros(Bl, bool)
t_rm0 = remover.t_rm[lid // M].cpu().numpy() if remover is not None else np.full(Bl, np.inf)


def pick(mask, pref):
    for m in [mask & pref & (tid0 == 0), mask & (tid0 == 0), mask & pref, mask]:
        ii = np.nonzero(m)[0]
        if len(ii):
            return int(ii[0])
    return None


CASES = [("1/4 혼자 · 빈 트랙", ~props0 & ~opp0, np.ones(Bl, bool)),
         ("2/4 혼자 · 상자", props0 & ~opp0, rm0),
         ("3/4 상대 차 · 빈 트랙", ~props0 & opp0, np.ones(Bl, bool)),
         ("4/4 상대 차 · 상자", props0 & opp0, rm0)]
sel = [(lab, pick(m, p)) for lab, m, p in CASES]
sel = [(lab, b) for lab, b in sel if b is not None]
B_ = torch.tensor([b for _, b in sel], device=dev)
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car")]
i_hd = [CC.index(k) for k in ("hard_wall", "hard_prop", "hard_car")]

# drive the whole world, record the four cars
REC_ = {k: [] for k in ("st", "op", "op_on", "px", "py", "sid", "pose", "on", "hd", "rs", "lap", "tid")}
for kd in range(int(secs / PS.DT)):
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    out, _, (x, y, _) = drv.decision(torch.tanh(mean), p)
    if remover is not None:
        remover.step()
    if restarter is not None:
        restarter.step()
    b_ = lid[B_]
    REC_["st"].append(env.sim.state[b_, :4].float().cpu().numpy())
    if others is not None:
        o_ = others[B_]
        REC_["op"].append(env.sim.state[o_, :4].float().cpu().numpy()); REC_["op_on"].append((~absent[o_]).cpu().numpy())
    REC_["px"].append(x[B_].float().cpu().numpy()); REC_["py"].append(y[B_].float().cpu().numpy())
    if po is not None:
        REC_["sid"].append(po.p_sid[eid[B_]].cpu().numpy()); REC_["pose"].append(po.p_poses[eid[B_]].float().cpu().numpy())
    cf = env.sim.contact_frame()[b_].float().cpu().numpy()
    REC_["on"].append(cf[:, i_on]); REC_["hd"].append(cf[:, i_hd])
    REC_["rs"].append(out["restarted"][B_].cpu().numpy()); REC_["lap"].append((out["lap"][B_] > 0).cpu().numpy())
    REC_["tid"].append(env.sim.tid[b_].cpu().numpy())
R = {k: np.stack(v) for k, v in REC_.items() if v}                  # (T, 4, ...)
T = R["st"].shape[0]

# draw
trk = env.sim.track
_edt = trk.edt.float().cpu().numpy(); _off = trk.t_off.cpu().numpy(); _H = trk.t_H.cpu().numpy().astype(int); _W = trk.t_W.cpu().numpy().astype(int)
_res = trk.t_res.cpu().numpy(); _org = trk.t_origin.cpu().numpy()


def grid(k):
    g = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k])
    return g <= 0.0, [_org[k][0], _org[k][0] + _W[k] * _res[k], _org[k][1], _org[k][1] + _H[k] * _res[k]]


_fp = {}


def footprints(sid, pose):
    polys = []
    for j, k in enumerate(sid):
        if k < 0:
            continue
        if int(k) not in _fp:
            sh = po.shapes[int(k)]
            _fp[int(k)] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, float)
        q = _fp[int(k)]; c, s_ = np.cos(pose[j, 2]), np.sin(pose[j, 2])
        polys.append(np.stack([pose[j, 0] + q[:, 0] * c - q[:, 1] * s_, pose[j, 1] + q[:, 0] * s_ + q[:, 1] * c], 1))
    return polys


hl, hw = 0.5 * float(env.cfg.vehicle.length), 0.5 * float(env.cfg.vehicle.width)


def car_poly(x, y, yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    q = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.stack([x + q[:, 0] * c - q[:, 1] * s_, y + q[:, 0] * s_ + q[:, 1] * c], 1)


name = f"{os.path.basename(os.path.dirname(ck_path))} {os.path.basename(ck_path).replace('plan_', '').replace('.pt', '')}"
WPX, HPX, DPI = 1280, 720, 80
fig = plt.figure(figsize=(WPX / DPI, HPX / DPI), dpi=DPI, facecolor="white")
ax = fig.add_axes([0.01, 0.03, 0.58, 0.84]); bx = fig.add_axes([0.61, 0.03, 0.38, 0.84])
title = fig.text(0.01, 0.96, "", fontsize=13, va="center")
title2 = fig.text(0.01, 0.925, "", fontsize=11, va="center")
fig.text(0.01, 0.89, "학습 세계 · 학습 조건(plan대로 정확히) · 흰 = 달릴 수 있는 곳, 회색 = 벽, 갈색 = obstacle, 파랑 = 차, 빨강 = 상대 차, "
         "검은 선 = policy의 plan, 색 점 = 지나온 3 s(속도 색), X = 접촉(빨강 벽 · 주황 obstacle · 자홍 차, 굵게 = 세게)",
         fontsize=8.5, color="0.3", va="center")
enc = REC.FfmpegEncoder(out_mp4, WPX, HPX, 20)
KCOL = ("red", "darkorange", "magenta")
MAPC = ListedColormap(["#666666", "white"])                      # wall, free
frames = 0
for ci, (lab, b) in enumerate(sel):
    for a in (ax, bx):
        a.clear(); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_facecolor("#666666")   # off the map = wall
    cur_tid, map_art, box_art, box_key = None, [], [], None
    trail_sc = [a.scatter([], [], c=[], s=6, cmap="turbo", vmin=0, vmax=8, zorder=3) for a in (ax, bx)]
    car_p = [a.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc="tab:blue", ec="k", lw=1.0, zorder=5)) for a in (ax, bx)]
    opp_p = [a.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc="tab:red", ec="k", lw=1.0, zorder=5)) for a in (ax, bx)]
    plan_l = [a.plot([], [], "k-", lw=1.4, zorder=4)[0] for a in (ax, bx)]
    hit_art = []
    cnt = np.zeros(3, int); hcnt = np.zeros(3, int); laps = 0; n_rs = 0; removed_at = None
    prev_on, prev_hd = R["on"][0, ci], R["hd"][0, ci]
    t0 = 0
    for k in range(T):
        st = R["st"][k, ci]
        if R["rs"][k, ci]:                                           # a new episode: new layout, counters start again
            n_rs += 1; t0 = k; prev_on, prev_hd = R["on"][k, ci], R["hd"][k, ci]
            for h in hit_art:
                h.remove()
            hit_art = []
        if R["tid"][k, ci] != cur_tid:
            cur_tid = int(R["tid"][k, ci])
            for m_ in map_art:
                m_.remove()
            occ, ext = grid(cur_tid)
            map_art = [a.imshow(~occ, cmap=MAPC, vmin=0, vmax=1, origin="lower", extent=ext, zorder=0) for a in (ax, bx)]
        if po is not None:
            sid, pose = R["sid"][k, ci], R["pose"][k, ci]
            key = (sid.tobytes(), np.round(pose, 3).tobytes())
            if key != box_key:
                if box_key is not None and (sid >= 0).sum() == 0 and removed_at is None and not R["rs"][k, ci]:
                    removed_at = k * PS.DT
                box_key = key
                for h in box_art:
                    h.remove()
                box_art = [a.fill(q[:, 0], q[:, 1], color="saddlebrown", alpha=0.9, lw=0, zorder=2)[0]
                           for q in footprints(sid, pose) for a in (ax, bx)]
        on, hd = R["on"][k, ci], R["hd"][k, ci]
        for j in range(3):
            if on[j] > prev_on[j]:
                hard = bool(hd[j] > prev_hd[j]); cnt[j] += 1; hcnt[j] += int(hard)
                for a in (ax, bx):
                    hit_art.append(a.plot(st[0], st[1], "x", color=KCOL[j], ms=(14 if hard else 8), mew=(3 if hard else 1.5), zorder=6)[0])
        prev_on, prev_hd = on, hd
        laps += int(R["lap"][k, ci])
        tp = R["st"][max(t0, k - 29): k + 1, ci]
        cp = car_poly(*st[:3])
        has_o = "op" in R and bool(R["op_on"][k, ci])
        op_xy = car_poly(*R["op"][k, ci, :3]) if has_o else np.zeros((4, 2))
        for i in range(2):
            trail_sc[i].set_offsets(tp[:, :2]); trail_sc[i].set_array(tp[:, 3])
            car_p[i].set_xy(cp); opp_p[i].set_xy(op_xy)
            plan_l[i].set_data(R["px"][k, ci], R["py"][k, ci])
        occ, ext = grid(cur_tid)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        bx.set_xlim(st[0] - 4, st[0] + 4); bx.set_ylim(st[1] - 4, st[1] + 4)
        trk_name = "원래 트랙" if cur_tid == 0 else f"변형 트랙 {cur_tid}"
        title.set_text(f"{lab} · {trk_name} · {name} · {k * PS.DT:4.1f} s · 속도 {st[3]:4.1f} m/s · 바퀴 {laps}")
        title2.set_text(f"접촉: 벽 {cnt[0]}(세게 {hcnt[0]}) · obstacle {cnt[1]}(세게 {hcnt[1]})"
                        + (f" · 상대 차 {cnt[2]}(세게 {hcnt[2]}), 상대 속도 {R['op'][k, ci, 3]:.1f} m/s" if has_o else "")
                        + (f" · obstacle 치움 {removed_at:.1f} s" if removed_at is not None else "")
                        + (f" · 다시 출발 {n_rs}번(새 배치)" if n_rs else ""))
        fig.canvas.draw()
        enc.write(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).tobytes()); frames += 1
    print(f"{lab} | {('원래 트랙' if tid0[b] == 0 else f'변형 트랙 {tid0[b]}')} | 바퀴 {laps} | 접촉 벽 {cnt[0]}(세게 {hcnt[0]}) "
          f"obstacle {cnt[1]}(세게 {hcnt[1]}) 상대 차 {cnt[2]}(세게 {hcnt[2]})"
          + (f" | obstacle 치움 {removed_at:.1f} s" if removed_at is not None else (f" | 치움 예정 {t_rm0[b]:.1f} s" if np.isfinite(t_rm0[b]) and props0[b] else ""))
          + (f" | 다시 출발 {n_rs}번" if n_rs else ""), flush=True)
rc = enc.close()
print(f"영상 {out_mp4} | {frames}장, {frames / 20:.0f} s | {REC.describe_size(out_mp4)} | ffmpeg {rc} {enc.error or ''}", flush=True)
