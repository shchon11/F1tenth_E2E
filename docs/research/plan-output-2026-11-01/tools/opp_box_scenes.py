"""학습 세계의 상대 차가 상자에 닿는 장면 (11-11: 상자 있는 경주에서 상대 차가 상자에 9번/분 닿고 18 %를 서 있음 -- 네 종류 모두,
속도 배율·이벤트 옆 이동과 무관. 사용자 규칙 "LOOK AT SCENES"). 학습 세계(plan_rl.build, 상자 치움 포함), 학습 조건, policy 평균 행동,
고정 시드. 상대 차가 상자에 새로 닿는 순간마다 한 건(차마다 첫 건만): 종류별로 몇 장씩 12장, 확대(±3 m) -- 닿기 3 s 전부터의 상대 차
자취(점, 시간 색: 옅음 -> 진함), 닿는 순간 상대 차 차체(빨강), 우리 차(파랑), 상자(갈색), 제목 = 종류, 속도 배율, 닿을 때 속도.
인자: checkpoint 출력png [환경 수, 기본 512] [초, 기본 40]"""
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
from f1sim import props as _props
from f1sim import plan_steps as PS
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

torch.set_grad_enabled(False)
ck_path, out_png = sys.argv[1], sys.argv[2]
n_env = int(sys.argv[3]) if len(sys.argv) > 3 else 512
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
SEED = 4242
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:
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
others = env.sim.other_idx[lid][:, 0]
eid = env.sim.eid[lid]
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
KN = list(getattr(env, "alt_teacher_kinds", ())) + ["raceline"]
i_op, i_hp = CC.index("onsets_prop"), CC.index("hard_prop")
H = 30                                                           # 3 s before the contact
OP, ST, ON = [], [], []
events = []
seen = set()
prev = env.sim.contact_frame()[others, i_op].float().clone()
for kd in range(int(secs / PS.DT)):
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    out, _, _ = drv.decision(torch.tanh(mean), p)
    if remover is not None:
        remover.step()
    OP.append(env.sim.state[others, :4].float().cpu()); ST.append(env.sim.state[lid, :4].float().cpu())
    ON.append((~absent[others]).cpu())
    c = env.sim.contact_frame()[others, i_op].float()
    hit = ((c - prev) > 0.5) & ~absent[others] & ~out["restarted"]
    prev = c.clone()
    for b in torch.nonzero(hit)[:, 0].tolist():
        if b in seen or kd < 5:
            continue
        seen.add(b)
        code = int(env.slot_kind_code[others[b]]) if hasattr(env, "slot_kind_code") else -1
        scale = float(env.teacher.speed_scale[others[b]]) if torch.is_tensor(getattr(env.teacher, "speed_scale", None)) else 1.0
        events.append(dict(kd=kd, b=b, kind=KN[code] if code >= 0 else "raceline", scale=scale,
                           sid=po.p_sid[eid[b]].cpu().numpy(), pose=po.p_poses[eid[b]].float().cpu().numpy(),
                           tid=int(env.sim.tid[lid[b]])))
OP, ST, ON = torch.stack(OP).numpy(), torch.stack(ST).numpy(), torch.stack(ON).numpy()
print(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}: 상대 차가 상자에 처음 닿은 차 {len(events)}대 "
      f"(학습 차 {Bl}대 x {secs:.0f} s) | 종류별 " + " · ".join(f"{k} {sum(e['kind'] == k for e in events)}" for k in KN), flush=True)
trk = env.sim.track
_edt = trk.edt.float().cpu().numpy(); _off = trk.t_off.cpu().numpy(); _H = trk.t_H.cpu().numpy().astype(int); _W = trk.t_W.cpu().numpy().astype(int)
_res = trk.t_res.cpu().numpy(); _org = trk.t_origin.cpu().numpy()
MAPC = ListedColormap(["#666666", "white"])
hl, hw = 0.5 * float(env.cfg.vehicle.length), 0.5 * float(env.cfg.vehicle.width)
_fp = {}


def grid(k):
    g = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k])
    return g <= 0.0, [_org[k][0], _org[k][0] + _W[k] * _res[k], _org[k][1], _org[k][1] + _H[k] * _res[k]]


def body(x, y, yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    q = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.stack([x + q[:, 0] * c - q[:, 1] * s_, y + q[:, 0] * s_ + q[:, 1] * c], 1)


def boxes(sid, pose):
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


def trk_name(t):
    return "원래 트랙" if t == 0 else f"변형 트랙 {t}"


pick = []
for k in KN:                                                     # three of every kind first, then the rest in order
    pick += [e for e in events if e["kind"] == k][:3]
pick += [e for e in events if e not in pick]
pick = pick[:12]
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
for ax in axs.flat:
    ax.set_axis_off()
for ax, e in zip(axs.flat, pick):
    ax.set_axis_on(); ax.set_facecolor("#666666")
    kd, b = e["kd"], e["b"]
    k0 = max(0, kd - H)
    occ, ext = grid(e["tid"])
    ax.imshow(~occ, cmap=MAPC, vmin=0, vmax=1, origin="lower", extent=ext, zorder=0)
    for q in boxes(e["sid"], e["pose"]):
        ax.fill(q[:, 0], q[:, 1], color="saddlebrown", alpha=0.9, lw=0, zorder=2)
    o_tr = OP[k0:kd + 1, b]
    tt = np.linspace(0, 1, len(o_tr))
    ax.scatter(o_tr[:, 0], o_tr[:, 1], c=tt, cmap="Reds", s=16, vmin=-0.3, vmax=1, zorder=3)
    ax.add_patch(Polygon(body(*o_tr[-1, :3]), closed=True, fc="tab:red", ec="k", zorder=5))
    s_ = ST[kd, b]
    ax.add_patch(Polygon(body(*s_[:3]), closed=True, fc="tab:blue", ec="k", zorder=5))
    x, y = o_tr[-1, 0], o_tr[-1, 1]
    ax.set_xlim(x - 3, x + 3); ax.set_ylim(y - 3, y + 3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    v3 = o_tr[0, 3]
    ax.set_title(f"{e['kind']} · 속도 배율 {e['scale']:.2f} · 닿을 때 {o_tr[-1, 3]:.1f} m/s (3 s 전 {v3:.1f})\n"
                 f"{trk_name(e['tid'])} · {kd * PS.DT:.1f} s", fontsize=11)
fig.suptitle("학습 세계의 상대 차(빨강)가 상자(갈색)에 처음 닿는 순간 -- 점 = 닿기 3 s 전부터의 자취(옅음 -> 진함), 파랑 = 우리 차",
             fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(out_png, dpi=70); plt.close(fig)
print(f"그림 {out_png}", flush=True)
