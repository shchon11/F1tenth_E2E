"""상대 차와 세게 닿는 장면 (11-10, 다음 목표 = 상대 차 접촉; 사용자 규칙 "LOOK AT SCENES": 숫자보다 장면 먼저). 학습 세계(plan_rl.build,
학습 조건 = plan대로 정확히, 고정 시드)를 policy로 몰며 상대 차와 세게(>= 1 m/s) 닿는 순간을 모두 모아:
  * 갈래: 닿는 순간 상대 차가 우리 차 기준 앞(dx > 0.3 m) = '앞차를 박음', 뒤(dx < -0.3 m) = '뒤차에 박힘', 그 사이 = '옆으로 닿음',
    두 차 속도, 상대 차의 속도 배율(학습 세계의 상대는 레이싱 라인을 0.4-1.2배로 달림)
  * 그림 1: 맵 위 닿은 자리(갈래 색)  그림 2: 장면 12개 확대(±3.5 m) -- 닿기 1.5 s 전부터 두 차의 자취(점, 시간 색), 닿는 순간 차체(파랑 =
    우리, 빨강 = 상대), 닿기 0.5 s 전 우리 plan(검은 선)
인자: checkpoint 출력머리 [환경 수, 기본 512] [초, 기본 40]"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import font_manager
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2, PT_IDX
from f1sim.learn.plan_reward import PlanReward
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim import plan_steps as PS
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

torch.set_grad_enabled(False)
ck_path, head = sys.argv[1], sys.argv[2]
n_env = int(sys.argv[3]) if len(sys.argv) > 3 else 512
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, n_env, 21, dev, fmt="v2")
lid = env.learner_ids; Bl = int(lid.numel())
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
others = env.sim.other_idx[lid][:, 0] if env.M > 1 else None
if others is None:
    raise SystemExit("이 세계에는 상대 차 자리가 없음")
i_hc = CC.index("hard_car")
T = int(secs / PS.DT)
H = 15                                                           # 1.5 s of history before a contact
ST, OP, PL, OPP_ON = [], [], [], []
prev = env.sim.contact_frame()[lid, i_hc].clone()
events = []
for kd in range(T):
    st = env.sim.state[lid, :4].float().clone(); so = env.sim.state[others, :4].float().clone()
    absent = env.sim.absent[others] if getattr(env.sim, "absent_any", False) else torch.zeros(Bl, dtype=torch.bool, device=dev)
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    out, p_, (x, y, _) = drv.decision(torch.tanh(mean), p)
    ii = PT_IDX.to(dev)
    ST.append(st.cpu()); OP.append(so.cpu()); OPP_ON.append((~absent).cpu())
    PL.append(torch.stack([x[:, ii], y[:, ii]], -1).cpu())
    now = env.sim.contact_frame()[lid, i_hc]
    hit = (now - prev) > 0.5
    prev = now.clone()
    for b in torch.nonzero(hit & ~out["restarted"])[:, 0].tolist():
        events.append((kd, b))
TID = env.sim.tid[lid].cpu().numpy()
ST, OP, PL, OPP_ON = torch.stack(ST), torch.stack(OP), torch.stack(PL), torch.stack(OPP_ON)   # (T, Bl, ...)
mins = secs * float(OPP_ON[0].float().sum()) / 60.0
rows = []
for kd, b in events:
    s, o = ST[kd, b].numpy(), OP[kd, b].numpy()
    c, s_ = np.cos(s[2]), np.sin(s[2])
    d = o[:2] - s[:2]
    dx, dy = d[0] * c + d[1] * s_, -d[0] * s_ + d[1] * c
    kind = "앞차를 박음" if dx > 0.3 else ("뒤차에 박힘" if dx < -0.3 else "옆으로 닿음")
    rows.append(dict(kd=kd, b=b, kind=kind, dx=dx, dy=dy, v=float(s[3]), vo=float(o[3]), x=float(s[0]), y=float(s[1]), tid=int(TID[b])))
kinds = ("앞차를 박음", "옆으로 닿음", "뒤차에 박힘")
cols = {"앞차를 박음": "tab:red", "옆으로 닿음": "tab:orange", "뒤차에 박힘": "tab:purple"}
n = {k: sum(r["kind"] == k for r in rows) for k in kinds}
print(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}: 상대 차와 세게 닿음 {len(rows)}번 "
      f"({len(rows) / max(mins, 1e-6):.2f}/분, 상대 있는 차 {int(OPP_ON[0].sum())}대 x {secs:.0f} s) | "
      + " · ".join(f"{k} {n[k]}" for k in kinds)
      + (f" | 닿을 때 속도 중앙: 우리 {np.median([r['v'] for r in rows]):.1f}, 상대 {np.median([r['vo'] for r in rows]):.1f} m/s" if rows else ""),
      flush=True)
# 1) where on the map -- each event on ITS track (the training world mixes the track with warped copies: tid 0 = the track)
trk = env.sim.track
_edt = trk.edt.float().cpu().numpy(); _off = trk.t_off.cpu().numpy(); _H = trk.t_H.cpu().numpy().astype(int); _W = trk.t_W.cpu().numpy().astype(int)
_res = trk.t_res.cpu().numpy(); _org = trk.t_origin.cpu().numpy()
def grid(k):
    g = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k])
    return g <= 0.0, [_org[k][0], _org[k][0] + _W[k] * _res[k], _org[k][1], _org[k][1] + _H[k] * _res[k]]
occ, ext = grid(0)
n0 = sum(r['tid'] == 0 for r in rows)
fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.6)
for k in kinds:
    pts = [(r["x"], r["y"]) for r in rows if r["kind"] == k and r["tid"] == 0]
    if pts:
        a = np.array(pts); ax.scatter(a[:, 0], a[:, 1], s=30, c=cols[k], label=f"{k} {len(pts)}", edgecolors="k", linewidths=0.5)
ax.legend(fontsize=10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"상대 차와 세게 닿은 자리 -- 원래 트랙에서 난 {n0}번(전체 {len(rows)}번 중; 나머지는 변형 트랙 16개)", fontsize=11)
plt.tight_layout(); plt.savefig(head + "_map.png", dpi=80); plt.close(fig)
# 2) scenes
hl, hw = 0.29, 0.15
def body(x, y, yaw):
    cc, ss = np.cos(yaw), np.sin(yaw)
    P = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.stack([x + P[:, 0] * cc - P[:, 1] * ss, y + P[:, 0] * ss + P[:, 1] * cc], 1)
pick = []
for k in kinds:                                                   # a few of every kind, the commonest kind most
    rk = [r for r in rows if r["kind"] == k]
    pick += rk[: max(2, int(round(12 * len(rk) / max(1, len(rows)))))]
pick = pick[:12]
if pick:
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    for ax, r in zip(axs.flat, pick):
        kd, b = r["kd"], r["b"]
        k0 = max(0, kd - H)
        occ_k, ext_k = grid(r["tid"])
        ax.imshow(~occ_k, cmap="gray", origin="lower", extent=ext_k, alpha=0.6)
        s_tr = ST[k0:kd + 1, b].numpy(); o_tr = OP[k0:kd + 1, b].numpy()
        tt = np.linspace(0, 1, len(s_tr))
        ax.scatter(s_tr[:, 0], s_tr[:, 1], c=tt, cmap="Blues", s=14, vmin=-0.3, vmax=1)
        ax.scatter(o_tr[:, 0], o_tr[:, 1], c=tt, cmap="Reds", s=14, vmin=-0.3, vmax=1)
        ax.add_patch(Polygon(body(*s_tr[-1, :3]), closed=True, fc="tab:blue", ec="k"))
        ax.add_patch(Polygon(body(*o_tr[-1, :3]), closed=True, fc="tab:red", ec="k"))
        kp = max(0, kd - 5)
        pl = PL[kp, b].numpy(); ax.plot(pl[:, 0], pl[:, 1], "k-", lw=1.5)
        ax.set_xlim(r["x"] - 3.5, r["x"] + 3.5); ax.set_ylim(r["y"] - 3.5, r["y"] + 3.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r['kind']} · 우리 {r['v']:.1f} / 상대 {r['vo']:.1f} m/s · 상대 자리 앞 {r['dx']:+.1f} 옆 {r['dy']:+.1f} m"
                     + (" (원래 트랙)" if r['tid'] == 0 else f" (변형 {r['tid']})"), fontsize=10)
    for ax in list(axs.flat)[len(pick):]:
        ax.axis("off")
    fig.suptitle("상대 차와 세게 닿기 1.5 s 전부터 (파랑 = 우리, 빨강 = 상대, 점 = 자취(진할수록 최근), 검은 선 = 닿기 0.5 s 전 우리 plan)", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(head + "_scenes.png", dpi=70); plt.close(fig)
print(head + "_map.png", head + "_scenes.png", flush=True)
