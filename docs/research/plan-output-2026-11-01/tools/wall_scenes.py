"""벽에 세게 닿는 장면, 콘솔 경로 (11-10: MPC가 차 128대에서 벽에 세게 1.1-1.5번/분 -- 지금 제어기 0.38 -- 왜인지 숫자 말고 장면으로).
콘솔 워커(console_wobble과 같은 길, 혼자 빈 트랙)로 몰며 걸음마다 차의 자리·방향·속도·실제 조향·보낸 조향 명령·요 각속도와, 결정마다
policy의 plan(제어기가 따라가는 자리)을 모으고, 벽에 세게(>= 1 m/s) 닿는 순간마다 그림 한 칸: 닿기 2 s 전부터 차 자취(시간 색), 닿기
0.3 s 전 plan(검정), 닿는 순간 차 자리(빨강 x), 벽(트랙 격자). 아래 줄: 같은 2 s의 조향(실제 / 명령 / plan의 첫 조향)과 속도.
인자: checkpoint 출력png [초, 기본 180]   (제어기는 F1SIM_PLAN_MODE / F1SIM_TRACK_MPC 등 콘솔과 같은 환경 변수)"""
import os, sys, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim import dynamics as dyn
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, out = sys.argv[1], sys.argv[2]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name="real/iccas25", seed=0, races=1, cars_per_race=1, device="cuda",
                                    compile=False, collision_mode="soft", procedural=False, episode_s=1e4, randomize=False), 0)
env = s["env"]
pr = s["act_fn"]
f = pr.follower
i_hw = CC.index("hard_wall")
for _ in range(80):
    w._step_once(s)
prev = float(env.sim.contact_frame()[0, i_hw])
rec, plans, events = [], {}, []
n = int(secs / 0.025)
for k in range(n):
    w._step_once(s)
    st = env.sim.state[0].float().cpu().numpy()
    cmd = float(env.ext_cmd[0, 0]) if getattr(env, "ext_cmd", None) is not None else float("nan")
    p0 = float(pr.cmd_d[0, 0]) if hasattr(pr, "cmd_d") else float("nan")
    rec.append((st[0], st[1], st[2], st[dyn.IVX], math.degrees(st[dyn.ISTEER]), math.degrees(cmd), math.degrees(p0),
                math.degrees(st[dyn.IR])))
    if int(f.k[0]) == 1:                                           # the plan the follower got this decision
        plans[k] = (f.X[0].float().cpu().numpy().copy(), f.Y[0].float().cpu().numpy().copy())
    now = float(env.sim.contact_frame()[0, i_hw])
    if now - prev > 0.5:
        events.append(k)
    prev = now
R = np.array(rec)
mode = os.environ.get("F1SIM_PLAN_MODE", "track") + (" MPC" if os.environ.get("F1SIM_TRACK_MPC") == "1" else "")
print(f"[{mode}] {secs:.0f} s: 벽에 세게 닿음 {len(events)}번 ({len(events) / secs * 60:.2f}/분)", flush=True)
trk = env.sim.track
edt = trk.edt.float().cpu().numpy(); off = int(trk.t_off[0]); Hh = int(trk.t_H[0]); Ww = int(trk.t_W[0])
res = float(trk.t_res[0]); org = trk.t_origin[0].cpu().numpy()
occ = edt[off: off + Hh * Ww].reshape(Hh, Ww) <= 0.0
ext = [org[0], org[0] + Ww * res, org[1], org[1] + Hh * res]
ev = events[:6]
if not ev:
    raise SystemExit(0)
fig, axs = plt.subplots(2, len(ev), figsize=(4.2 * len(ev), 8.4), squeeze=False)
for c, k in enumerate(ev):
    a, b = max(0, k - 80), k + 1
    ax = axs[0, c]
    ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.6)
    ax.scatter(R[a:b, 0], R[a:b, 1], c=np.arange(b - a), cmap="viridis", s=6)
    kp = max([j for j in plans if j <= k - 12] or [None]) if plans else None
    if kp is not None:
        ax.plot(plans[kp][0], plans[kp][1], "k-", lw=1.4, label="닿기 0.3 s 전 plan")
    ax.plot(R[k, 0], R[k, 1], "rx", ms=12, mew=2.5)
    ax.set_xlim(R[k, 0] - 2.5, R[k, 0] + 2.5); ax.set_ylim(R[k, 1] - 2.5, R[k, 1] + 2.5); ax.set_aspect("equal")
    ax.set_title(f"{(k + 80) * 0.025:.1f} s, 속도 {R[k, 3]:.1f} m/s", fontsize=10)
    if c == 0:
        ax.legend(fontsize=8, loc="upper left")
    ax2 = axs[1, c]
    t = (np.arange(a, b) - k) * 0.025
    ax2.plot(t, R[a:b, 4], color="#eb6834", lw=1.3, label="실제 조향")
    ax2.plot(t, R[a:b, 5], color="#2a78d6", lw=1.0, alpha=0.8, label="보낸 명령")
    ax2.plot(t, R[a:b, 6], color="#1baf7a", lw=1.2, drawstyle="steps-post", label="plan의 첫 조향")
    ax2.axhline(24, color="gray", lw=0.6, ls="--"); ax2.axhline(-24, color="gray", lw=0.6, ls="--")
    ax2.set_xlabel("닿기까지 [s]"); ax2.set_ylim(-26, 26); ax2.grid(alpha=0.25)
    if c == 0:
        ax2.set_ylabel("조향 [deg]"); ax2.legend(fontsize=8, loc="lower left")
fig.suptitle(f"벽에 세게 닿는 장면 -- {mode}, {os.path.basename(os.path.dirname(run))}/{os.path.basename(run)}, 혼자 빈 트랙 "
             f"({len(events)}번 / {secs:.0f} s)", fontsize=12)
plt.tight_layout(); plt.savefig(out, dpi=75)
print(out, flush=True)
