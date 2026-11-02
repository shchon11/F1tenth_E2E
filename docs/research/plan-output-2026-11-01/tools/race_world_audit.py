"""상대 단계 학습 세계 진단 (11-02 13:20): 체크포인트를 그 학습과 똑같은 세계(args 그대로: 상대 종류·배속·막기 등 성향·돌발 행동·
상자·치우기·혼자 선생 혼합)에서 학습 때처럼 탐색 잡음으로 몰고, 세게 부딪힐 때마다 상황을 적는다 -- 무엇에(벽/상자/차), 누가
몰았나(상대 단계 정책 / 혼자 선생), 상대가 있나·거리·배속·성향·돌발 행동 중인가, 가까운 상자 거리, 내 속도. 상황별 km당 충돌과 진행,
J80. 충돌 장면 일부는 그림(2 s 궤적 + 상자). 인자: 체크포인트 판수 초 시드 그림머리 [잡음 1/0]"""
import os, sys, math, json, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

path, n_env, secs, seed, head = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
noise = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, seed, "cuda"
env = SR.build_env(a, dev)
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=seed)
lid = env.learner_ids; B = int(lid.numel())
remover = SR.PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=seed) \
    if float(getattr(a, "removal_p", 0) or 0) > 0 else None
tp = SR.TeacherPolicy(env, ck, dev)
opp = env.sim.other_idx[lid][:, 0]                                         # (B,) each learner's opponent row
act = torch.zeros(B, tp.A, device=dev); eps = torch.zeros_like(act)
act_full = torch.zeros(env.B, env.act_dim, device=dev)
K = {k: CC.index(k) for k in ("hard_wall", "hard_prop", "hard_car")}
cf_prev = env.sim.contact_frame()[lid].float()
R = int(a.repeat); steps = int(secs / env.sim.control_dt)
HB = 80                                                                    # 2 s of positions behind a hit
hist_a = torch.zeros(B, HB, 2, device=dev); hist_b = torch.zeros(B, HB, 2, device=dev)
rows = []                                                                  # one per hard hit
pics = []
# progress per situation: (opponent within 10 m?, prop within 6 m?, race policy driving?)
prog_s = torch.zeros(8, device=dev); time_s = torch.zeros(8, device=dev)
k_step = 0
po = env.procedural
for t in range(steps // R):
    if noise:
        tp.o = tp.sobs(act)
        dist = tp.net.dist(tp.nrm(tp.o))
        al = float(a.noise_corr)
        eps = al * eps + math.sqrt(1.0 - al * al) * torch.randn_like(eps)
        a_pol = SR.clamp_any(dist.mean + dist.stddev * eps, tp.ar)
        act = tp.mix(a_pol, act) if tp.mix is not None else a_pol
    else:
        act = tp(act)
    act_full[lid] = act
    gate = tp.mix.gate.clone() if tp.mix is not None else torch.ones(B, dtype=torch.bool, device=dev)
    for _ in range(R):
        _, r_, te, trc, info = env.step(act_full)
        if remover is not None:
            remover.step()
        k_step += 1
        st = env.sim.state
        hist_a = torch.roll(hist_a, -1, 1); hist_a[:, -1] = st[lid, :2]
        hist_b = torch.roll(hist_b, -1, 1); hist_b[:, -1] = st[opp, :2]
        pres = ~env.sim.absent[opp] if env.sim.absent_any else torch.ones(B, dtype=torch.bool, device=dev)
        d_car = torch.where(pres, (st[lid, :2] - st[opp, :2]).norm(dim=1), torch.full((B,), 99.0, device=dev))
        if po is not None and po.C > 0:
            pp = po.p_poses[lid][..., :2]; live = po.p_sid[lid] >= 0
            dp = torch.where(live, (pp - st[lid, None, :2]).norm(dim=-1), torch.full(live.shape, 99.0, device=dev)).amin(1)
        else:
            dp = torch.full((B,), 99.0, device=dev)
        sit = (d_car < 10).long() * 4 + (dp < 6).long() * 2 + gate.long()
        p_ = info["progress"][lid].float().clamp_min(0)
        prog_s.index_add_(0, sit, p_); time_s.index_add_(0, sit, torch.full_like(p_, env.sim.control_dt))
        cf = env.sim.contact_frame()[lid].float()
        dh = {k: (cf[:, i] - cf_prev[:, i]) > 0 for k, i in K.items()}
        cf_prev = cf
        hit = dh["hard_wall"] | dh["hard_prop"] | dh["hard_car"]
        if bool(hit.any()):
            ids = torch.nonzero(hit).flatten().tolist()
            g = env.signed_gaps(env.last_result.s, env.sim.tid)[lid][:, 0]
            for i in ids:
                kind = "차" if bool(dh["hard_car"][i]) else ("상자" if bool(dh["hard_prop"][i]) else "벽")
                o = int(opp[i])
                rows.append(dict(t=k_step * env.sim.control_dt, kind=kind, race=bool(gate[i]), present=bool(pres[i]),
                                 d_car=float(d_car[i]), gap=float(g[i]), scale=float(env.opp_scale[o]),
                                 disp=int(env.events.disp[o]) if env.events is not None else 0,
                                 event=int(env.events.kind[o]) if env.events is not None else 0,
                                 d_prop=float(dp[i]), v=float(st[lid[i], 3]), v_opp=float(st[o, 3])))
                if len(pics) < 24 and (kind != "벽" or len(pics) < 8):
                    props = None
                    if po is not None and po.C > 0:
                        lv = po.p_sid[lid[i]] >= 0
                        props = po.p_poses[lid[i]][lv][:, :2].cpu().numpy()
                    pics.append(dict(row=rows[-1], a=hist_a[i].cpu().numpy(), b=hist_b[i].cpu().numpy(), props=props,
                                     tid=int(env.sim.tid[lid[i]])))
km = float(prog_s.sum()) / 1000.0
n_hit = {k: sum(1 for r in rows if r["kind"] == k) for k in ("벽", "상자", "차")}
print(f"== {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} | {'탐색 잡음' if noise else '평균 행동'} | {B}대 x {secs:.0f} s = {km:.1f} km | "
      f"진행 {float(prog_s.sum()) / float(time_s.sum()):.2f} m/s | 세게 벽 {n_hit['벽'] / km:.2f} 상자 {n_hit['상자'] / km:.2f} 차 {n_hit['차'] / km:.2f} /km | "
      f"J80 {(float(prog_s.sum()) - 80 * len(rows)) / float(time_s.sum()) * env.sim.control_dt:.4f}/스텝")
names = {0: "차X 상자X 혼자선생", 1: "차X 상자X 상대단계", 2: "차X 상자O 혼자선생", 3: "차X 상자O 상대단계",
         4: "차O 상자X 혼자선생", 5: "차O 상자X 상대단계", 6: "차O 상자O 혼자선생", 7: "차O 상자O 상대단계"}
for s_ in range(8):
    tt = float(time_s[s_])
    if tt < 1.0:
        continue
    km_s = float(prog_s[s_]) / 1000.0
    sel = [r for r in rows if ((r["d_car"] < 10) * 4 + (r["d_prop"] < 6) * 2 + int(r["race"])) == s_]
    c = {k: sum(1 for r in sel if r["kind"] == k) for k in ("벽", "상자", "차")}
    print(f"  {names[s_]:14s} 시간 {tt / float(time_s.sum()) * 100:5.1f} % | 진행 {float(prog_s[s_]) / tt:.2f} m/s | 세게 벽 {c['벽']} 상자 {c['상자']} 차 {c['차']} "
          f"({len(sel) / max(km_s, 1e-3):.2f}/km)")
# opponent-side breakdown of car hits and of hits with a car near
for lab, f in (("상대 막기 성향", lambda r: r["disp"] & 1), ("상대 돌발 행동 중", lambda r: r["event"] > 0),
               ("상대 0.75배속 미만", lambda r: r["scale"] < 0.75)):
    near = [r for r in rows if r["d_car"] < 10]
    print(f"  차 10 m 안 충돌 {len(near)}건 중 {lab}: {sum(1 for r in near if f(r))}")
json.dump(rows, open(head + "_rows.json", "w"), ensure_ascii=False)
# pictures
if pics:
    from f1sim import maps
    n = len(pics); cols = 4; rws = (n + cols - 1) // cols
    fig, axs = plt.subplots(rws, cols, figsize=(5 * cols, 5 * rws), squeeze=False)
    tr = env.sim.track
    for ax, p in zip(axs.flat, pics):
        A_, B_ = p["a"], p["b"]
        ax.plot(A_[:, 0], A_[:, 1], "-", color="tab:blue", lw=1.5); ax.plot(A_[-1, 0], A_[-1, 1], "o", color="tab:blue")
        if p["row"]["present"]:
            ax.plot(B_[:, 0], B_[:, 1], "-", color="orange", lw=1.5); ax.plot(B_[-1, 0], B_[-1, 1], "o", color="orange")
        if p["props"] is not None and len(p["props"]):
            ax.plot(p["props"][:, 0], p["props"][:, 1], "s", color="saddlebrown", ms=6)
        cl = tr.cl[p["tid"]].cpu().numpy()
        ax.plot(cl[:, 0], cl[:, 1], ":", color="gray", lw=0.6)
        c = A_[-1]
        ax.set_xlim(c[0] - 4, c[0] + 4); ax.set_ylim(c[1] - 4, c[1] + 4); ax.set_aspect("equal")
        r = p["row"]
        ax.set_title(f"{r['kind']} | {'상대단계' if r['race'] else '혼자선생'} | 내 {r['v']:.1f} 상대 {r['v_opp']:.1f} m/s\n"
                     f"차 {r['d_car']:.1f} m 배속 {r['scale']:.2f} 성향 {r['disp']} 사건 {r['event']} | 상자 {r['d_prop']:.1f} m", fontsize=8)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    from matplotlib import font_manager
    fig.suptitle(f"{os.path.basename(path)}: 세게 부딪힌 순간 앞 2 s (파랑 = 학습 차, 주황 = 상대, 갈색 = 상자, 점선 = 중심선)")
    fig.tight_layout(); fig.savefig(head + "_hits.png", dpi=55); plt.close(fig)
    print("그림", head + "_hits.png")
