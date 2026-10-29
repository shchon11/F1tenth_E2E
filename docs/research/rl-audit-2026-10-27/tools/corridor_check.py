"""상자 통로 관측(state_rl.corridor_block)이 실제 상자와 맞는지 그림으로 확인 (2026-10-29).
상자 세계(한국)에서 차 몇 대를 평균 행동으로 몰다가, 앞 13지점 x 가로 7칸의 표시(빨강 = 막힘, 초록 = 빔)를 상자 참 모양 위에 그림."""
import argparse, math, sys
import numpy as np, torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl
from f1sim.learn.state_rl import ActorCritic, RunningNorm, StateObs, AHEAD, CORRIDOR_U, CORRIDOR_S, CORRIDOR_BINS, corridor_block

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--out", required=True); a0 = ap.parse_args()
dev = torch.device("cuda")
a = argparse.Namespace(stage="props", warps=0, beams=108, envs=8, episode_s=40.0, seed=3, prop_density=1.5, prop_max=18)
env = state_rl.build_env(a, dev, extra=dict(stagger_first_episode=False))
from f1sim.learn.graph_runtime import prepare_graph_runtime
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=3)
ck = torch.load(a0.ckpt, map_location=dev, weights_only=False)
sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)))
lid = env.learner_ids; B = int(lid.numel()); last = torch.zeros(B, 2, device=dev)
o = sobs(last); net = ActorCritic(o.shape[1]).to(dev); net.load_state_dict(ck["model"], strict=False)
nrm = RunningNorm(o.shape[1], dev); nrm.load_state_dict(ck["norm"]); act_full = torch.zeros(env.B, env.act_dim, device=dev)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=100); k_img = 0
tr = env.sim.track
with torch.no_grad():
    for t in range(600):
        act = net.pi(nrm(sobs(last))).clamp(-1, 1); last = act; act_full[lid] = act
        for r in range(2):
            env.step(act_full)
        if t % 60 != 30 or k_img >= 6:
            continue
        st = env.last_result.state[lid]; s = env.last_result.s[lid]; tid = env.sim.tid[lid]
        n = len(AHEAD)
        blk = corridor_block(env, lid, s, tid).reshape(B, n, CORRIDOR_BINS).cpu().numpy()
        offs = torch.tensor(CORRIDOR_S, device=dev); R = offs.numel()
        seg = torch.bucketize(offs, torch.tensor(AHEAD, device=dev)).cpu().numpy()
        xy_k, yaw_k = tr.pose_at_s((s[:, None] + offs[None, :]).reshape(-1), tid.repeat_interleave(R))
        xy_k = xy_k.reshape(B, R, 2); yaw_k = yaw_k.reshape(B, R)
        w = tr.sample_edt(xy_k, tid[:, None].expand(-1, R))
        i = int(np.argmax(blk.reshape(B, -1).sum(1)))           # the car with the most blocked cells ahead
        ax = axs.flat[k_img]; k_img += 1
        track = tr.tracks[0]; occ = track.occupancy; res = track.resolution; ox, oy = track.origin
        x0, y0 = [float(v) for v in st[i, :2]]; W = 9.0
        ax.imshow(occ, origin="lower", cmap="Greys", extent=(ox, ox + occ.shape[1] * res, oy, oy + occ.shape[0] * res), vmin=0, vmax=1.5)
        po = env.procedural; row = int(lid[i])
        for c in (po.p_sid[row] >= 0).nonzero().flatten().tolist():
            nn_, dd_ = po.p_n[row, c].cpu().numpy(), po.p_d[row, c].cpu().numpy(); ok = np.isfinite(dd_); nn_, dd_ = nn_[ok], dd_[ok]
            ang = np.arctan2(nn_[:, 1], nn_[:, 0]); oo = np.argsort(ang); nn_, dd_ = nn_[oo], dd_[oo]
            cs = [np.linalg.solve(np.array([nn_[q], nn_[(q + 1) % len(nn_)]]), np.array([dd_[q], dd_[(q + 1) % len(nn_)]])) for q in range(len(nn_))
                  if abs(np.linalg.det(np.array([nn_[q], nn_[(q + 1) % len(nn_)]]))) > 1e-9]
            if len(cs) >= 3:
                cs = np.array(cs); px, py, pyaw = po.p_poses[row, c].cpu().numpy(); cc, ss = math.cos(pyaw), math.sin(pyaw)
                wd = cs @ np.array([[cc, ss], [-ss, cc]]) + np.array([px, py]); Q = np.vstack([wd, wd[:1]])
                ax.fill(Q[:, 0], Q[:, 1], color="darkorange", alpha=0.8)
        U = np.array(CORRIDOR_U); xk = xy_k[i].cpu().numpy(); yk = yaw_k[i].cpu().numpy(); wk = w[i].cpu().numpy()
        for kk in range(R):
            nx, ny = -math.sin(yk[kk]), math.cos(yk[kk])
            for m, u in enumerate(U):
                px_, py_ = xk[kk, 0] + u * wk[kk] * nx, xk[kk, 1] + u * wk[kk] * ny
                bad = blk[i, seg[kk], m // 2] > 0.5
                ax.plot(px_, py_, "o", ms=2.5, color=("red" if bad else "limegreen"), alpha=(0.9 if bad else 0.35))
        yaw0 = float(st[i, 2]); ax.arrow(x0, y0, 0.6 * math.cos(yaw0), 0.6 * math.sin(yaw0), width=0.08, color="blue")
        ax.set_xlim(x0 - W, x0 + W); ax.set_ylim(y0 - W, y0 + W); ax.set_aspect("equal")
        ax.set_title(f"차 {i}, 결정 {t}: 앞 13구간 x 가로 7칸 (빨강 = 그 구간·칸이 상자에 막힘, 차 반폭 0.15 m 부풀림)", fontsize=8)
fig.tight_layout(); fig.savefig(a0.out); print("saved", a0.out)
