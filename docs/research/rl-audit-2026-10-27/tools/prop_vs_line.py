"""상자가 정책의 길 위에 놓이나 (2026-10-29, 사용자: "장애물 같은 게 비어 있는 트랙에서 정책이 가는 길에 안 놓아지는 문제인가??").

1) 정책이 빈 트랙(상자 없음, 상대 없음, 한국)에서 실제로 타는 길: 평균 행동으로 몰아 차 중심 궤적을 모음.
2) 학습 세계와 같은 상자 생성기(state_rl.build_env, 밀도·최대 개수 그대로)로 한국 배치를 많이 만듦.
3) 상자마다 그 참 모양(국소 반평면)에서 길까지의 최소 거리 - 차 반폭(0.135 m) = 여유. 여유 < 0 이면 빈 트랙의 길 그대로는
   부딪힘(=길을 막음), 0~0.3 m 는 스침 거리.
그림: 트랙, 정책의 길(굵은 선), 배치 N개의 상자를 겹침(길을 막는 상자는 빨강).
사용: python prop_vs_line.py CKPT [--student S] [--density 1.5 --max-props 18] [--layouts 400] --out PNG
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl  # noqa: E402
from f1sim.learn.state_rl import ActorCritic, RunningNorm, StateObs  # noqa: E402

HALF_W = 0.135


def drive_path(a, device, ckpt, student):
    """the policy's centre path on the empty Korea track: (P, 2) points from 16 cars x ~2 laps"""
    a2 = argparse.Namespace(**vars(a)); a2.stage = "solo"; a2.warps = 0
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    sck = torch.load(student, map_location=device, weights_only=False) if student else None
    if sck is not None:
        a2.beams = int(sck["beams"])
        env = state_rl.build_env(a2, device, extra=dict(obs_version=1, scan_stack=int(sck["frames"]),
                                                        scan_stride=int(sck["args"].get("stride", 3)),
                                                        stagger_first_episode=False))
    else:
        a2.beams = int(ck["args"].get("beams", 36))
        env = state_rl.build_env(a2, device, extra=dict(stagger_first_episode=False))
    from f1sim.learn.graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    obs, _ = env.reset(seed=11)
    lid = env.learner_ids; B = int(lid.numel())
    if sck is not None:
        from f1sim.learn.state_student import Student
        st = Student(int(sck["frames"]), int(sck["beams"]), int(sck["ego_dim"])).to(device); st.load_state_dict(sck["student"]); st.eval()
        keys = sck.get("ego_keys") or sorted(k for k in obs if k != "scan")
        egov = lambda ob: torch.cat([ob[k][lid].float().reshape(B, -1) for k in keys], 1)
    else:
        sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)))
        last = torch.zeros(B, 2, device=device)
        o = sobs(last)
        net = ActorCritic(o.shape[1]).to(device); net.load_state_dict(ck["model"], strict=False); net.eval()
        nrm = RunningNorm(o.shape[1], device); nrm.load_state_dict(ck["norm"])
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    pts = []
    with torch.no_grad():
        for t in range(int(a.path_steps) // 2):
            if sck is not None:
                act = st(obs["scan"][lid].float(), egov(obs))
            else:
                act = net.pi(nrm(sobs(last))).clamp(-1, 1)
                last = act
            act_full[lid] = act
            for r in range(2):
                obs, _, te, trc, info = env.step(act_full)
            if t > 40:                                   # skip the launch
                pts.append(env.sim.state[lid, :2].cpu().numpy())
    track = env.sim.track.tracks[0]
    return np.concatenate(pts, 0), track


def layouts(a, device):
    """props of many layouts drawn by the training world's generator on Korea: list of (K, 2) polygons (world frame)"""
    a2 = argparse.Namespace(**vars(a)); a2.stage = "props"; a2.warps = 0; a2.beams = 36
    a2.prop_density = float(a.density); a2.prop_max = int(a.max_props)
    env = state_rl.build_env(a2, device, extra=dict(stagger_first_episode=False))
    polys = []
    n_per = 0
    seed = 100
    while len(polys) < int(a.layouts) * 3 and n_per < int(a.layouts):
        env.reset(seed=seed); seed += 1
        po = env.procedural
        for row in range(env.B):
            if n_per >= int(a.layouts):
                break
            n_per += 1
            live = (po.p_sid[row] >= 0).nonzero().flatten().tolist()
            for c in live:
                n = po.p_n[row, c]; d = po.p_d[row, c]
                ok = torch.isfinite(d)
                n, d = n[ok].cpu().numpy(), d[ok].cpu().numpy()
                # polygon corners: intersect consecutive half-plane boundaries (convex, ordered by angle)
                ang = np.arctan2(n[:, 1], n[:, 0]); o = np.argsort(ang); n, d = n[o], d[o]
                cs = []
                for i in range(len(n)):
                    j = (i + 1) % len(n)
                    A = np.array([n[i], n[j]]); bvec = np.array([d[i], d[j]])
                    if abs(np.linalg.det(A)) < 1e-9:
                        continue
                    cs.append(np.linalg.solve(A, bvec))
                if len(cs) < 3:
                    continue
                cs = np.array(cs)
                px, py, pyaw = po.p_poses[row, c].cpu().numpy()
                cc, ss = math.cos(pyaw), math.sin(pyaw)
                w = cs @ np.array([[cc, ss], [-ss, cc]]) + np.array([px, py])
                polys.append(w)
    return polys, n_per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--student", default="")
    ap.add_argument("--density", type=float, default=1.5)
    ap.add_argument("--max-props", type=int, default=18)
    ap.add_argument("--layouts", type=int, default=400)
    ap.add_argument("--path-steps", type=int, default=1400)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    path, track = drive_path(a, device, a.ckpt, a.student)
    polys, n_lay = layouts(a, device)
    from scipy.spatial import cKDTree
    tree = cKDTree(path)
    clear = []
    for P in polys:
        # dense samples along the polygon's edges; the distance from the path to the prop's surface
        edge = np.concatenate([np.linspace(P[i], P[(i + 1) % len(P)], 12) for i in range(len(P))], 0)
        dd, _ = tree.query(edge)
        inside = False
        clear.append(float(dd.min()) - HALF_W)
    clear = np.array(clear)
    block = clear < 0.0; graze = (clear >= 0) & (clear < 0.3)
    who = "학생 " + os.path.basename(a.student) if a.student else "선생 " + os.path.basename(a.ckpt)
    print(f"[길과 상자] {who}: 배치 {n_lay}개, 상자 {len(polys)}개 (배치당 {len(polys) / max(n_lay, 1):.1f}) | 빈 트랙의 길을 막는 상자 "
          f"{block.mean() * 100:.1f} % (배치당 {block.sum() / max(n_lay, 1):.2f}개), 스침 거리(0~0.3 m) {graze.mean() * 100:.1f} %, "
          f"여유 중앙 {np.median(clear):.2f} m", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    occ = track.occupancy; res = track.resolution; ox, oy = track.origin
    ax.imshow(occ, origin="lower", cmap="Greys", extent=(ox, ox + occ.shape[1] * res, oy, oy + occ.shape[0] * res), vmin=0, vmax=1.5,
              interpolation="nearest")
    show = min(len(polys), int(a.layouts) * 2)
    for i, P in enumerate(polys[:show * 4]):
        Q = np.vstack([P, P[:1]])
        ax.fill(Q[:, 0], Q[:, 1], color=("red" if block[i] else "darkorange"), alpha=(0.5 if block[i] else 0.15), lw=0)
    ax.plot(path[:, 0], path[:, 1], ".", color="royalblue", ms=0.6, alpha=0.5)
    m = np.isfinite(path[:, 0])
    ax.set_xlim(path[m, 0].min() - 1.5, path[m, 0].max() + 1.5); ax.set_ylim(path[m, 1].min() - 1.5, path[m, 1].max() + 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"{who}: 빈 트랙에서 타는 길(파랑)과 학습 세계 상자 배치 {n_lay}개(주황; 길을 막는 상자 빨강 {block.mean() * 100:.1f} %)\n"
                 f"상자 밀도 {a.density}, 최대 {a.max_props}개", fontsize=9)
    fig.tight_layout(); fig.savefig(a.out); plt.close(fig)
    np.savez(a.out.replace(".png", ".npz"), path=path, clear=clear)


if __name__ == "__main__":
    main()
