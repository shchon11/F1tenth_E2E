"""(11-25) plan_rl --arch pts: batch-1 latency of the point-token policy on this GPU (the lab's RTX 4060 Ti), and the Orin estimate.

What is timed (fp32, batch 1, 1081 beams, the default sizes; random scans with ~70 % returns):
  * net      -- from the tokens on: the queries, the two blocks, the GRU, the heads; the policy's AND the free-space head's (the
                policy reads the head's answer every decision, so both run on the car)
  * tokens   -- the points (5 frames dead-reckoned) and the per-point MLP / pooling into tokens, both networks
  * all      -- one whole decision as the car runs it: the observation's own work (the odometry chain, the frame shift, the
                anchors: `PtsObs.push_scan` + `pts_input`) + tokens + net
  * graph    -- `all` captured once in a CUDA graph and replayed (what TensorRT / a captured graph removes: the launch cost)
Also the parameters and FLOPs (torch's FlopCounterMode). Orin AGX: x4 the 4060 Ti's time, the probe's scale (~5 vs ~22 TFLOPS
fp32; a launch-bound net scales by the CPU / launch overhead instead, which `graph` bounds from below).
usage: python pts_latency.py [OUT.json]"""
import json
import math
import sys
import time
import types

import torch

from f1sim.learn import pts_policy as T

dev = torch.device("cuda")
torch.manual_seed(0)
BEAMS = 1081
ang = torch.linspace(-math.radians(135), math.radians(135), BEAMS)
actor = T.PtsActor(ang, (0.12555, 0.0, 0.0), 10.0, memory=128).to(dev).eval()
for p_ in actor.parameters():
    p_.requires_grad_(False)


def ms(fn, n=300, warm=30):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / n


# a fake one-car env for the observation's own work (odometry, scan)
ns = types.SimpleNamespace
env = ns(learner_ids=torch.arange(1, device=dev), device=dev,
         sim=ns(lidar=ns(angles=ang.to(dev)), cfg=ns(lidar=ns(range_max=10.0)), state=torch.zeros(1, 8, device=dev),
                odom=ns(state=torch.zeros(1, 5, device=dev))),
         last_result=ns(scan=torch.rand(1, BEAMS, device=dev) * 14.0, imu=torch.zeros(1, 2, 6, device=dev)))
from f1sim import map_controller as M
from f1sim import plan_steps as PS
lut, nominal = M.config_lut()
lim = PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), dev)
obs = T.PtsObs(env, lim)
for _ in range(6):                                   # a full history
    env.sim.odom.state[:, 0] += 0.5
    env.last_result.scan = torch.rand(1, BEAMS, device=dev) * 14.0
    obs.push_scan()
    obs.u_pending = torch.randn(1, 20, device=dev) * 0.3
    obs.set_plan2(PS.roll(torch.tanh(obs.u_pending), torch.full((1,), 4.0, device=dev), torch.zeros(1, device=dev), lim))
P = obs.pts_input()
s = obs.state()
h = torch.zeros(1, 128, device=dev)
s_n = actor.norm(s)

with torch.no_grad():
    # the token parts, both networks
    def tokens():
        X, Y, r, ok = actor.geom.points(P.frames, P.rel, P.fok)
        actor.core.front(X, Y, r, ok); actor.free.core.front(X, Y, r, ok)
    X, Y, r, ok = actor.geom.points(P.frames, P.rel, P.fok)
    tk_a = actor.core.front(X, Y, r, ok); tk_f = actor.free.core.front(X, Y, r, ok)

    def net():
        # the policy's and the head's queries, blocks, GRU, heads, from ready tokens (stage1 minus the token build)
        for core, tk in ((actor.free.core, tk_f), (actor.core, tk_a)):
            tok, pos, age, val = tk
            anc = P.anc
            u = torch.tanh(P.u_anc)
            okf = P.anc_ok.float()[:, None].expand(1, T.N_PQ)
            ext = torch.stack([torch.cos(anc[..., 2]), torch.sin(anc[..., 2]), P.anc_v / 5.0, okf, u[:, :T.N_PQ], u[:, T.N_PQ:]], -1)
            qp = core.q_time(core.tcode)[None] + core.q_anc(T.fourier(anc[..., :2])) + core.q_ext(ext)
            qe = core.q_ego(torch.cat([s_n, P.rel[:, 1] / core.inc_scale, P.fok[:, 1:2].float()], 1))
            q = core.q0[None] + torch.cat([qp, qe[:, None]], 1)
            a2 = torch.cat([anc[..., :2], torch.zeros(1, 1, 2, device=dev)], 1)
            q = core.blk1(q, a2, tok, pos, age, val)
            if core is actor.free.core:
                o = actor.free.out(core.stage2(q, (tok, pos, age, val, a2))[:, :T.N_PQ])
                fs = T.fs_feed(o[..., 0], o[..., 1])
            else:
                h1 = actor.gru(q[:, -1], h)
                actor.heads(actor.trunk(q, (tok, pos, age, val, a2), h1, fs, P.u_anc))

    def step_obs():
        obs.push_scan()
        return obs.pts_input()

    def all_():
        P_ = step_obs()
        actor(P_, obs.state(), h)

    t_tok, t_net, t_all = ms(tokens), ms(net), ms(all_)
    # a CUDA graph of the whole decision (fixed shapes: the token count is padded; the observation's tensors updated in place)
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(5):
            out_ = actor(obs.pts_input(), obs.state(), h)
    torch.cuda.current_stream().wait_stream(st)
    P_static = obs.pts_input()
    s_static = obs.state()
    with torch.cuda.graph(g):
        out_g = actor(P_static, s_static, h)
    t_graph = ms(lambda: g.replay())
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        actor(P, s, h)
    flops = float(fc.get_total_flops())
n_pol = sum(p_.numel() for p_ in actor.policy_parameters()); n_free = sum(p_.numel() for p_ in actor.free_parameters())
res = {"gpu": torch.cuda.get_device_name(0), "beams": BEAMS, "tokens": T.n_tokens(BEAMS), "params_policy": n_pol,
       "params_free_head": n_free, "mflop_decision": flops / 1e6, "ms_net": t_net, "ms_tokens": t_tok, "ms_all": t_all,
       "ms_graph": t_graph, "orin_x4": {k: 4.0 * v for k, v in (("net", t_net), ("tokens", t_tok), ("all", t_all), ("graph", t_graph))},
       "torch": torch.__version__}
print(json.dumps(res, indent=1))
print(f"[지연] batch 1, {res['gpu']}: 망만(토큰 뒤) {t_net:.2f} ms, 토큰 만들기 {t_tok:.2f} ms, 한 결정 전부(주행거리 연쇄 + 토큰 + 망) "
      f"{t_all:.2f} ms, CUDA graph로 한 결정 {t_graph:.2f} ms; 결정당 {flops / 1e6:.0f} MFLOP, 매개변수 policy {n_pol:,} + 빈 공간 머리 "
      f"{n_free:,}; Orin x4: {4 * t_all:.1f} ms (graph {4 * t_graph:.1f} ms)", flush=True)
if len(sys.argv) > 1:
    json.dump(res, open(sys.argv[1], "w"), indent=1)
