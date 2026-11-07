"""재설계 시험용 정해진 운전 규칙 (11-07): 학습 없이, 새 형식(plan_steps)의 plan을 고르는 단순한 규칙. 0.1 s마다 차마다 무작위 출력
K개(+ 직전 출력을 한 걸음 당긴 것)로 plan을 만들어, 보상이 plan을 매기는 그대로(plan 점 10개의 남은 거리 평균, 벽·상자를 지나가면 감점)
가장 좋은 것을 고름. 남은 거리 지도는 시뮬 정보라 policy는 못 보지만, 이 규칙은 형식·실행·보상이 맞물려 도는지 보는 시험이라 써도 됨.
plan대로 정확히 실행(env.enable_ideal)하며, gamma 1로 점수 합 = 바퀴 + 끝 Phi - 처음 Phi(정산)인지, 바퀴 시간, 접촉, plan 감점을 셈.
인자: 체크포인트(인자만) 차수 초 상자(0|1) [K]"""
import os, sys, math, argparse, time
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import PlanReward
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M

torch.set_grad_enabled(False)


def build(path, n_env, props, seed=5):
    dev = torch.device("cuda")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, seed, "cuda"
    a.prop_fraction = 1.0 if props else 0.0
    a.removal_p, a.opp_present, a.spawn_behind_prop, a.stage, a.episode_s = 0.0, 0.0, 0.0, "props", 3600.0
    a.prop_count = "uniform"
    env = SR.build_env(a, dev)
    env.enable_ideal()
    prepare_graph_runtime(env, log=lambda _t: None)
    env.reset(seed=seed)
    lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
    return env, lim, dev


def world(p, st):
    c, s_ = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
    x = st[:, 0:1] + p["x"] * c - p["y"] * s_
    y = st[:, 1:2] + p["x"] * s_ + p["y"] * c
    return x, y, st[:, dyn.IYAW:dyn.IYAW + 1] + p["psi"]


class Sampler:
    """pick, per car, the best of K plans by the reward's own plan terms"""

    def __init__(self, env, lim, pr, K, dev, seed=3):
        self.env, self.lim, self.pr, self.K, self.dev = env, lim, pr, K, dev
        self.g = torch.Generator(device=dev); self.g.manual_seed(seed)
        self.last = None

    def __call__(self, st):
        B = st.shape[0]; K = self.K
        cand = torch.rand(B, K, PS.ACT_DIM, device=self.dev, generator=self.g) * 2 - 1
        if self.last is not None:
            # the last choice moved on one step (its later steps held), so a good plan can be kept
            sh = torch.cat([self.last[:, 1:PS.N_STEP], self.last[:, PS.N_STEP - 1:PS.N_STEP],
                            self.last[:, PS.N_STEP + 1:], self.last[:, -1:]], 1)
            cand[:, 0] = sh
        flat = cand.reshape(B * K, -1)
        rep = lambda t: t.repeat_interleave(K, 0)
        p = PS.roll(flat, rep(st[:, dyn.IVX]), rep(st[:, dyn.ISTEER]), self.lim)
        x, y, psi = world(p, rep(st))
        lid = self.pr.lid
        # score each candidate as the reward would: Phi over its 10 points minus its price
        tid = self.env.sim.tid[lid].repeat_interleave(K, 0)
        L = self.pr.goal.lap[tid]
        idx = torch.arange(1, PS.N_STEP + 1, device=self.dev) * PS.SUB
        pts = torch.stack([x[:, idx], y[:, idx]], -1)
        d = self.pr.goal.remaining(pts, tid[:, None].expand(-1, PS.N_STEP))
        d = torch.where(torch.isfinite(d), d, L[:, None])
        # a plan whose points cross the line reads a lap more there: score the progress, not the jump
        d0 = self.pr.goal.remaining(st[:, :2], self.env.sim.tid[lid]).repeat_interleave(K, 0)
        d = torch.where(d > d0[:, None] + 0.5 * L[:, None], d - L[:, None], d)
        phi = -d.mean(1) / L
        # the price, as `PlanReward.plan_cost` (its rows are cars, so call it per K-block)
        c, s_ = torch.cos(psi)[..., None], torch.sin(psi)[..., None]
        cor = self.pr.corners
        cx = x[..., None] + cor[:, 0] * c - cor[:, 1] * s_
        cy = y[..., None] + cor[:, 0] * s_ + cor[:, 1] * c
        cp = torch.stack([cx, cy], -1)
        hit = (self.env.sim.track.sample_edt(cp, tid[:, None, None].expand_as(cx)) <= 0.0).any(-1)
        proc = getattr(self.env, "procedural", None)
        if proc is not None:
            eid = self.env.sim.eid[lid].repeat_interleave(K, 0)[:, None, None].expand_as(cx).reshape(-1)
            cl = proc.clearance(cp.reshape(-1, 2), eid).reshape(cx.shape)
            hit = hit | (cl <= 0.0).any(-1)
        hit[:, 0] = False
        score = phi - 10.0 * hit.any(1).float()
        best = score.view(B, K).argmax(1)
        self.last = cand[torch.arange(B, device=self.dev), best]
        return self.last


if __name__ == "__main__":
    path, n_env, secs, props = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1"
    K = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    env, lim, dev = build(path, n_env, props)
    pr = PlanReward(env, 1.0, dev)
    lid = env.learner_ids; B = int(lid.numel())
    pol = Sampler(env, lim, pr, K, dev)
    act0 = torch.zeros(env.B, env.act_dim, device=dev)
    phi0 = pr.phi.clone()
    tot = torch.zeros(B, device=dev); laps = torch.zeros(B, device=dev); price = torch.zeros(B, device=dev)
    lap_t, t_last = [], torch.full((B,), float("nan"), device=dev)
    stop = torch.zeros(B, device=dev)
    t0 = time.time()
    n_dec = int(secs / PS.DT)
    for kd in range(n_dec):
        st = env.sim.state[lid].clone()
        a_ = pol(st)
        p = PS.roll(a_, st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
        x, y, psi = world(p, st)
        pc = pr.plan_cost(x, y, psi, p["v"])
        env.sim.ideal_vcap[lid] = 1e3
        for j in range(PS.SUB):
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
            env.step(act0)
            pr.control_step()
        idx = torch.arange(1, PS.N_STEP + 1, device=dev) * PS.SUB
        out = pr.decision(torch.stack([x[:, idx], y[:, idx]], -1), pc["price"])
        tot += out["lap"] + out["shaping"]; laps += out["lap"]; price += pc["price"]
        tnow = (kd + 1) * PS.DT
        done = out["lap"] > 0
        for i in torch.nonzero(done)[:, 0].tolist():
            if not math.isnan(float(t_last[i])):
                lap_t.append(tnow - float(t_last[i]))
            t_last[i] = tnow
        stop += (env.sim.state[lid, dyn.IVX].abs() < 0.3).float() * PS.DT
    err = (tot - (laps + pr.phi - phi0)).abs()
    lt = torch.tensor(lap_t) if lap_t else torch.tensor([float("nan")])
    mins = secs * B / 60
    print(f"정해진 규칙(K {K}) {'상자 배치' if props else '빈 트랙'}, 차 {B}대 x {secs:.0f} s ({time.time() - t0:.0f} s 걸림) | 바퀴 {float(laps.sum()):.0f}개, "
          f"한 바퀴 중앙 {float(lt.median()):.2f} s (최소 {float(lt.min()):.2f}) | 접촉 벽 {float(pr.n_contacts[:, 0].sum()) / mins:.1f} 상자 "
          f"{float(pr.n_contacts[:, 1].sum()) / mins:.1f} /분 | plan 감점 결정 {float((price > 0).float().mean()) * 100:.0f} % 차 | 멈춘 시간 "
          f"{float(stop.mean()) / secs * 100:.0f} % | 정산 오차 최대 {float(err.max()):.1e}", flush=True)
