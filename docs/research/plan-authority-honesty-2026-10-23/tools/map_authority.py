"""Research (2026-10-23): how much each of the plan's 8 numbers moves what the car EXECUTES -- under MAP (our follower
since 10-19) and under the iLQR the wp4 format was designed around (docs/research/output-format-2026-10-14.md, 4절 권한 표).

Hypothesis: MAP commands the speed of the waypoint it stands on (Korea round-5: speed_lookahead 0, steering speed = the
planned speed 0.06 s ahead) and steers at an L1 point ~2 m into a ~3 m plan, so v_2..v_4 reach the command only through
decode's backward braking pass and kappa_4 not at all -- the "general speed profile" would then be decorative, and PPO
has no gradient to shape it (the entry's profile: 0.143 m/s off a line, 0 % dips). Falsifier: v_2..v_4 authority on the
speed command under MAP >= 0.2 m/s per unit (the iLQR's v_4 was 0.24 in the 10-14 table).

States: every 20th learner step of CKPT in the finals solo protocol (Korea, rep floor, 32 cars x 30 s, MAP + dial 1.0489),
captured at the follower's input (action, v_meas, speed_cap, yaw_rate, delay) with MAP's own state (acc_now, curr_steer,
curv_wp). Authority of dim i: |cmd(a + h e_i) - cmd(a - h e_i)| / 2h, h 0.05, each call from the captured state.
usage: map_authority.py CKPT [OUT.json]"""
import json
import os
import sys

import torch

from f1sim import gym_env, map_controller as mc, mpc
from f1sim.learn.evaluate import evaluate
from f1sim.mpc import contract_of
from f1sim.params import Config

CK = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
TRACK = os.environ.get("AUTH_TRACK", "real:korea_2026_competition")
EVERY = int(os.environ.get("AUTH_EVERY", "20"))
H = 0.05
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extra = torch.load(CK, map_location="cpu", weights_only=False)["extra"]
contract = contract_of(extra)
MAPJ = dict(contract.get("plan_follower_map") or {})
rec = {k: [] for k in ("a", "v", "cap", "yaw", "delay", "acc", "steer0", "curv", "mu", "muf")}
calls = {"n": 0}
live = {}                                   # the env's own follower parts: its LUT, MAP numbers, nominal, tracker spec
orig = mc.MAPPlanTracker.__call__


def spy(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    if self.P is not None and bool(self.map_rows.any()):
        calls["n"] += 1
        if not live:
            live.update(lut=self.map.lut, p=self.map.p, nominal=self.nominal, spec=self._inner.spec,
                        v_max=float(self._inner.v_max), wb=float(self._inner.wb), s_max=float(self._inner.s_max))
        if calls["n"] % EVERY == 0:
            r = self.map_rows
            B = action.shape[0]
            dly = delay if torch.is_tensor(delay) else torch.full((B,), float(delay or 0.0), device=action.device)
            rec["a"].append(action[r].float().cpu()); rec["v"].append(v_meas[r].float().cpu())
            rec["cap"].append(speed_cap[r].float().cpu())
            rec["yaw"].append((torch.zeros(B, device=action.device) if yaw_rate is None else yaw_rate)[r].float().cpu())
            rec["delay"].append(dly[r].float().cpu())
            rec["acc"].append(self.map.acc_now[r].float().cpu()); rec["steer0"].append(self.map.curr_steer[r].float().cpu())
            rec["curv"].append(self.map.curv_wp[r].float().cpu())
            rec["mu"].append(self.P["mu"].reshape(-1)[r].float().cpu())
            rec["muf"].append(self.P["mu_f_scale"].reshape(-1)[r].float().cpu())
    return orig(self, action, v_meas, speed_cap, yaw_rate, delay)


mc.MAPPlanTracker.__call__ = spy
cfg = Config()
cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0)
cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
evaluate(CK, [TRACK], 32, 1200, 9.0, str(dev), seed=123, cfg=cfg, protocol="trials", budget_laps=3.0,
         plan_follower="map", plan_follower_map=MAPJ, dial_value=1.0489)
mc.MAPPlanTracker.__call__ = orig
S = {k: torch.cat(v).to(dev) for k, v in rec.items()}
N = S["a"].shape[0]
spec, V_MAX, lut, nominal, params = live["spec"], live["v_max"], live["lut"], live["nominal"], live["p"]
# AUTH_MAP_OVERRIDE='{"speed_lookahead": 0.235}': the same states, the follower changed (what a design option would do)
if os.environ.get("AUTH_MAP_OVERRIDE"):
    import dataclasses
    params = dataclasses.replace(params, **json.loads(os.environ["AUTH_MAP_OVERRIDE"]))


def inner():
    t = mpc.PlanTracker(N, dev, live["wb"], live["s_max"], V_MAX, spec=spec, compile_solver=False)
    t.reset(torch.arange(N, device=dev))
    return t


def map_cmd(a):
    tr = mc.MAPPlanTracker(inner(), lut, params, nominal)
    tr.map_rows = torch.ones(N, dtype=torch.bool, device=dev)
    tr.P = {"mu": S["mu"], "mu_f_scale": S["muf"]}
    tr.map.acc_now = S["acc"].clone(); tr.map.curr_steer = S["steer0"].clone(); tr.map.curv_wp = S["curv"].clone()
    out = tr(a, S["v"], S["cap"], S["yaw"], delay=S["delay"])
    return out, tr


def ilqr_cmd(a):
    return inner()(a, S["v"], S["cap"], S["yaw"], S["delay"])


base_map, tr0 = map_cmd(S["a"])
L = mpc.plan_length(S["v"], spec)
last = tr0.last_map
l1_frac = ((last["idx_near"].float() * mc.PLAN_WP_SPACING + last["l1"]) / L).clamp(max=5.0)
speed_frac = (last["idx_near"].float() * mc.PLAN_WP_SPACING / L)
names = [f"kappa{i + 1}" for i in range(4)] + [f"v{i + 1}" for i in range(4)] if spec.speed_mode == "wp4" else \
    [f"dim{i}" for i in range(S["a"].shape[1])]
bins = {"all": torch.ones(N, dtype=torch.bool, device=dev), "v<3": S["v"] < 3, "3-5": (S["v"] >= 3) & (S["v"] < 5),
        "v>=5": S["v"] >= 5}
res = {"ckpt": CK, "states": N, "track": TRACK, "map_params": MAPJ, "h": H, "override": os.environ.get("AUTH_MAP_OVERRIDE"), "v_max": V_MAX, "speed_mode": spec.speed_mode,
       "l1_point_over_plan": {"mean": float(l1_frac.mean()), "p10": float(l1_frac.quantile(0.1)),
                              "p90": float(l1_frac.quantile(0.9)), "beyond_plan_share": float((l1_frac > 1.0).float().mean())},
       "speed_read_over_plan": {"mean": float(speed_frac.mean()), "p90": float(speed_frac.quantile(0.9))},
       "v_meas": {"mean": float(S["v"].mean()), "p10": float(S["v"].quantile(0.1)), "p90": float(S["v"].quantile(0.9))},
       "map": {}, "ilqr": {}}
for fol, fn in (("map", lambda a: map_cmd(a)[0]), ("ilqr", ilqr_cmd)):
    for i, nm in enumerate(names):
        e = torch.zeros_like(S["a"]); e[:, i] = H
        up, dn = fn((S["a"] + e).clamp(-1, 1)), fn((S["a"] - e).clamp(-1, 1))
        d = (up - dn).abs() / (2 * H)
        res[fol][nm] = {b: {"steer_deg": round(float(d[m, 0].mean()) * 57.2958, 3), "v_cmd": round(float(d[m, 1].mean()), 3),
                            "n": int(m.sum())} for b, m in bins.items()}
    print(fol, {nm: (res[fol][nm]["all"]["steer_deg"], res[fol][nm]["all"]["v_cmd"]) for nm in names}, flush=True)
print("L1 point / plan length:", res["l1_point_over_plan"], "| speed read at / plan length:", res["speed_read_over_plan"],
      "| v_meas:", res["v_meas"], flush=True)
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
