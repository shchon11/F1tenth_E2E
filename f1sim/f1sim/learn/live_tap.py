"""One car of a running training, written out every decision for a live view (11-17, the user: "콘솔에서 막 학습중인거
실시간으로 뭐 차 하나만 볼 수 없나", "학습중에 네트워크 반응이나 뭐 기억값같은거나 수치들 뭐 재밌게 볼 수 있게").

The console's '학습 중' entries drive the newest checkpoint in a world of their own -- the mean action, no exploration noise,
no scene replays, up to 25 updates late. This is the training's own car: what it did, with the noise it drove with, in the
episode it was in (a race from scratch or a replayed scene), against whoever drove the other car -- and what the network
was doing: the plan it meant (its mean) next to the plan it drove (mean + noise), the critic's value, the reward's parts,
its speed estimate, the noise it drew, the memory's 128 values.

`LiveTap(env, run_dir, row)` writes once `<run_dir>/live_tracks.npz` (the tracks' distance fields and the box footprints, to
draw on) and then, every `record`, `<run_dir>/live_tap.json` (atomic replace): the newest decision in full and a short trail
of the car (the page keeps its own history from what it polls). One small device->host copy a decision; nothing the
training reads or writes changes because of it.
"""
import json
import os
import time
from collections import deque

import numpy as np
import torch


def _r(a, n=3):
    return np.round(np.asarray(a, dtype=np.float64), n).tolist()


class LiveTap:
    def __init__(self, env, run_dir: str, row: int = 0, trail: int = 80, beam_every: int = 4):
        self.env, self.row = env, int(row)
        self.path = os.path.join(run_dir, "live_tap.json")
        self.run = os.path.basename(os.path.normpath(run_dir))
        self.trail = deque(maxlen=int(trail))
        lid = env.learner_ids
        self.r = int(lid[self.row])
        self.o = int(env.sim.other_idx[self.r, 0]) if getattr(env.sim, "other_idx", None) is not None and env.M > 1 else -1
        tr = env.sim.track
        fps = {}
        po = env.procedural
        if po is not None:
            from .. import props as _props
            for k, sh in enumerate(getattr(po, "shapes", []) or []):
                try:
                    fps[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
                except Exception:
                    pass
        np.savez(os.path.join(run_dir, "live_tracks.npz"), edt=tr.edt.float().cpu().numpy(), t_off=tr.t_off.cpu().numpy(),
                 t_H=tr.t_H.cpu().numpy(), t_W=tr.t_W.cpu().numpy(), t_res=tr.t_res.cpu().numpy(),
                 t_origin=tr.t_origin.cpu().numpy(), **fps)
        sim = env.sim
        self.beam_every = int(beam_every)
        self.angles = sim.lidar.angles.float()[::self.beam_every].cpu().numpy()
        try:
            self.mount_x = float(sim.P["mount_x"][self.r])
        except Exception:
            self.mount_x = 0.0
        self.hard0 = sim.contact_counts[self.r, 3:6].clone()
        self.prev_plan = None                 # the last decision's plan (world), for how much the new one changed
        self.prev_mem = None
        self.prev_xy = None

    @torch.no_grad()
    def record(self, upd: int, kd: int, pts: torch.Tensor, pts_mean: torch.Tensor, plan_v: torch.Tensor,
               value: torch.Tensor, reward: torch.Tensor, lap: torch.Tensor, contact: torch.Tensor, shaping: torch.Tensor,
               vhat: torch.Tensor, noise: torch.Tensor, std: torch.Tensor, mem=None, replay: bool = False,
               copy_opp: bool = False) -> None:
        """after a decision, learner `row`'s: `pts` (10, 2) the plan it drove (world), `pts_mean` (10, 2) the plan its mean
        action makes, `plan_v` (10,) the driven plan's speeds, the critic's `value`, the reward and its parts, the speed
        estimate `vhat`, the size of the noise drawn (`noise`, in std units) and the policy's mean std, `mem` (memory,) the
        memory after this decision. The car as it stands at the decision's end."""
        env, sim, r, o = self.env, self.env.sim, self.r, self.o
        dev = sim.state.device
        st = sim.state[r, :4].float()
        ost = sim.state[o, :4].float() if o >= 0 else torch.zeros(4, device=dev)
        present = (~sim.absent[o]).float().reshape(1) if (o >= 0 and getattr(sim, "absent_any", False)) else \
            torch.full((1,), float(o >= 0), device=dev)
        res = env.last_result
        rng = res.scan[r, ::self.beam_every].float() if getattr(res, "scan", None) is not None else torch.zeros(0, device=dev)
        hard = (sim.contact_counts[r, 3:6] - self.hard0).float()
        self.hard0 = sim.contact_counts[r, 3:6].clone()
        po = env.procedural
        box = torch.cat([po.p_sid[r].float()[:, None], po.p_poses[r].float()], 1).reshape(-1) if po is not None else \
            torch.zeros(0, device=dev)
        m = mem.float().reshape(-1) if mem is not None else torch.zeros(0, device=dev)
        sc = lambda x: torch.as_tensor(x, device=dev).float().reshape(-1)
        parts = [st, ost, present, sc(pts), sc(pts_mean), sc(plan_v), hard, sim.tid[r].float().reshape(1),
                 sc(value), sc(reward), sc(lap), sc(contact), sc(shaping), sc(vhat), sc(noise), sc(std), box, m, rng]
        sizes = [int(p_.numel()) for p_ in parts]
        flat = torch.cat(parts).cpu().numpy().astype(np.float64)
        out, i = [], 0
        for n_ in sizes:
            out.append(flat[i:i + n_]); i += n_
        car, opp, pres, plan, plan_m, pv, hit, tid, val, rew, lap_, con, shp, vh, noi, sd, bx, mm, rr = out
        a = self.angles + car[2]
        sx, sy = car[0] + self.mount_x * np.cos(car[2]), car[1] + self.mount_x * np.sin(car[2])
        ok = np.isfinite(rr) & (rr > 0.05) & (rr < 29.0)
        scan = np.stack([sx + rr * np.cos(a), sy + rr * np.sin(a)], 1)[ok]
        bx = bx.reshape(-1, 4)
        plan_xy = plan.reshape(10, 2)
        # a new episode: the car was put somewhere else (a reset, a scene) -- nothing to compare with
        new_ep = self.prev_xy is None or float(np.hypot(*(car[:2] - self.prev_xy))) > 2.0
        # the network's reaction: the new plan against the last one at the same moments (point k now = point k+1 then)
        plan_change = None if new_ep or self.prev_plan is None else \
            float(np.linalg.norm(plan_xy[:-1] - self.prev_plan[1:], axis=1).mean())
        mem_change = None if new_ep or self.prev_mem is None or not len(mm) else float(np.abs(mm - self.prev_mem).mean())
        self.prev_plan, self.prev_mem, self.prev_xy = plan_xy, (mm.copy() if len(mm) else None), car[:2].copy()
        if new_ep:
            self.trail.clear()
        self.trail.append(_r([car[0], car[1], car[3]], 2))
        now = {"t": round(time.time(), 3), "upd": int(upd), "k": int(kd), "tid": int(tid[0]),
               "car": _r(car), "opp": _r(opp) if pres[0] > 0.5 else None,
               "plan": _r(plan.reshape(10, 2)), "plan_mean": _r(plan_m.reshape(10, 2)), "plan_v": _r(pv, 2),
               "hit": [int(x) for x in hit], "boxes": [[int(b[0])] + _r(b[1:]) for b in bx if b[0] >= 0],
               "scan": _r(scan, 2), "replay": bool(replay), "copy": bool(copy_opp),
               "value": round(float(val[0]), 4), "reward": round(float(rew[0]), 5), "lap": round(float(lap_[0]), 5),
               "contact": round(float(con[0]), 5), "shaping": round(float(shp[0]), 5), "vhat": round(float(vh[0]), 3),
               "noise": round(float(noi[0]), 3), "std": round(float(sd[0]), 4), "new_ep": bool(new_ep),
               "plan_change": None if plan_change is None else round(plan_change, 4),
               "mem_change": None if mem_change is None else round(mem_change, 5)}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"run": self.run, "row": self.row, "now": now, "trail": list(self.trail)}, f, separators=(",", ":"))
        os.replace(tmp, self.path)
