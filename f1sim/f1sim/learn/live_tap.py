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

(11-24, the user: the reward panel did not tell whether every term shows) `now` also carries:
* ``parts``: [{"n": name, "win": sum over the last 10 s, "ep": sum over the episode}, ...] -- one entry per term that enters
  this car's reward (`set_parts`: the trainer names them, Korean), the last one "합" (their sum = the reward the training
  used). A term the trainer can only price after the rollout (plan_rl --plan-car-cost: it needs the other car's next second)
  is added to the decisions it belongs to when it arrives (`settle`); ``parts_pending_s`` = the window's seconds still
  without it. ``parts_resid`` = the episode's reward minus the sum of the parts known per decision (~0: a check).
* ``pred``: the prediction head's answer for this car (plan_rl --pred-opp) in the world -- {"pts": 10 x [x, y], "p":
  P(visible)} -- from the pose the decision was made at (the console's `pred_w` transform); null without the head.
* ``opp_plan``: 10 x [x, y] in the world, the plan of the policy copy driving the other car this decision (its path; a
  slowed copy covers it at its pace factor); null when no copy drives it.
* ``car_dims`` / ``opp_dims``: [length, width] [m] of the two cars.
The old ``parts_ep`` / ``parts_win`` (lap, shaping, contact, plan hit, total) stay.
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
    def __init__(self, env, run_dir: str, row: int = 0, trail: int = 80, beam_every: int = 4, name: str = "live_tap.json",
                 label: str = ""):
        self.env, self.row = env, int(row)
        self.path = os.path.join(run_dir, name)
        self.label = str(label)
        # (11-22, the user: "항목별 보상값들이라던지 아니면 좀 기억관련 좀 직관적이게 볼 수 있는것들?") the reward's parts summed
        # over this episode and the last 10 s, and the episode's lap times -- the page polls slower than decisions come
        self.ep_sum = np.zeros(5); self.win = deque(maxlen=100); self.laps = deque(maxlen=6)
        # (11-24) the named parts (`set_parts`), per decision with its serial number so a late term finds its decisions
        self.part_names, self.late = None, None
        self.ent = deque(maxlen=100); self.ep2 = None; self.serial = 0; self.ep_first = 1; self.resid_ep = 0.0
        self.dec_ep, self.cross = 0, None
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
        # (11-18) the car's length and width: the page draws the plan's footprints and the half-width band with them
        car = np.array([float(env.cfg.vehicle.length), float(env.cfg.vehicle.width)])
        self.car_dims = _r(car, 3)
        if name == "live_tap.json":                                 # the drawing set once, by the first car's tap
            np.savez(os.path.join(run_dir, "live_tracks.npz"), edt=tr.edt.float().cpu().numpy(), t_off=tr.t_off.cpu().numpy(),
                     t_H=tr.t_H.cpu().numpy(), t_W=tr.t_W.cpu().numpy(), t_res=tr.t_res.cpu().numpy(),
                     t_origin=tr.t_origin.cpu().numpy(), car=car, **fps)
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

    def set_parts(self, names, late=None) -> None:
        """(11-24) the reward's terms by name, in the order `record(parts=...)` gives them; `late`: the name of one more term
        priced after the rollout (`settle`)"""
        self.part_names, self.late = list(names), late
        self.ent.clear(); self.ep2 = None

    def settle(self, values) -> None:
        """(11-24) the late term of this car's last len(values) decisions (the rollout just priced), as reward parts (a cost
        negative): added to the decisions still in the window and in this episode"""
        if self.late is None:
            return
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        s0 = self.serial - v.size + 1
        for e in self.ent:
            i = e[0] - s0
            if 0 <= i < v.size and not e[2]:
                e[1][-1] += v[i]; e[2] = True
        if self.ep2 is not None:
            first = max(s0, self.ep_first)
            if first <= self.serial:
                self.ep2[-1] += float(v[first - s0:].sum())

    @torch.no_grad()
    def record(self, upd: int, kd: int, pts: torch.Tensor, pts_mean: torch.Tensor, plan_v: torch.Tensor,
               value: torch.Tensor, reward: torch.Tensor, lap: torch.Tensor, contact: torch.Tensor, shaping: torch.Tensor,
               vhat: torch.Tensor, noise: torch.Tensor, std: torch.Tensor, mem=None, replay: bool = False,
               copy_opp: bool = False, mem_read=None, mem_pose=None, known=None, parts=None, pred=None, pred_pose=None,
               opp_plan=None, opp_live=None) -> None:
        """after a decision, learner `row`'s: `pts` (10, 2) the plan it drove (world), `pts_mean` (10, 2) the plan its mean
        action makes, `plan_v` (10,) the driven plan's speeds, the critic's `value`, the reward and its parts, the speed
        estimate `vhat`, the size of the noise drawn (`noise`, in std units) and the policy's mean std, `mem` (memory,) the
        memory after this decision. The car as it stands at the decision's end. (11-24) `parts` (n,) the reward's terms
        named by `set_parts`; `pred` (21,) the prediction head's raw output and `pred_pose` (3,) the pose the decision was
        made at; `opp_plan` (10, 2) the other car's copy's plan (world) and `opp_live` () whether a copy drives it."""
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
        e0 = torch.zeros(0, device=dev)
        extra = [sc(parts) if parts is not None else e0, sc(pred) if pred is not None else e0,
                 sc(pred_pose)[:3] if pred_pose is not None else e0, sc(opp_plan) if opp_plan is not None else e0,
                 sc(opp_live) if opp_live is not None else e0]
        parts_l = [st, ost, present, sc(pts), sc(pts_mean), sc(plan_v), hard, sim.tid[r].float().reshape(1),
                   sc(value), sc(reward), sc(lap), sc(contact), sc(shaping), sc(vhat), sc(noise), sc(std), box, m, rng] + extra
        sizes = [int(p_.numel()) for p_ in parts_l]
        flat = torch.cat(parts_l).cpu().numpy().astype(np.float64)        # still the one device->host copy a decision
        out, i = [], 0
        for n_ in sizes:
            out.append(flat[i:i + n_]); i += n_
        car, opp, pres, plan, plan_m, pv, hit, tid, val, rew, lap_, con, shp, vh, noi, sd, bx, mm, rr = out[:19]
        npart, pq, ppose, oplan, olive = out[19:]
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
        self.serial += 1
        if new_ep:
            self.trail.clear()
            self.ep_sum[:] = 0.0; self.win.clear(); self.laps.clear(); self.dec_ep, self.cross = 0, None
            self.ent.clear(); self.ep2 = None; self.ep_first = self.serial; self.resid_ep = 0.0
        # (11-24) the named parts: this decision's (+ a slot for the late term), the window's and the episode's sums
        named = None
        if self.part_names is not None and npart.size == len(self.part_names):
            vec = np.concatenate([npart, np.zeros(1 if self.late is not None else 0)])
            self.ent.append([self.serial, vec.copy(), False])
            self.ep2 = vec.copy() if self.ep2 is None else self.ep2 + vec
            self.resid_ep += float(rew[0] - npart.sum())
            win = np.sum([e[1] for e in self.ent], 0)
            names = self.part_names + ([self.late] if self.late is not None else [])
            named = [{"n": n_, "win": round(float(w_), 4), "ep": round(float(e_), 4)} for n_, w_, e_ in zip(names, win, self.ep2)]
            named.append({"n": "합", "win": round(float(win.sum()), 4), "ep": round(float(self.ep2.sum()), 4)})
        pend = (sum(1 for e in self.ent if not e[2]) * 0.1) if (named is not None and self.late is not None) else 0.0
        pred_js = None
        if pq.size >= 21 and ppose.size == 3:
            # the head's 10 points in the car's frame at the decision (x PRED_SCALE 5 m) and its 'visible' logit, in the world
            from .plan_policy import PRED_N, PRED_SCALE
            q = pq[:2 * PRED_N].reshape(PRED_N, 2) * PRED_SCALE
            c_, s_ = np.cos(ppose[2]), np.sin(ppose[2])
            wpts = np.stack([ppose[0] + q[:, 0] * c_ - q[:, 1] * s_, ppose[1] + q[:, 0] * s_ + q[:, 1] * c_], 1)
            pred_js = {"pts": _r(wpts, 3), "p": round(float(1.0 / (1.0 + np.exp(-pq[-1]))), 4)}
        opp_js = _r(oplan.reshape(10, 2), 3) if (oplan.size == 20 and olive.size == 1 and olive[0] > 0.5) else None
        self.trail.append(_r([car[0], car[1], car[3]], 2))
        # the reward's parts: lap bonus, distance shaping, contacts, the plan-hit price (what the total leaves), the total
        parts = np.array([lap_[0], shp[0], con[0], rew[0] - lap_[0] - shp[0] - con[0], rew[0]])
        self.ep_sum += parts; self.win.append(parts); self.dec_ep += 1
        if lap_[0] > 0.5:
            if self.cross is not None:
                self.laps.append(round((self.dec_ep - self.cross) * 0.1, 1))
            self.cross = self.dec_ep
        # the memory's read-out (`Actor2.aux_read`, 3 slots of present-logit, dx/5, dy/5, ds/5 in the car's frame at the
        # decision's start): where it thinks the boxes ahead are, in the world
        mem_boxes = None
        if mem_read is not None and mem_pose is not None:
            a_ = mem_read.float().reshape(-1).cpu().numpy(); x0, y0, h0 = (float(v_) for v_ in mem_pose.float().reshape(-1)[:3].cpu())
            mem_boxes = []
            for k_ in range(min(3, len(a_) // 4)):
                pk = 1.0 / (1.0 + np.exp(-a_[4 * k_])); dx_, dy_, ds_ = 5.0 * a_[4 * k_ + 1], 5.0 * a_[4 * k_ + 2], 5.0 * a_[4 * k_ + 3]
                if pk > 0.05:
                    mem_boxes.append([round(float(pk), 2), round(x0 + dx_ * np.cos(h0) - dy_ * np.sin(h0), 2),
                                      round(y0 + dx_ * np.sin(h0) + dy_ * np.cos(h0), 2), round(float(ds_), 1)])
        now = {"t": round(time.time(), 3), "upd": int(upd), "k": int(kd), "tid": int(tid[0]),
               "car": _r(car), "opp": _r(opp) if pres[0] > 0.5 else None,
               "plan": _r(plan.reshape(10, 2)), "plan_mean": _r(plan_m.reshape(10, 2)), "plan_v": _r(pv, 2),
               "hit": [int(x) for x in hit], "boxes": [[int(b[0])] + _r(b[1:]) for b in bx if b[0] >= 0],
               "scan": _r(scan, 2), "replay": bool(replay), "copy": bool(copy_opp),
               "value": round(float(val[0]), 4), "reward": round(float(rew[0]), 5), "lap": round(float(lap_[0]), 5),
               "contact": round(float(con[0]), 5), "shaping": round(float(shp[0]), 5), "vhat": round(float(vh[0]), 3),
               "noise": round(float(noi[0]), 3), "std": round(float(sd[0]), 4), "new_ep": bool(new_ep),
               "plan_change": None if plan_change is None else round(plan_change, 4),
               "mem_change": None if mem_change is None else round(mem_change, 5),
               "label": self.label, "parts_ep": _r(self.ep_sum, 3), "parts_win": _r(np.sum(self.win, 0), 3),
               "laps": list(self.laps), "ep_s": round(self.dec_ep * 0.1, 1), "mem_boxes": mem_boxes,
               "known": None if known is None else int(known),
               "parts": named, "parts_pending_s": round(pend, 1), "parts_resid": round(self.resid_ep, 6),
               "pred": pred_js, "opp_plan": opp_js, "car_dims": self.car_dims, "opp_dims": self.car_dims}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"run": self.run, "row": self.row, "now": now, "trail": list(self.trail)}, f, separators=(",", ":"))
        os.replace(tmp, self.path)
