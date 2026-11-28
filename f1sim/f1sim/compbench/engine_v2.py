"""Session engine for a contract-v2 checkpoint (`learn.model_v2`, the October wp4 policies) on the suite's sessions.

exam_record.py only drives plan_steps2 checkpoints, so this is a second loop with the SAME session definition, built
from the same pieces exam_record uses (nothing of them edited):
  - the world: `state_rl.build_env` on the suite's world json, as exam_record builds it (two seats, the second empty
    with the world's own empty-seat kind; one episode per car = secs + 5 s; the real map only; bank layouts "fixed";
    `F1SIM_NO_STAGGER`); the physics profile and the map swap as engine_exam
  - over it, the checkpoint's own contract, exactly as `learn.evaluate.evaluate` configures an env for it: observation
    v2 (frames, history, gyro scale, v_max), wp4 plans in "plan" action mode, the follower it was trained with (MAP and
    its 20 recorded settings), its speed cap, its held grip dial; its own LiDAR place (the default 0.297 m: the
    checkpoint's own sensor, the rule exam_record applies to every checkpoint)
  - the boxes: bank layout i -> race i % n, written into every car of the race after the reset (exam_record's LINE3
    block, all sessions boxed), removed at `remove_at` by `state_rl.PropRemoval` (exam_record's call)
  - the record: exam_record's fields the scorer reads (st, op, on, rs, cnt, ocnt, sid, pose, tid, rem, orem, lapL ...)
    every 0.1 s; the walls-only distance to go from the same `plan_reward.GoalField`
The policy acts every control step (its decision_every, 1), deterministic (mean action), memory cleared at a restart.
Usage: python -m f1sim.compbench.engine_v2 job.json"""
import os
import sys
import json
import time
import argparse


def main(job_path):
    job = json.load(open(job_path))
    os.environ["F1SIM_NO_STAGGER"] = "1"
    os.environ.pop("F1SIM_WORLD_TRACKS", None)
    os.environ.pop("F1SIM_LIDAR_MOUNT_X", None)
    os.environ.pop("F1SIM_PLAN_LAT_MARGIN", None)
    sys.path.insert(0, job["f1sim_src"])
    import numpy as np
    import torch
    torch.set_grad_enabled(False)
    if job["mode"] != "map":
        raise ValueError("a v2 checkpoint drives through its own follower: conditions real / stress only (no ideal)")
    dev = torch.device("cuda")
    print(f"[compbench] device {torch.cuda.get_device_name(0)}", flush=True)
    import f1sim.learn.common as C
    from f1sim.learn import state_rl as SR
    from f1sim.learn.model import load_checkpoint
    from f1sim.learn.conditioning import held_dial_of
    from f1sim.learn.watch import checkpoint_speed_cap
    from f1sim.learn.plan_reward import GoalField
    from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
    from f1sim import props as _props
    from .engine_exam import install_profile, install_map

    SEED, secs, rm_t = int(job["seed"]), float(job["secs"]), float(job["remove_at"])
    install_profile(job, C, torch)
    track = install_map(job, C)
    ck = job["driver"]["ckpt"]
    model, meta = load_checkpoint(ck, dev, allow_oracle=True, allow_conditional=True, allow_controller=False)
    model.eval()
    spec = meta.get("spec", {})
    assert int(spec.get("version", 1)) == 2, "engine_v2: a contract-v2 checkpoint"
    from f1sim.learn.obs import ObsSpecV2
    out_ = C.plan_output_of(meta)
    exp = meta.get("experiment") or {}
    cap = checkpoint_speed_cap(meta or {}, ck, fallback=10.0)
    extra = dict(action_mode="plan", obs_version=2, scan_stack=int(spec["n_frames"]), scan_stride=int(spec["frame_stride"]),
                 hist_len=int(spec["hist_len"]), hist_stride=int(spec["hist_stride"]), v_max_policy=float(spec["v_max"]),
                 imu_gyro_scale=float(spec["gyro_scale"]), ego_actions=bool(ObsSpecV2.from_dict(spec).ego_actions),
                 speed_cap=float(cap), plan_kappa_mode=str(exp.get("plan_kappa_mode", "absolute")),
                 plan_kappa_a_lat=float(exp.get("plan_kappa_a_lat", 8.0)),
                 speed_mode=out_["speed_mode"], speed_command=out_["speed_command"],
                 tracker_speed_weight=out_.get("tracker_speed_weight"), tracker_a_max=out_.get("tracker_a_max"),
                 tracker_profile_from_prediction=bool(out_.get("tracker_profile_from_prediction")),
                 tracker_speed_ff=bool(out_.get("tracker_speed_ff")),
                 plan_follower=out_.get("plan_follower", "mpc"), plan_follower_map=out_.get("plan_follower_map"),
                 decision_every=int(out_.get("decision_every", 1)), plan_speed_scale=1.0,
                 plan_a_brake=float(meta.get("plan_a_brake", 3.0)), compile_tracker=False, spawn_speed_max=None)
    extra = {k: v for k, v in extra.items() if v is not None}
    a = argparse.Namespace(**json.load(open(job["world"])))
    a.action = "direct"                     # the world's plan_rl plan fields stay off; the checkpoint's own come in `extra`
    if getattr(a, "stage", "race") != "race":
        a.stage, a.opp_present = "race", 0.0
        if getattr(a, "empty_seat", None):
            a.opp_slot = json.dumps({"kind": str(a.empty_seat), "kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0})
    a.envs, a.seed, a.device = int(job["n_env"]), SEED, dev.type
    a.episode_s = secs + 5.0
    a.warps, a.track_scales = 0, None
    a.prop_bank, a.prop_count = job["bank"], "fixed"
    a.lidar_mount_x = 0.0                   # the checkpoint's own scanner place (cfg default)
    torch.manual_seed(SEED)
    env = SR.build_env(a, dev, extra=extra)
    C.validate_policy_observation(spec, env)
    env.sim.warmup()
    dc = held_dial_of(meta)
    dial = (lambda: torch.full((env.B,), float(dc), device=env.device)) if dc is not None else None
    print(f"[compbench] v2 {os.path.basename(ck)}: follower {extra.get('plan_follower')}, cap {cap}, dial {dc}, "
          f"mount_x {float(env.cfg.lidar.mount_x):.3f}, beams {int(env.cfg.lidar.n_beams)}", flush=True)
    policy = C.student_policy(model, env, dev, dial=dial)
    obs, _ = env.reset(seed=SEED)
    if hasattr(policy, "reset"):
        policy.reset()
    lid = env.learner_ids
    Bl = int(lid.numel())
    po = env.procedural
    # exam_record's LINE3 + ALL_BOXES block: bank layout k -> the k-th race, in learner order, in every car of the race
    box_l = lid
    n_lay = int(po.b_n[0])
    k_l = torch.arange(box_l.numel(), device=dev) % n_lay
    t_l = env.sim.tid[box_l]
    sid_l, pose_l = po.b_sid[t_l, k_l], po.b_pose[t_l, k_l]
    mates = env.sim.other_idx[box_l]
    rows_ = torch.cat([box_l[:, None], mates], 1)
    src_ = torch.arange(box_l.numel(), device=dev)[:, None].expand_as(rows_)
    ok_ = rows_ >= 0
    dst_, src_ = rows_[ok_], src_[ok_]
    po.last_live.index_copy_(0, dst_, torch.zeros(box_l.numel(), po.P, dtype=torch.bool, device=dev)[src_])
    po._write(dst_, src_, pose_l[..., :2], pose_l[..., 2], sid_l.clamp_min(0), sid_l >= 0, P=po.C, Q=1)
    nk = (po.p_sid[box_l] >= 0).sum(1)
    print(f"[compbench] boxes: {Bl} races, layouts 0..{n_lay - 1} in order, {int(nk.min())}-{int(nk.max())} boxes a race", flush=True)
    remover = SR.PropRemoval(env, 1.0, rm_t, rm_t, seed=SEED)
    goal = GoalField(env.sim.track, dev)
    others = env.sim.other_idx[lid][:, 0]
    eid = env.sim.eid[lid]
    absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
    COLS = ("onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")
    I_CNT = [CC.index(k) for k in COLS]
    SUB = int(round(0.1 / float(env.sim.control_dt)))
    T = int(round(secs / 0.1))
    R = {k: [] for k in ("st", "op", "on", "rs", "cnt", "ocnt", "held", "sid", "pose", "tid", "rem", "orem", "vy")}
    i_hk, i_hx, i_hy, i_hs = (CC.index(k) for k in ("held_kind", "held_x", "held_y", "held_severity"))

    def snap():
        R["st"].append(env.sim.state[lid, :4].float().cpu()); R["op"].append(env.sim.state[others, :4].float().cpu())
        R["on"].append((~absent[others]).cpu())
        R["sid"].append(po.p_sid[eid].to(torch.int16).cpu()); R["pose"].append(po.p_poses[eid].float().cpu())
        R["tid"].append(env.sim.tid[lid].to(torch.int16).cpu())
        R["rem"].append(goal.remaining(env.sim.state[lid, :2].float(), env.sim.tid[lid]).float().cpu())
        R["orem"].append(goal.remaining(env.sim.state[others, :2].float(), env.sim.tid[others]).float().cpu())

    t0 = time.time()
    for kd in range(T):
        snap()
        R["vy"].append(env.sim.state[lid, 4].float().cpu())
        restarted = torch.zeros(Bl, dtype=torch.bool, device=dev)
        for _ in range(SUB):
            obs, _, term, trunc, _info = env.step(policy(obs))
            done = term | trunc
            if hasattr(policy, "reset"):
                policy.reset(done)
            restarted |= done[lid]
        remover.step()
        R["rs"].append(restarted.cpu())
        cf = env.sim.contact_frame()
        R["cnt"].append(cf[lid].float()[:, I_CNT].cpu()); R["ocnt"].append(cf[others].float()[:, I_CNT].cpu())
        R["held"].append(cf[lid].float()[:, [i_hk, i_hx, i_hy, i_hs]].cpu())
        if kd in (9, 99) or (kd + 1) % 300 == 0:
            print(f"  {(kd + 1) * 0.1:5.1f} s / {secs:g} s ({time.time() - t0:.0f} s)", flush=True)
    snap()
    D = {k: torch.stack(v).numpy() for k, v in R.items()}
    trk = env.sim.track
    D.update(edt=trk.edt.float().cpu().numpy(), t_off=trk.t_off.cpu().numpy(), t_H=trk.t_H.cpu().numpy(),
             t_W=trk.t_W.cpu().numpy(), t_res=trk.t_res.cpu().numpy(), t_origin=trk.t_origin.cpu().numpy(),
             car=np.array([float(env.cfg.vehicle.length), float(env.cfg.vehicle.width)]), dt=np.array(0.1),
             cols=np.array(COLS), name=np.array(f"{os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)}"),
             lapL=goal.lap.float().cpu().numpy(), mode=np.array("map"), count=np.array("line3"), rm_t=np.array(rm_t),
             seed=np.array(SEED), map=np.array(track), bank=np.array(job["bank"]), engine=np.array("compbench.engine_v2"),
             lidar_mount_x=np.array(float(env.cfg.lidar.mount_x)), exam=np.array("compbench-v2"))
    for k in sorted({int(s) for s in np.unique(D["sid"]) if s >= 0}):
        sh = po.shapes[k]
        D[f"fp_{k}"] = np.asarray(_props.build(sh.style, seed=0, **dict(sh.dims)).envelope.footprint, np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(job["out"])), exist_ok=True)
    np.savez_compressed(job["out"], **D)
    c = D["cnt"][-1].sum(0)
    print(f"[compbench v2] {D['name']}: {Bl} sessions x {secs:.0f} s (boxes removed at {rm_t:g} s), restarts {int(D['rs'].sum())} | "
          + " · ".join(f"{k} {int(v)}" for k, v in zip(COLS, c)) + f" | {time.time() - t0:.0f} s | {job['out']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
