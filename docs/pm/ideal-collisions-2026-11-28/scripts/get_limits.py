"""이상 실행(ideal)에서 계획이 쓰는 차 한계(PS.Limits)를 checkpoint 설정 그대로 CPU에서 만들어 npz로 저장.
exam_record.py의 env 구성 부분만 따라 함 (환경 2개). 인자: ckpt out.npz"""
import os, sys, json, argparse
os.environ["F1SIM_NO_STAGGER"] = "1"
os.environ.pop("F1SIM_WORLD_TRACKS", None)
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
import numpy as np, torch
torch.set_grad_enabled(False)
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of
mx = lidar_mount_of(ck)
if mx > 0: os.environ["F1SIM_LIDAR_MOUNT_X"] = str(mx)
mg = (ck.get("args") or {}).get("plan_lat_margin")
if mg is not None: os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(mg)
from f1sim.learn import state_rl as SR
from f1sim import plan_steps as PS
from f1sim import map_controller as M
from world_of import world_path
W = world_path(ck)
a = argparse.Namespace(**json.load(open(W)))
if getattr(a, "stage", "race") != "race":
    a.stage, a.opp_present = "race", 0.0
    if getattr(a, "empty_seat", None):
        a.opp_slot = json.dumps({"kind": str(a.empty_seat), "kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0})
a.envs, a.seed, a.device = 2, 1, "cpu"
a.episode_s = 10.0; a.warps, a.track_scales = 0, None
a.prop_bank, a.prop_count = "", "max2"
env = SR.build_env(a, torch.device("cpu"))
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), torch.device("cpu"))
nom = env.tracker.nominal
out = dict(s_max=lim.s_max, sv_max=lim.sv_max, a_acc=lim.a_acc, a_brake=lim.a_brake, wheelbase=lim.wheelbase, lr=lim.lr,
           v_grid=lim.v_grid.numpy(), k_tyre=lim.k_tyre.numpy(), mount_x=mx, margin=float(mg) if mg is not None else 1.0,
           range_max=float(env.cfg.lidar.range_max), fov=float(env.cfg.lidar.fov), n_beams=int(env.cfg.lidar.n_beams),
           car_len=float(nom.get("length", 0.58)), car_w=float(nom.get("width", 0.31)),
           sev_min=float(getattr(env.ecfg, "contact_severity_min", float("nan"))))
np.savez(sys.argv[2], **out)
print({k: (v if np.ndim(v) == 0 else f"arr{np.shape(v)}") for k, v in out.items()})
ar = ck.get("args") or {}
print("args:", {k: ar.get(k) for k in ("gamma", "gae_lambda", "hard_contact_ends", "contact_cost", "contact_scale", "lap_bonus",
      "plan_lat_margin", "box_range", "scene_replay", "online_boxes", "online_late", "stall_cost", "reverse_cost", "plan_hit_cost",
      "noise_corr_box", "log_std_floor", "tail_target", "memory", "box_memory")})
