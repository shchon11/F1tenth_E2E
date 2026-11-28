"""The profiles through the LOADER path (yaml -> Config.from_dict -> the simulator's own parameter set) give the same
replay as the fitted override dicts: comp_car_map08x23 on the competition final bags, pre_car_map12x16 on 015428."""
import json, pickle
import numpy as np, torch
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN
from f1sim.profiles.loader import config_from_profile, unapplied
import evalcore as E

torch.set_num_threads(1)
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
Wof = lambda keys: WN.concat([D[k]["W"] for k in keys if k in D and len(D[k]["W"]["r"])])
for prof, keys in (("comp_car_map08x23", SPLIT["comp"]["final"]), ("pre_car_map12x16", ["20260725-015428"]),
                   ("default_comp_car_pre_place", ["20260725-015428"])):
    cfg = config_from_profile(prof)
    W = Wof(keys)
    pr = M.sim_rollout(W, cfg=cfg)
    T = E.table(W, {prof: pr, "시뮬 명목": M.sim_rollout(W)})
    a = T["all"][prof]
    print(prof, "on", keys, f"yaw1 {a['yaw'][1]:.3f} [p90 {a['yaw1_p90']:.3f}] pos2 {a['pos'][2]:.2f} spin {100 * a['spin']:.1f} %",
          "| nominal", f"{T['all']['시뮬 명목']['yaw'][1]:.3f} / {T['all']['시뮬 명목']['pos'][2]:.2f}", flush=True)
    print("   not applied:", len(unapplied(prof)), "entries", flush=True)
