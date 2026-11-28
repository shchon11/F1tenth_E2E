"""Audit C: the exam3 layouts (same 64 layouts, same world coordinates) keyed under a held-out hose variant's track name, so
exam_record.py's line3 exam runs unchanged on that variant. Writes <out_dir>/heldout_hose<k>.pt and reports how many boxes sit
within 0.15 m of a wall there (a box moved into a hose would be an unfair layout). Args: k out_dir"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import common

k, out = int(sys.argv[1]), sys.argv[2]
os.makedirs(out, exist_ok=True)
base = "real:korea26_scan+duct+smooth"
src = torch.load("/home/shchon11/f1sim_runs/prop_banks_exam3_korea26scan/korea26scan_exam3.pt", map_location="cpu", weights_only=False)
per = src["per_track"]
print("bank tracks:", list(per.keys()), {kk: (v.shape if hasattr(v, "shape") else v) for kk, v in per[list(per.keys())[0]].items()})
trs, _ = common.load_tracks([f"{base}+hose{k}"], racelines=False)
tr = trs[0]
print("variant", tr.name)
key0 = list(per.keys())[0]
ent = dict(per[key0])
# length as the env measures it


L = float(os.environ.get("C_LEN", per[key0]["length"]))      # the env's lap length of the variant (its bank check says it)
print("base length", per[key0]["length"], "variant length", L)
ent["length"] = L
dst = dict(src); dst["per_track"] = {tr.name: ent}
torch.save(dst, os.path.join(out, f"heldout_hose{k}.pt"))
# boxes near the variant's walls
pose = ent["pose"].numpy().reshape(-1, 3); sid = ent["sid"].numpy().reshape(-1)
xy = pose[sid >= 0, :2]
occ = np.asarray(tr.occupancy); res = float(tr.resolution); org = np.asarray(tr.origin)[:2]
from scipy.ndimage import distance_transform_edt
free = occ == 0 if occ.max() <= 1 else occ < 50
edt = distance_transform_edt(free) * res
c = np.floor((xy[:, 0] - org[0]) / res).astype(int); r = np.floor((xy[:, 1] - org[1]) / res).astype(int)
ok = (r >= 0) & (r < edt.shape[0]) & (c >= 0) & (c < edt.shape[1])
d = np.where(ok, edt[r.clip(0, edt.shape[0] - 1), c.clip(0, edt.shape[1] - 1)], 0)
print(f"boxes {len(xy)}: centre within 0.15 m of a wall {int((d < 0.15).sum())}, within 0.25 m {int((d < 0.25).sum())}")
