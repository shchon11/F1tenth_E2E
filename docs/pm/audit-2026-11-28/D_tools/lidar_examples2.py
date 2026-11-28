import json, numpy as np, torch, sys
sys.argv = ["x"]
import lidar_cmp as L
from f1sim import maps
from f1sim.track import TrackTensors
from f1sim.params import Config
from f1sim.randomization import ParamSet
from f1sim.lidar import Lidar
import sig
tr = maps.load("real:korea_2026_competition"); tt = TrackTensors(tr, "cpu"); cfg = Config()
ps = ParamSet(cfg, 1, torch.device("cpu"), torch.Generator().manual_seed(0)); li = Lidar(tt, cfg.lidar.n_beams, cfg.lidar.fov, "cpu", compile=False)
f = [x for x in sig.files("comp") if "20260827-115713" in x][0]; Z = np.load(f)
st, sc, pf, core = Z["scan_t"], Z["scan"], Z["pf"], Z["core"]
v = np.interp(st, core[:, 0], core[:, 2]) / 4204.0; yaw_u = np.unwrap(pf[:, 4])
out = []
for tq in (40.0, 61.0):
    i = np.flatnonzero((st > tq) & (v > 3.0))[0]
    rt, rn, typ, ci = L.cast(li, ps, tt, pf, yaw_u, np.array([st[i] + 0.02]), noisy=True)
    out.append(dict(t=float(st[i]), v=float(v[i]), real=sc[i].tolist(), sim=rn[0].tolist()))
json.dump(out, open("lidar_examples.json", "w"))
print([(o["t"], o["v"]) for o in out])
