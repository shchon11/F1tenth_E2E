"""Debugging decoders of a plan policy's internal state (11-26, the user: "그 정보들이 네트워크에 있는지 디버깅용 디코더 ... 출력은
맵, 장애물 배치, ego state와 pose ... 시각화도 가능하게. 사이트나 콘솔에서도").

Read-only: a decoder reads what the frozen policy computed at a decision -- `feat` (the scan encoder's output and the normalised
state input, what the memory reads: `Actor2.features` with `mem_front`) and/or `mem` (the memory state after the decision,
`PlanObs2.mem_h`) -- and outputs, per decision:
  - pose: x, y, heading in the map frame
  - ego: forward speed, side speed [m/s], yaw rate [rad/s], steering [rad]
  - walls: an egocentric occupancy grid (`GRID_*`: x ahead -3..9 m, y left -6..6 m, 0.25 m cells), walls out of view included
  - boxes: the same grid, a box's zone (its outline grown by BOX_GROW)
  - the other car: P(within OPP_RANGE m), its position, heading, speed, and where it is 0.5 s and 1.0 s later (in the car's
    frame at the decision)
Nothing flows back into the policy: the decoders are trained afterwards on exam records (tools/state_decoder_train.py) and
only ever run under no_grad. Three input sets per checkpoint so information can be attributed: "feat" (the scan of the
moment), "mem" (the memory), "feat+mem".

The network: per input set, one two-layer MLP tower per output group (pose+ego 256, walls 512, boxes 512, other car 256) and a
linear head per output, on inputs standardised with the training decisions' mean / std.

File (`<record>_dec.npz`, and `plan_uN_dec.npz` beside the checkpoint for the console; numpy, no pickle): "pose_ext" (the
track's x0, x1, y0, y1), "ego_scale" (4,), "grid" (x0, x1, y0, y1, res), "towers" (JSON [[tower, [outputs]], ...]),
"hidden", "opp_scale", "box_grow", "checkpoint", "record", "has_opp", "version"; per input set S: "S|mu", "S|sd" and
"S|<state dict key>" (fp16). The site runs the same file in numpy (tools/dec_replay.py)."""
from __future__ import annotations

import json
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

GRID_X0, GRID_X1, GRID_Y0, GRID_Y1, GRID_RES = -3.0, 9.0, -6.0, 6.0, 0.25
NX = int(round((GRID_X1 - GRID_X0) / GRID_RES))          # 48 rows, along x (ahead)
NY = int(round((GRID_Y1 - GRID_Y0) / GRID_RES))          # 48 columns, along y (left)
N_CELL = NX * NY
OPP_RANGE = 10.0                                         # [m] the other car counts as there within this (the LiDAR's range)
OPP_SCALE = 5.0                                          # [m] the other car's coordinates are regressed / this
OPP_FUT = (5, 10)                                        # decisions ahead: 0.5 s, 1.0 s
BOX_GROW = 0.30            # [m] a cell is 'box' when its centre lies within the box's outline grown by this (a 0.2 m box ~ 3x3 cells)
SETS = ("feat", "mem", "feat+mem")
EGO_NAMES = ("vx", "vy", "r", "steer")
VERSION = 2
# one tower (two hidden layers) per group of outputs, so the groups do not compete for one body: a decoder reads each kind of
# information as well as it can on its own (11-26: with one shared body the other car's loss pulled the pose error from 0.2 to
# 0.9 m on a race record)
TOWERS = (("pe", ("pose", "ego"), 256), ("wall", ("wall",), 512), ("box", ("box",), 512), ("opp", ("opp",), 256))
HEAD_DIM = {"pose": 4, "ego": 4, "wall": N_CELL, "box": N_CELL, "opp": 10}
# pose: (x, y) / the track's extent, cos, sin; ego: vx, vy, r, steer / ego_scale; wall, box: logits a cell; opp: there logit,
# x, y / OPP_SCALE, cos, sin (its heading in our frame), speed / OPP_SCALE, x, y @0.5 s, x, y @1.0 s (/ OPP_SCALE)


def cell_centres() -> np.ndarray:
    """(N_CELL, 2) the cells' centres in the car's frame [m]: cell (ix, iy) at index ix * NY + iy, x ahead, y left"""
    xc = GRID_X0 + (np.arange(NX) + 0.5) * GRID_RES
    yc = GRID_Y0 + (np.arange(NY) + 0.5) * GRID_RES
    gx, gy = np.meshgrid(xc, yc, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)


class DecoderNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 512):
        super().__init__()
        k = hidden / 512.0
        w = {t: max(16, int(h * k)) for t, _, h in TOWERS}
        self.towers = nn.ModuleDict({t: nn.Sequential(nn.Linear(d_in, w[t]), nn.ReLU(), nn.Linear(w[t], w[t]), nn.ReLU())
                                     for t, _, _ in TOWERS})
        self.heads = nn.ModuleDict({o: nn.Linear(w[t], HEAD_DIM[o]) for t, outs, _ in TOWERS for o in outs})

    def forward(self, x: torch.Tensor, only=None) -> Dict[str, torch.Tensor]:
        out = {}
        for t, outs, _ in TOWERS:
            if only is not None and not any(o in only for o in outs):
                continue
            h = self.towers[t](x)
            for o in outs:
                out[o] = self.heads[o](h)
        return out


def inputs_of(feat: torch.Tensor, mem: torch.Tensor, which: str) -> torch.Tensor:
    if which == "feat":
        return feat
    if which == "mem":
        return mem
    if which == "feat+mem":
        return torch.cat([feat, mem], -1)
    raise ValueError(which)


def physical(o: Dict[str, torch.Tensor], pose_ext, ego_scale) -> Dict[str, np.ndarray]:
    """a decoder's raw outputs -> physical units (the keys present only): pose (n, 3) map x, y, yaw; ego (n, 4); wall, box
    (n, NX, NY) probabilities; opp_p (n,); opp (n, 4) x, y, heading, speed in the car's frame; opp_fut (n, 2, 2)"""
    x0, x1, y0, y1 = pose_ext
    out = {}
    if "pose" in o:
        p = o["pose"].float().cpu()
        out["pose"] = torch.stack([x0 + p[:, 0] * (x1 - x0), y0 + p[:, 1] * (y1 - y0), torch.atan2(p[:, 3], p[:, 2])], 1).numpy()
    if "ego" in o:
        out["ego"] = (o["ego"].float().cpu() * torch.as_tensor(np.asarray(ego_scale, np.float32))).numpy()
    for k in ("wall", "box"):
        if k in o:
            out[k] = torch.sigmoid(o[k].float()).cpu().view(-1, NX, NY).numpy()
    if "opp" in o:
        q = o["opp"].float().cpu()
        out["opp_p"] = torch.sigmoid(q[:, 0]).numpy()
        out["opp"] = torch.stack([q[:, 1] * OPP_SCALE, q[:, 2] * OPP_SCALE, torch.atan2(q[:, 4], q[:, 3]), q[:, 5] * OPP_SCALE], 1).numpy()
        out["opp_fut"] = (q[:, 6:10].reshape(-1, 2, 2) * OPP_SCALE).numpy()
    return out


class StateDecoder:
    """the three decoders of one checkpoint (a `_dec.npz`), in physical units"""

    def __init__(self, path: str, device="cpu"):
        z = np.load(path, allow_pickle=False)
        if int(z["version"]) != VERSION:
            raise ValueError(f"decoder file version {int(z['version'])} (this code reads {VERSION})")
        self.device = torch.device(device)
        self.hidden = int(z["hidden"])
        self.pose_ext = [float(v) for v in z["pose_ext"]]
        self.ego_scale = z["ego_scale"].astype(np.float32)
        self.feat_dim, self.mem_dim = int(z["feat_dim"]), int(z["mem_dim"])
        self.checkpoint, self.has_opp = str(z["checkpoint"]), bool(z["has_opp"])
        sets: Dict[str, dict] = {}
        for k in z.files:
            if "|" in k:
                s, n = k.split("|", 1)
                sets.setdefault(s, {})[n] = z[k]
        self.nets, self.norm = {}, {}
        for name, w in sets.items():
            mu, sd = w.pop("mu"), w.pop("sd")
            net = DecoderNet(int(mu.shape[0]), self.hidden)
            net.load_state_dict({k: torch.as_tensor(v.astype(np.float32)) for k, v in w.items()})
            net.eval().to(self.device)
            for p in net.parameters():
                p.requires_grad_(False)
            self.nets[name] = net
            self.norm[name] = (torch.as_tensor(mu, dtype=torch.float32, device=self.device),
                               torch.as_tensor(sd, dtype=torch.float32, device=self.device))

    @classmethod
    def load(cls, path: str, device="cpu") -> "StateDecoder":
        return cls(path, device)

    @torch.no_grad()
    def raw(self, feat, mem, which: str = "feat+mem", only=None) -> Dict[str, torch.Tensor]:
        net = self.nets[which]
        mu, sd = self.norm[which]
        x = inputs_of(torch.as_tensor(feat, dtype=torch.float32, device=self.device),
                      torch.as_tensor(mem, dtype=torch.float32, device=self.device), which)
        return net((x - mu) / sd, only)

    @torch.no_grad()
    def run(self, feat, mem, which: str = "feat+mem", batch: int = 4096) -> Dict[str, np.ndarray]:
        """feat (n, feat_dim), mem (n, mem_dim) -> `physical`'s dict"""
        feat = torch.as_tensor(np.asarray(feat, np.float32)); mem = torch.as_tensor(np.asarray(mem, np.float32))
        outs = []
        for i in range(0, feat.shape[0], batch):
            outs.append({k: v.cpu() for k, v in self.raw(feat[i:i + batch], mem[i:i + batch], which).items()})
        if not outs:
            return {}
        return physical({k: torch.cat([d[k] for d in outs], 0) for k in outs[0]}, self.pose_ext, self.ego_scale)


def save(path: str, nets: Dict[str, DecoderNet], norms: Dict[str, tuple], pose_ext, ego_scale, feat_dim: int, mem_dim: int,
         record: str = "", checkpoint: str = "", hidden: int = 512, has_opp: bool = False, extra: Optional[dict] = None) -> None:
    w = {"version": np.array(VERSION), "pose_ext": np.asarray(pose_ext, np.float32), "ego_scale": np.asarray(ego_scale, np.float32),
         "grid": np.array([GRID_X0, GRID_X1, GRID_Y0, GRID_Y1, GRID_RES], np.float32),
         "towers": np.array(json.dumps([[t, list(outs)] for t, outs, _ in TOWERS])), "hidden": np.array(int(hidden)),
         "opp_scale": np.array(OPP_SCALE), "box_grow": np.array(BOX_GROW), "checkpoint": np.array(str(checkpoint)),
         "record": np.array(str(record)), "has_opp": np.array(bool(has_opp)), "feat_dim": np.array(int(feat_dim)),
         "mem_dim": np.array(int(mem_dim))}
    for name, net in nets.items():
        mu, sd = norms[name]
        w[f"{name}|mu"], w[f"{name}|sd"] = np.asarray(mu, np.float32), np.asarray(sd, np.float32)
        for k, v in net.state_dict().items():
            w[f"{name}|{k}"] = v.detach().cpu().half().numpy()
    for k, v in (extra or {}).items():
        w[k] = np.asarray(v)
    with open(path, "wb") as f:                  # a file object: np.savez would append .npz to a name without it
        np.savez(f, **w)


def decoder_path_of(ckpt_path: str) -> str:
    """where the console looks for a checkpoint's decoder: `.../plan_uN.pt` -> `.../plan_uN_dec.npz`"""
    return (ckpt_path[:-3] if ckpt_path.endswith(".pt") else ckpt_path) + "_dec.npz"


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
