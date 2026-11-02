"""The console's obstacle layout, held in a training generator's slots (2026-11-03).

`procedural_obstacles.ProceduralObstacles` draws a fresh layout at every full reset, from the simulator's own
generator. The console wants two other things of it (the user, 11-03: "왜 장애물 시드 못바꾸게해? ... 내가 직접
장애물을 추가/제거/위치/변경해서 바로 돌리게"):

* **a seed is a layout.** Seed N draws with a generator of its own (`draw_seed`), so the layout depends on the
  seed, the map and the run's generator settings -- not on how many cars race, the domain randomisation, or how
  many resets came before. It is cut to the slot count the run trained with (`keep`), which is exactly the layout
  a session of that many slots draws: `_write` fills the slots in one order whatever their number.
* **a layout someone made by hand** -- pieces added, removed, moved, turned, changed -- stands where it was put.

`LayoutHold` is both: installed on a session's generator, every full reset writes the held layout back instead of
drawing a new one (a 리셋 is the same race again), and `set_seed` / `set_pieces` change what is held.

Everything is written **in place** into the generator's own tensors (`p_sid`, `p_poses`, `p_n`, `p_d`, `p_zlo`,
`p_zhi`, `p_mass`, `p_vel`), with each piece's half-planes, height and mass from the catalogue, and a dead slot
written the way `PropRemoval` and a fresh generator write one. Everything that sees a prop reads those tensors --
the LiDAR, the contact test, the spawn rejection, the privileged observations, the prop-aware opponents -- and the
console's CUDA graphs replay against their addresses, so nothing is rebuilt and nothing is missed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .console import layout_file as LF


def catalogue(proc) -> List[Dict[str, Any]]:
    """The generator's shapes as the console draws and names them: id, style, dims, label, footprint (prop frame,
    metres), height, mass. The footprint is the declared envelope the collision and the LiDAR use."""
    from .. import props as _props
    small = set(int(i) for i in proc.small_id.tolist())
    out = []
    for i, sh in enumerate(proc.shapes):
        dims = {k: float(v) for k, v in sh.dims}
        fp = _props.build(sh.style, seed=0, **dims).envelope.footprint
        out.append({"id": i, "style": sh.style, "dims": dims, "label": LF.shape_label(sh.style, dims),
                    "footprint": [[float(x), float(y)] for x, y in fp], "height": float(sh.z1),
                    "mass": float(sh.mass), "small": i in small})
    return out


def run_slots(proc, max_props: int) -> int:
    """The slot count a run's generator has: its `procedural_max_props`, or the automatic one (0)."""
    from ..procedural_obstacles import PIECE_BUDGET
    if int(max_props) > 0:
        return int(max_props)
    return int(min(proc.P * proc.Q, max(6, round(proc.P * PIECE_BUDGET))))


def read(proc, row: int = 0) -> List[Dict[str, Any]]:
    """The live pieces of one env row, in slot order: [{"sid", "x", "y", "yaw"}]."""
    sid = proc.p_sid[row].tolist()
    pose = proc.p_poses[row].tolist()
    return [{"sid": int(s), "x": float(p[0]), "y": float(p[1]), "yaw": float(p[2])}
            for s, p in zip(sid, pose) if int(s) >= 0]


def resolve(proc, portable) -> List[Dict[str, Any]]:
    """Portable pieces (file, `SessionConfig.prop_layout`) -> this generator's shape ids. ValueError in Korean."""
    return LF.from_portable(portable, catalogue(proc))


@torch.no_grad()
def write(proc, pieces: Sequence[Dict[str, Any]], rows: Optional[torch.Tensor] = None) -> None:
    """Put `pieces` into the first slots of every row in `rows` (all rows when None) and kill the rest. In place."""
    C, dev = int(proc.C), proc.device
    n = len(pieces)
    if n > C:
        raise ValueError(f"장애물 {n}개는 이 세션의 슬롯 {C}개보다 많습니다")
    n_shapes = len(proc.shapes)
    for p in pieces:
        if not 0 <= int(p["sid"]) < n_shapes:
            raise ValueError(f"장애물 모양 번호 {p['sid']} 가 목록(0~{n_shapes - 1}) 밖입니다")
    rows = torch.arange(proc.B, device=dev) if rows is None else rows.to(dev).long()
    sid = torch.full((C,), -1, dtype=torch.long, device=dev)
    pose = torch.zeros(C, 3, device=dev, dtype=proc.p_poses.dtype)
    if n:
        sid[:n] = torch.tensor([int(p["sid"]) for p in pieces], dtype=torch.long, device=dev)
        pose[:n] = torch.tensor([[float(p["x"]), float(p["y"]), float(p["yaw"])] for p in pieces],
                                dtype=pose.dtype, device=dev)
    live = sid >= 0
    s = sid.clamp_min(0)
    pn = torch.where(live[:, None, None], proc.sh_n[s], torch.zeros_like(proc.sh_n[s]))
    pd = torch.where(live[:, None], proc.sh_d[s], torch.full_like(proc.sh_d[s], float("inf")))
    zlo = torch.where(live, proc.sh_z0[s], torch.zeros_like(proc.sh_z0[s]))
    zhi = torch.where(live, proc.sh_z1[s], torch.full_like(proc.sh_z1[s], -1.0))      # dead: z_hi < z_lo
    mass = torch.where(live, proc.sh_mass[s], torch.zeros_like(proc.sh_mass[s]))
    proc.p_sid[rows] = sid
    proc.p_poses[rows] = pose
    proc.p_n[rows] = pn.to(proc.p_n.dtype)
    proc.p_d[rows] = pd.to(proc.p_d.dtype)
    proc.p_zlo[rows] = zlo.to(proc.p_zlo.dtype)
    proc.p_zhi[rows] = zhi.to(proc.p_zhi.dtype)
    proc.p_mass[rows] = mass.to(proc.p_mass.dtype)
    proc.p_vel[rows] = 0.0


@torch.no_grad()
def draw_seed(proc, seed: int, keep: int, tid: torch.Tensor) -> List[Dict[str, Any]]:
    """The layout seed `seed` draws on track `tid` (1,), cut to `keep` slots, written into every row.

    The generator's own `redraw` with a generator seeded here, then the slots past `keep` killed: the first `keep`
    slots of a draw are the draw a `keep`-slot generator makes (`_write` compacts the pieces in one order and takes
    the first C), so the console's spare slots never put more crates on the track than the run trained with."""
    g = torch.Generator(device=proc.device)
    g.manual_seed(int(seed))
    saved = proc.gen
    proc.gen = g
    try:
        B = int(proc.B)
        type(proc).redraw(proc, tid.reshape(1).to(proc.device),
                          torch.arange(B, device=proc.device), torch.zeros(B, dtype=torch.long, device=proc.device))
    finally:
        proc.gen = saved
    keep = max(0, min(int(keep), int(proc.C)))
    sid = proc.p_sid[0].tolist()
    pose = proc.p_poses[0].tolist()
    pieces = [{"sid": int(s), "x": float(p[0]), "y": float(p[1]), "yaw": float(p[2])}
              for j, (s, p) in enumerate(zip(sid, pose)) if j < keep and int(s) >= 0]
    write(proc, pieces)
    return pieces


class LayoutHold:
    """One layout, held across every full reset of a session (see the module doc)."""

    def __init__(self, proc, keep: int, tid: torch.Tensor):
        self.proc = proc
        self.keep = int(keep)
        self.tid = tid.reshape(-1)[:1].clone()
        self.pieces: List[Dict[str, Any]] = []
        #: The seed the held layout was drawn from; None once it was set by hand.
        self.seed: Optional[int] = None
        self._installed = False

    # -- what is held
    def set_seed(self, seed: int) -> List[Dict[str, Any]]:
        self.pieces = draw_seed(self.proc, int(seed), self.keep, self.tid)
        self.seed = int(seed)
        return self.pieces

    def set_pieces(self, pieces: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pieces = [{"sid": int(p["sid"]), "x": float(p["x"]), "y": float(p["y"]), "yaw": float(p["yaw"])}
                  for p in pieces]
        write(self.proc, pieces)                       # raises before anything is held
        self.pieces = pieces
        self.seed = None
        return self.pieces

    # -- the hook
    def install(self) -> "LayoutHold":
        """Every redraw writes the held layout instead (an instance attribute shadows the method)."""
        self.proc.redraw = self._redraw
        self._installed = True
        return self

    def release(self) -> None:
        if self._installed:
            self.proc.__dict__.pop("redraw", None)
            self._installed = False

    def _redraw(self, draw_tid, dst, src):
        write(self.proc, self.pieces, rows=dst)

    # -- for the console
    def portable(self, cat: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        return LF.to_portable(self.pieces, cat if cat is not None else catalogue(self.proc))

    def state(self) -> Dict[str, Any]:
        """What the console is told after every change: the pieces, where they came from, the room left."""
        return {"layout": [dict(p) for p in self.pieces], "layout_seed": self.seed,
                "layout_custom": self.seed is None, "prop_slots": int(self.proc.C), "prop_keep": self.keep}
