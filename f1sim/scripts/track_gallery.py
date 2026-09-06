"""Render overview galleries of the track catalog (docs/tracks_*.png) and a JSON table.

    python3 scripts/track_gallery.py                 # all groups
    python3 scripts/track_gallery.py --group real    # one group

Each tile shows occupancy (grey walls / orange ducts), the auto or given centerline (cyan), the
start pose (red) plus loop length, median and minimum lane width (2 x clearance at the
centerline), lap direction and map extent."""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from f1sim import maps
from f1sim.track import Track

DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
GROUPS = {
    "gen_competition": ("gen:competition:  procedural, duct-hose lanes with hairpins / folded straights (odd seeds mirrored -> CW)",
                        [f"gen:competition:{i}" for i in range(12)], 6),
    "gen_other": ("gen:hallway / gen:circuit / +obs  (hallway = wide corridors, circuit = smooth wide loop, +obs = random lane boxes)",
                  [f"gen:hallway:{i}" for i in range(4)] + [f"gen:circuit:{i}" for i in range(4)] + ["gen:competition:5+obs", "gen:competition:6+obs"], 5),
    "real": ("real:  SLAM maps of physical venues", [f"real:{k}" for k in maps.REAL], 6),
    "rt": ("rt:  f1tenth_racetracks (scaled F1 circuits, ~1:10)", [f"rt:{os.path.basename(d)}" for d in sorted(glob.glob(os.path.join(maps.RACETRACKS, "*"))) if glob.glob(os.path.join(d, "*_map.yaml"))], 6),
}


def lane_stats(t: Track):
    cl = t.centerline
    rc = np.stack([(cl[:, 1] - t.origin[1]) / t.resolution, (cl[:, 0] - t.origin[0]) / t.resolution])
    d = ndimage.map_coordinates(t.edt, rc, order=1, mode="nearest")
    seg = np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1)
    area = 0.5 * np.sum(cl[:, 0] * np.roll(cl[:, 1], -1) - np.roll(cl[:, 0], -1) * cl[:, 1])
    return float(seg.sum()), float(np.median(2 * d)), float((2 * d).min()), "CCW" if area > 0 else "CW"


def draw(ax, t: Track, name: str):
    H, W = t.occupancy.shape
    ext = [t.origin[0], t.origin[0] + W * t.resolution, t.origin[1], t.origin[1] + H * t.resolution]
    img = np.full((H, W, 3), 0.92)
    img[t.occupancy] = (0.22, 0.24, 0.30)
    if t.duct is not None:
        img[t.duct] = (0.95, 0.55, 0.15)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    L, wmed, wmin, dirn = lane_stats(t)
    rr, cc = np.nonzero(~t.occupancy)                    # crop to the free space (SLAM canvases are mostly unknown)
    cl = t.centerline
    ax.plot(np.r_[cl[:, 0], cl[0, 0]], np.r_[cl[:, 1], cl[0, 1]], color="cyan", lw=0.8)
    ax.plot(cl[0, 0], cl[0, 1], "o", color="red", ms=3)
    ax.set_title(f"{name}\n{L:.0f} m, lane {wmed:.1f} m (min {wmin:.1f}), {(cc.max() - cc.min()) * t.resolution:.0f}x{(rr.max() - rr.min()) * t.resolution:.0f} m, {dirn}", fontsize=6)
    m = 1.0
    ax.set_xlim(t.origin[0] + cc.min() * t.resolution - m, t.origin[0] + cc.max() * t.resolution + m)
    ax.set_ylim(t.origin[1] + rr.min() * t.resolution - m, t.origin[1] + rr.max() * t.resolution + m)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return [name, L, wmed, wmin, dirn, (cc.max() - cc.min()) * t.resolution, (rr.max() - rr.min()) * t.resolution]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="*", default=list(GROUPS))
    ap.add_argument("--out", default=DOCS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info_path = os.path.join(a.out, "tracks_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    for g in a.group:
        title, names, ncol = GROUPS[g]
        nrow = int(np.ceil(len(names) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow), dpi=160)
        axes = np.atleast_1d(axes).ravel()
        rows = []
        for ax, n in zip(axes, names):
            try:
                t = maps.load(n)
                rows.append(draw(ax, t, n))
            except Exception as e:  # keep rendering the rest
                ax.set_title(f"{n}\nERROR {type(e).__name__}", fontsize=6, color="red")
                rows.append([n, None, None, None, None, None, None])
                print(f"{n}: {e}")
        for ax in axes[len(names):]:
            ax.axis("off")
        fig.suptitle(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"tracks_{g}.png"))
        plt.close(fig)
        info[g] = rows
        for r in rows:
            if r[1] is not None:
                print(f"{r[0]:28s} {r[1]:6.0f} m  lane {r[2]:.2f} (min {r[3]:.2f})  {r[4]:3s}  {r[5]:.0f}x{r[6]:.0f} m")
    json.dump(info, open(info_path, "w"), indent=1)


if __name__ == "__main__":
    main()
