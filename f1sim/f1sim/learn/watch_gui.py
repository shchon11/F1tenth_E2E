"""Launcher window for the viewer: pick the run (newest by default, auto-reloading), the map and the
view settings, press Start. Returns the argument list for f1sim.learn.watch.main."""
import glob
import os
from typing import Optional

from . import common


def _runs():
    out = []
    for d in glob.glob(os.path.join(common.RUNS_DIR, "*")):
        cks = [p for p in (os.path.join(d, "ppo_latest.pt"), os.path.join(d, "student_latest.pt")) if os.path.exists(p)]
        if cks:
            out.append((max(os.path.getmtime(p) for p in cks), os.path.basename(d)))
    return [n for _, n in sorted(out, reverse=True)]


def _map_sets():
    """Named groups for the picker, so "did it ever train on this?" is answered by which list the
    map came from rather than by remembering the catalog.

    The distinction is the whole point of looking at a map: a held-out track shows generalisation, a
    training track shows whether the thing was learnt at all, and mixing them in one flat list makes
    it easy to draw a conclusion from the wrong one. "+obs" hugs the lane edge (the racing line stays
    clear); "+rlobs" stands on the racing line; "+pinch" closes the lane down.
    """
    def uniq(seq):
        out = []
        for n in seq:
            if n not in out:
                out.append(n)
        return out
    catalog = ([f"real:{k}" for k in common.maps.REAL]
               + [f"rt:{k}" for k in common.maps.racetrack_names()]
               + [f"gen:{s}:{i}" for s in ("competition", "control", "serpentine", "circuit", "hallway")
                  for i in range(3)])
    return {
        "held-out  (never trained on)": uniq(common.EVAL_TRACKS),
        "held-out + obstacles": uniq(common.track_names("eval_obstacles")),
        "training set": uniq(common.TRAIN_TRACKS),
        "whole catalog": uniq(catalog),
    }


def _maps():
    sets_ = _map_sets()
    out = []
    for v in sets_.values():
        out += [n for n in v if n not in out]
    return out


def _cap_of(run_name: str) -> Optional[float]:
    """Training speed cap of a run, for the launcher's default."""
    from .watch import checkpoint_speed_cap
    import torch
    d = common.RUNS_DIR if run_name == "latest" else os.path.join(common.RUNS_DIR, run_name)
    if run_name == "latest":
        runs = _runs()
        if not runs:
            return None
        d = os.path.join(common.RUNS_DIR, runs[0])
    for name in ("ppo_latest.pt", "student_latest.pt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            try:
                extra = torch.load(p, map_location="cpu").get("extra", {})
                return checkpoint_speed_cap(extra, p, fallback=None) or None
            except Exception:
                return None
    return None


def ask() -> Optional[list]:
    import tkinter as tk
    from tkinter import ttk
    runs, maps_ = _runs(), _maps()
    root = tk.Tk(); root.title("f1sim viewer"); root.resizable(False, False)
    f = ttk.Frame(root, padding=14); f.grid()
    row = [0]
    def line(label, widget):
        ttk.Label(f, text=label).grid(row=row[0], column=0, sticky="w", pady=3, padx=(0, 10)); widget.grid(row=row[0], column=1, sticky="we", pady=3); row[0] += 1
    run = tk.StringVar(value="latest")
    line("policy run", ttk.Combobox(f, textvariable=run, values=["latest"] + runs, width=34))
    ttk.Label(f, text="'latest' follows the newest run; every run auto-reloads its newest checkpoint", foreground="#666").grid(row=row[0], column=0, columnspan=2, sticky="w"); row[0] += 1
    sets_ = _map_sets()
    set_names = list(sets_)
    which = tk.StringVar(value=set_names[0])
    mp = tk.StringVar(value=sets_[set_names[0]][0])
    set_box = ttk.Combobox(f, textvariable=which, values=set_names, width=34, state="readonly")
    line("which maps", set_box)
    map_box = ttk.Combobox(f, textvariable=mp, values=sets_[set_names[0]], width=34)
    line("map", map_box)
    count = ttk.Label(f, text="", foreground="#666")
    count.grid(row=row[0], column=0, columnspan=2, sticky="w"); row[0] += 1

    def on_set(*_):
        names = sets_[which.get()]
        map_box["values"] = names
        if mp.get() not in names:
            mp.set(names[0])
        count.config(text=f"{len(names)} maps in this set   |   editable: any catalog name works, "
                          f"e.g. real:korea_2026_competition+rlobs911~rev")
    which.trace_add("write", on_set); on_set()
    # Which sets are *loaded*, so the running viewer can be flipped between them without a restart.
    # Loading is the expensive part (about half a second a map, and its distance field on the GPU),
    # so the whole catalogue is not loaded on the off chance -- you tick what you want to be able to
    # reach, and the estimate below says what that costs before you press Start.
    avail = {n: tk.BooleanVar(value=(n == set_names[0])) for n in set_names}
    ttk.Label(f, text="sets to load (switchable while running, G in the viewer or the panel)").grid(
        row=row[0], column=0, sticky="nw", pady=3, padx=(0, 10))
    ab = ttk.Frame(f); ab.grid(row=row[0], column=1, sticky="w"); row[0] += 1
    cost = ttk.Label(f, text="", foreground="#666")
    cost.grid(row=row[0], column=0, columnspan=2, sticky="w"); row[0] += 1

    def loaded_names():
        out = [mp.get()] if mp.get() else []
        for n in set_names:
            if avail[n].get():
                out += [m for m in sets_[n] if m not in out]
        return out

    def on_avail(*_):
        k = len(loaded_names())
        cost.config(text=f"{k} maps will be loaded   |   about {k * 0.55:.0f} s to start"
                         f" and {k * 5:.0f} MB on the GPU")
    for i, n in enumerate(set_names):
        cb = ttk.Checkbutton(ab, text=f"{n}  ({len(sets_[n])})", variable=avail[n], command=on_avail)
        cb.grid(row=i, column=0, sticky="w")
    which.trace_add("write", lambda *_: (avail[which.get()].set(True), on_avail()))
    mp.trace_add("write", on_avail)
    on_avail()
    cars = tk.IntVar(value=1)
    line("parallel runs shown (not opponents)", ttk.Spinbox(f, from_=1, to=1024, textvariable=cars, width=8))
    cap = tk.DoubleVar(value=_cap_of("latest") or 6.0)
    line("speed cap [m/s]  (follows the run)", ttk.Spinbox(f, from_=1.0, to=10.0, increment=0.5, textvariable=cap, width=8))
    def _sync_cap(*_):
        c = _cap_of(run.get())
        if c:
            cap.set(c)
    run.trace_add("write", _sync_cap)
    race = tk.IntVar(value=1); line("cars per race (1 = no opponents)", ttk.Spinbox(f, from_=1, to=4, textvariable=race, width=8))
    opp = tk.StringVar(value="teacher"); line("opponents driven by", ttk.Combobox(f, textvariable=opp, values=["teacher", "policy"], width=12, state="readonly"))
    stoch = tk.BooleanVar(value=False); line("sample actions like training", ttk.Checkbutton(f, variable=stoch))
    internals = tk.BooleanVar(value=False); line("show raw network activations", ttk.Checkbutton(f, variable=internals))
    fast = tk.BooleanVar(value=False); line("run as fast as possible (no real-time pacing)", ttk.Checkbutton(f, variable=fast))
    compile_enabled = tk.BooleanVar(value=True); line("CUDA graphs (real-time; ~17 s longer to start)", ttk.Checkbutton(f, variable=compile_enabled))
    dr = tk.BooleanVar(value=True)
    line("randomise friction/delays per car (as in training)", ttk.Checkbutton(f, variable=dr))
    panel = tk.BooleanVar(value=True)
    line("keep a control panel open beside the viewer", ttk.Checkbutton(f, variable=panel))
    gl = tk.StringVar(value="nvidia"); line("render the window on", ttk.Combobox(f, textvariable=gl, values=["nvidia", "amd"], width=12, state="readonly"))
    ttk.Label(f, text="'amd' draws through Mesa on the integrated Radeon so the NVIDIA GPU only runs the simulation", foreground="#666").grid(row=row[0], column=0, columnspan=2, sticky="w"); row[0] += 1
    ttk.Label(f, text="in the viewer:  C camera   [ ] other car   M N track in set   G switch set   L lidar   T trails   S screenshot   space pause", foreground="#666").grid(row=row[0], column=0, columnspan=2, sticky="w", pady=(8, 0)); row[0] += 1
    out = {"args": None}
    def start():
        if cars.get() < race.get():                            # a race needs at least one full grid
            cars.set(race.get())
        chosen = loaded_names()
        # the picked map leads so the viewer opens on it; each set keeps its own membership, which is
        # what M / N walks -- the old build handed over the first ten maps of the flat union of every
        # set, so whichever set was picked, M / N wandered straight out of it
        sets_arg = ";".join(f"{n}={','.join(sets_[n])}" for n in set_names if avail[n].get())
        args = ["--run", run.get(), "--map", ",".join(chosen), "--cars", str(cars.get()), "--speed-cap", str(cap.get()),
                "--race-size", str(race.get()), "--opponent", opp.get(),
                "--map-sets", sets_arg, "--map-set", which.get()]
        if stoch.get(): args.append("--stochastic")
        if internals.get(): args.append("--internals")
        if fast.get(): args.append("--fast")
        if not compile_enabled.get(): args.append("--no-compile")
        if not dr.get(): args.append("--no-dr")
        args += ["--gl", gl.get()]
        out["args"] = args
        if panel.get():
            _open_panel({n: sets_[n] for n in set_names if avail[n].get()}, which.get())
        root.destroy()
    b = ttk.Frame(f); b.grid(row=row[0], column=0, columnspan=2, pady=(12, 0))
    ttk.Button(b, text="Start", command=start).grid(row=0, column=0, padx=6); ttk.Button(b, text="Cancel", command=root.destroy).grid(row=0, column=1, padx=6)
    root.bind("<Return>", lambda e: start()); root.mainloop()
    return out["args"]


def panel(sets_: dict, active: str) -> None:
    """The window that stays open beside the viewer: change the set M / N walks, or jump to a map.

    Its own process. Tk and the GL window each want to own the main loop, so sharing one process
    makes either the simulation stutter or the panel stop repainting. It writes what it wants to a
    file and the viewer reads it on its next frame, which is a millisecond's work at 30 fps.
    """
    import tkinter as tk
    from tkinter import ttk
    names = list(sets_)
    root = tk.Tk(); root.title("f1sim viewer — maps"); root.attributes("-topmost", True)
    f = ttk.Frame(root, padding=12); f.grid()
    ttk.Label(f, text="set that M / N walks").grid(row=0, column=0, sticky="w", padx=(0, 10))
    which = tk.StringVar(value=active if active in names else (names[0] if names else ""))
    ttk.Combobox(f, textvariable=which, values=names, width=30, state="readonly").grid(row=0, column=1, sticky="we")
    ttk.Label(f, text="jump to map").grid(row=1, column=0, sticky="w", pady=(8, 0), padx=(0, 10))
    mp = tk.StringVar(value="")
    box = ttk.Combobox(f, textvariable=mp, width=30, state="readonly")
    box.grid(row=1, column=1, sticky="we", pady=(8, 0))
    note = ttk.Label(f, text="", foreground="#666")
    note.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def on_set(*_):
        ms = sets_.get(which.get(), [])
        box["values"] = ms
        if ms:
            mp.set(ms[0])
        common.viewer_command(set=which.get())
        note.config(text=f"{len(ms)} maps in this set — M / N stays inside it")
    which.trace_add("write", on_set); on_set()
    mp.trace_add("write", lambda *_: common.viewer_command(set=which.get(), map=mp.get()))
    ttk.Label(f, text="only sets loaded at Start are here; the viewer is not restarted",
              foreground="#666").grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
    root.mainloop()


def _open_panel(sets_: dict, active: str) -> None:
    """Spawn the panel as its own process so its event loop cannot fight the viewer's."""
    import json
    import subprocess
    import sys
    subprocess.Popen([sys.executable, "-m", "f1sim.learn.watch_gui", "--panel",
                      json.dumps({"sets": sets_, "active": active})],
                     start_new_session=True)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--panel":
        d = json.loads(sys.argv[2]); panel(d["sets"], d["active"])
    else:
        print(ask())
