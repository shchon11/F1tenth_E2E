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


def _maps():
    seen = []
    for n in common.TRAIN_TRACKS + common.EVAL_TRACKS + [f"real:{k}" for k in common.maps.REAL] + [f"rt:{k}" for k in common.maps.racetrack_names()] + [f"gen:competition:{i}" for i in range(4)]:
        if n not in seen:
            seen.append(n)
    return seen


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
    mp = tk.StringVar(value=common.EVAL_TRACKS[0])
    line("map", ttk.Combobox(f, textvariable=mp, values=maps_, width=34))
    cars = tk.IntVar(value=32); line("cars", ttk.Spinbox(f, from_=1, to=1024, textvariable=cars, width=8))
    cap = tk.DoubleVar(value=6.0); line("speed cap [m/s]", ttk.Spinbox(f, from_=1.0, to=8.0, increment=0.5, textvariable=cap, width=8))
    race = tk.IntVar(value=1); line("cars per race (1 = no opponents)", ttk.Spinbox(f, from_=1, to=4, textvariable=race, width=8))
    opp = tk.StringVar(value="teacher"); line("opponents driven by", ttk.Combobox(f, textvariable=opp, values=["teacher", "policy"], width=12, state="readonly"))
    stoch = tk.BooleanVar(value=False); line("sample actions like training", ttk.Checkbutton(f, variable=stoch))
    internals = tk.BooleanVar(value=False); line("show raw network activations", ttk.Checkbutton(f, variable=internals))
    fast = tk.BooleanVar(value=False); line("run as fast as possible (no real-time pacing)", ttk.Checkbutton(f, variable=fast))
    gl = tk.StringVar(value="nvidia"); line("render the window on", ttk.Combobox(f, textvariable=gl, values=["nvidia", "amd"], width=12, state="readonly"))
    ttk.Label(f, text="'amd' draws through Mesa on the integrated Radeon so the NVIDIA GPU only runs the simulation", foreground="#666").grid(row=row[0], column=0, columnspan=2, sticky="w"); row[0] += 1
    ttk.Label(f, text="in the viewer:  C camera   [ ] other car   L lidar   T trails   S screenshot   space pause", foreground="#666").grid(row=row[0], column=0, columnspan=2, sticky="w", pady=(8, 0)); row[0] += 1
    out = {"args": None}
    def start():
        args = ["--run", run.get(), "--map", mp.get(), "--cars", str(cars.get()), "--speed-cap", str(cap.get()),
                "--race-size", str(race.get()), "--opponent", opp.get()]
        if stoch.get(): args.append("--stochastic")
        if internals.get(): args.append("--internals")
        if fast.get(): args.append("--fast")
        args += ["--gl", gl.get()]
        out["args"] = args; root.destroy()
    b = ttk.Frame(f); b.grid(row=row[0], column=0, columnspan=2, pady=(12, 0))
    ttk.Button(b, text="Start", command=start).grid(row=0, column=0, padx=6); ttk.Button(b, text="Cancel", command=root.destroy).grid(row=0, column=1, padx=6)
    root.bind("<Return>", lambda e: start()); root.mainloop()
    return out["args"]
