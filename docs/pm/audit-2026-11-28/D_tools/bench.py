"""Simulator.step throughput (physics 25x1ms substeps + IMU + LiDAR 1081 beams) on the venue map."""
import sys, time, torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import maps
from f1sim.params import Config
from f1sim.sim import Simulator
dev = sys.argv[1]; Bs = [int(x) for x in sys.argv[2].split(",")]; secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10
leaves = len(sys.argv) > 4 and sys.argv[4] == "leaves"
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
tr = maps.load("real:korea_2026_competition")
if dev == "cuda": print(torch.cuda.get_device_name(0), flush=True)
for B in Bs:
    cfg = Config(); cfg.sim.device = dev; cfg.sim.compile = False; cfg.sim.compile_leaves = leaves
    cfg.sim.terminate_on_collision = False
    sim = Simulator(tr, cfg, num_envs=B, device=dev)
    sim.reset()
    act = torch.stack([torch.zeros(B), torch.full((B,), 3.0)], 1).to(sim.device)
    for _ in range(3): sim.step(act)
    if dev == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    n = 0; t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        act[:, 0] = 0.2 * torch.sin(torch.tensor(n * 0.05)); sim.step(act); n += 1
        if dev == "cuda" and n % 10 == 0: torch.cuda.synchronize()
    if dev == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = f" peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if dev == "cuda" else ""
    print(f"{dev} leaves={leaves} B={B:5d}: {n/dt:8.1f} steps/s  {B*n/dt:9.0f} env-steps/s  ({dt/n*1000:.1f} ms/step){mem}", flush=True)
    del sim
    if dev == "cuda": torch.cuda.empty_cache()
