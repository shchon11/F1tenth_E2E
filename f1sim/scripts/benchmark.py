"""Throughput benchmark: env-steps/s for several batch sizes."""
import sys, time, torch
from f1sim import Track, Config, Simulator

tr = Track.generate_random(0)
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
for B in [1, 256, 1024, 4096]:
    sim = Simulator(tr, Config(), num_envs=B, device=dev)
    act = torch.stack([torch.zeros(B), torch.full((B,), 2.0)], 1).to(sim.device)
    for _ in range(5):
        sim.step(act)
    if sim.device.type == "cuda": torch.cuda.synchronize()
    n = 200 if B <= 1024 else 60
    t0 = time.perf_counter()
    for i in range(n):
        r = sim.step(act)
        if i % 20 == 0:
            sim.reset(torch.nonzero(r.collision).flatten())
    if sim.device.type == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"B={B:5d}  {n/dt:7.1f} steps/s  {B*n/dt:10.0f} env-steps/s  ({dt/n*1000:.2f} ms/step)"
          + (f"  mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if sim.device.type == "cuda" else ""))
