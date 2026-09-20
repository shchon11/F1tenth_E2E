import json,time,sys
from dataclasses import replace
from pathlib import Path
import torch
from f1sim import mpc
from f1sim.learn.grip_control import GripMPC,GripSpec
from f1sim.viewer.graph_fastpath import GuardViolation

torch.set_num_threads(1)
sp=mpc.PlanSpec();gs=GripSpec(mode='estimated',profile_version='local-v2',drive_split_r=.5)
results=[]
for B in (2,48):
    args_cpu=[torch.zeros(B,8),torch.linspace(2.,9.,B),torch.full((B,),10.),torch.zeros(B),
              torch.full((B,),.035),torch.zeros(B,2),torch.zeros(B,sp.N,2),sp,.3302,.4189,10.]
    args_cpu[0][:,6:]=.5
    args_cpu[0][:,:6]=torch.linspace(0.,.6,B)[:,None]
    mu=torch.linspace(.5,1.1,B)
    cpu=mpc.PlanTracker(B,'cpu',.3302,.4189,10.,sp,False)
    cg=GripMPC(cpu,gs,B,'cpu',.3302,.4189,10.).install();cg.update(mu)
    args=[a.cuda() if torch.is_tensor(a) else a for a in args_cpu]
    gpu=mpc.PlanTracker(B,'cuda',.3302,.4189,10.,sp,False)
    gg=GripMPC(gpu,gs,B,'cuda',.3302,.4189,10.);gg.update(mu.cuda())
    start=time.monotonic();gg.install(graph=True,example_args=args);torch.cuda.synchronize()
    capture=time.monotonic()-start
    with torch.no_grad():
        co=cpu._solver(*args_cpu);go=gpu._solver(*args)
        torch.cuda.synchronize()
        errors=[(c-g.cpu()).abs().max().item() for c,g in zip(co,go)]
        for c,g in zip(co,go):torch.testing.assert_close(c,g.cpu(),atol=3e-4,rtol=3e-4)
        for repeat in range(3):
            newmu=mu+.03*repeat;cg.update(newmu);gg.update(newmu.cuda())
            feedback=args_cpu[1]+.1*repeat
            cg.update_feedback(feedback);gg.update_feedback(feedback.cuda())
            cout=cpu(args_cpu[0],args_cpu[1],args_cpu[2],args_cpu[3])
            gout=gpu(args[0],args[1],args[2],args[3])
            torch.testing.assert_close(cout,gout.cpu(),atol=4e-4,rtol=4e-4)
            for name in cg._solver_diagnostic_names:
                torch.testing.assert_close(cg.last_diagnostics[name],gg.last_diagnostics[name].cpu(),atol=4e-4,rtol=4e-4)
            torch.testing.assert_close(cg.last_diagnostics['command_motor_request'],gg.last_diagnostics['command_motor_request'].cpu(),atol=4e-4,rtol=4e-4)
        for _ in range(5):gpu._solver(*args)
        torch.cuda.synchronize();start=time.monotonic()
        for _ in range(50):gpu._solver(*args)
        torch.cuda.synchronize();solver_ms=(time.monotonic()-start)*1000/50
        start=time.monotonic()
        for _ in range(30):gpu(args[0],args[1],args[2],args[3])
        torch.cuda.synchronize();tracker_ms=(time.monotonic()-start)*1000/30
        old=gg.last_diagnostics['v_reachable'];gg.last_diagnostics['v_reachable']=old.clone()
        guarded=False
        try:gpu._solver(*args)
        except GuardViolation:guarded=True
        assert guarded
        gg.last_diagnostics['v_reachable']=old
        gpu._solver(*args)
    result={'batch':B,'capture_seconds':capture,'max_cpu_gpu_errors':errors,'graph_solver_ms':solver_ms,
            'tracker_with_command_ms':tracker_ms,'diagnostic_rebind_guard':guarded,
            'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20,'mutable_mu_feedback_parity':True}
    print(json.dumps(result),flush=True);results.append(result)
    gg.release();cg.release();del gg,gpu;torch.cuda.empty_cache()
Path('work/adaptive-racing-v3/controller-proof/graph_results.json').write_text(json.dumps(results,indent=2))
