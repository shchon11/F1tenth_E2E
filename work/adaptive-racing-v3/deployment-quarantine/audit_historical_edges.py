"""Document retained historical limitations, without altering its measured controller math."""
import json
from types import SimpleNamespace
from pathlib import Path
import torch
from f1sim import mpc
from f1sim.params import Config
from f1sim.learn.grip_runtime import automatic_grip_spec, DEFAULT_AUTO_PROFILE
from f1sim.learn.grip_control import GripMPC

report={'profile':DEFAULT_AUTO_PROFILE,'numeric_math_changed':False,'cases':{}}
for name,speed,wheel in [('rest_stop',0.,None),('crawl_stop',.1,None),('nonfinite_speed',float('nan'),None),('nonfinite_wheel',2.,float('nan'))]:
    tracker=mpc.PlanTracker(1,'cpu',.3302,.4189,10.,compile_solver=False)
    g=GripMPC(tracker,automatic_grip_spec(SimpleNamespace(cfg=Config())),1,'cpu',.3302,.4189,10.).install()
    a=torch.zeros(1,8);a[:,6:]=-1
    if wheel is not None:g.update_feedback(torch.tensor([wheel]))
    try:
        cmd=tracker(a,torch.tensor([speed]),torch.tensor([10.]))
        report['cases'][name]={'command':cmd.tolist(),'first_solver_accel':tracker.u_prev[:,1].tolist(),
            'reference_end_x':tracker.last_ref[:,-1,0].tolist(),'local_input_hook_installed':tracker._input_hook is not None,
            'local_fault_diagnostics_available':'input_fault' in g.last_diagnostics}
    except Exception as exc:
        report['cases'][name]={'error':type(exc).__name__,'message':str(exc)}
    finally:g.release()
report['scope']='Exact rest-stop, launch, finite fault-stop guarantees remain local-v2 only. Historical auto retains its pre-existing 0.3 m/s reference-walk floor and lacks the local sensor-input fault boundary. Wheel feedback is not consumed by its historical command conversion.'
print(json.dumps(report,indent=2))
Path('work/adaptive-racing-v3/deployment-quarantine/historical-edge-audit.json').write_text(json.dumps(report,indent=2))
