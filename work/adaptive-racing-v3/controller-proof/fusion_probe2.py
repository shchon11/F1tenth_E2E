import time,json,torch
from f1sim import mpc
from f1sim.learn import grip_control as gc

torch.set_num_threads(1)
sp=mpc.PlanSpec();gs=gc.GripSpec(mode='estimated',profile_version='local-v2',drive_split_r=.5)
kernels={}
for name,fn in [('project_control',gc.project_local_control),('reachable_step',gc._reachable_step)]:
 compiled=torch.compile(fn,fullgraph=True,dynamic=False,mode='default')
 def make_wrapper(compiled):
  @torch.no_grad()
  def call(*args):return compiled(*args)
  return call
 kernels[name]=make_wrapper(compiled)
for B in (2,48):
 a=[torch.zeros(B,8,device='cuda'),torch.full((B,),4.,device='cuda'),torch.full((B,),10.,device='cuda'),torch.zeros(B,device='cuda'),torch.full((B,),.035,device='cuda'),torch.zeros(B,2,device='cuda'),torch.zeros(B,sp.N,2,device='cuda'),sp,.3302,.4189,10.]
 t=mpc.PlanTracker(B,'cuda',.3302,.4189,10.,sp,False);g=gc.GripMPC(t,gs,B,'cuda',.3302,.4189,10.);g._kernels=kernels
 start=time.monotonic();g._bound(*a[:7],g.mu,*a[7:]);torch.cuda.synchronize();print('warm',B,time.monotonic()-start,flush=True)
 start=time.monotonic();g.install(graph=True,example_args=a);print('capture',B,time.monotonic()-start,flush=True)
 for _ in range(5):t._solver(*a)
 torch.cuda.synchronize();start=time.monotonic()
 for _ in range(30):t._solver(*a)
 torch.cuda.synchronize();print('ms',B,(time.monotonic()-start)*1000/30,flush=True)
 g.release()
