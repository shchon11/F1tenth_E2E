from prototype import *
from f1sim import maps
import sys
name=sys.argv[1] if len(sys.argv)>1 else 'real:icra2022'
t=maps.load(name).for_planning(); seed=Raceline.build(t,optimize_lap_time=False).xy
print(name,'L',np.linalg.norm(np.roll(seed,-1,axis=0)-seed,axis=1).sum(),flush=True)
p=Problem(seed,t,margin=.4); print('variables',len(p.x0),'gmin',p.evaluate(p.x0)[1:].min(),flush=True);r=p.run();np.savez('work/minimum-time-v4/'+name.split(':')[-1]+'.npz',x=r.x,seed=seed)
