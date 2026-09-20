from f1sim import maps
from f1sim.raceline import Raceline
from f1sim.minimum_time import _Problem
import numpy as np,sys
name=sys.argv[1] if len(sys.argv)>1 else 'real:icra2022'
t=maps.load(name).for_planning();seed=Raceline.build(t,optimize_lap_time=False).xy
L=np.linalg.norm(np.roll(seed,-1,axis=0)-seed,axis=1).sum();K=max(16,int(np.ceil(L/.8)));z=None
for d in [2,4,8,16]:
 p=_Problem(seed,t,dict(v_max=10.,mu=1.0489),.31,.4,None,K,d)
 r=p.solve(p.initial if z is None else z,250);z=r.x
 check=_Problem(seed,t,dict(v_max=10.,mu=1.0489),.31,.4,None,K,d*2)
 print('density',d,'f',r.fun,'groups',check.evaluate(z)[1:].reshape(-1,check.samples).min(1),flush=True)
 np.savez('work/minimum-time-v4/mesh-'+name.split(':')[-1]+'.npz',z=z,seed=seed,K=K,d=d)
