from prototype import *
from f1sim import maps
name='real:icra2022';t=maps.load(name).for_planning();seed=Raceline.build(t,optimize_lap_time=False).xy
p=Problem(seed,t,margin=.4)
def restore(z):
 zt=torch.tensor(z,requires_grad=True);g=p.values(zt)[1:];f=torch.minimum(g,torch.zeros_like(g)).square().sum();f.backward();return float(f),zt.grad.numpy()
print('init groups',p.evaluate(p.x0)[1:].reshape(-1,p.N).min(1),flush=True)
r=minimize(restore,p.x0,method='L-BFGS-B',jac=True,bounds=p.bounds,options={'maxiter':1000,'ftol':1e-14,'gtol':1e-8,'maxls':50});print('restore',r.message,r.fun,p.evaluate(r.x)[1:].reshape(-1,p.N).min(1),flush=True);p.x0=r.x;p.run()
