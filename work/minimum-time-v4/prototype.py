"""Temporary direct periodic path / energy NLP prototype."""
import time
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt
from f1sim.raceline import normals, speed_profile, track_widths, Raceline
from f1sim.track import resample_closed, Track
from f1sim.params import VehicleParams

class Problem:
 def __init__(self,seed,track,K=None,density=2,margin=.25,vehicle=None):
  self.p=vehicle or VehicleParams(); p=self.p; self.track=track
  L=np.linalg.norm(np.roll(seed,-1,axis=0)-seed,axis=1).sum()
  K=K or max(16,int(np.ceil(L/.8))); self.K=K; N=K*density;self.N=N
  ref=resample_closed(seed,K);normal=normals(ref)
  knots=np.arange(K+1)/K;query=np.arange(N)/N
  spline=CubicSpline(knots,np.vstack([np.eye(K),np.eye(K)[0]]),bc_type='periodic')
  self.b=[torch.tensor(spline(query,nu=i),dtype=torch.float64) for i in range(3)]
  self.ref=torch.tensor(ref);self.normal=torch.tensor(normal)
  # Piecewise-linear kinetic energy; independent positive nodal speeds.
  u=query*K; a=np.floor(u).astype(int); w=u-a; B=np.zeros((N,K)); B[np.arange(N),a]=1-w; B[np.arange(N),(a+1)%K]+=w
  self.vbasis=torch.tensor(B)
  self.edt=torch.tensor(track.edt.astype(float))
  wl,wr=track_widths(track,ref);reserve=np.minimum(margin,np.maximum(.12,.25*(wl+wr-p.width)))+p.width/2
  self.bounds=list(zip(-np.maximum(wr-reserve,0),np.maximum(wl-reserve,0)))+[(.01,1.)]*K
  self.x0=np.r_[np.zeros(K),(speed_profile(ref,a_lat=9.81*p.mu*p.mu_f_scale)/p.v_max)**2]
  self.radius=torch.tensor(B @ (reserve-p.width/2))+np.hypot(p.width/2,p.length/6)+track.resolution/np.sqrt(2)
  self.last=None; self.calls=0
 def field(self,q):
  t=self.track
  col=(q[...,0]-t.origin[0])/t.resolution; row=(q[...,1]-t.origin[1])/t.resolution
  i=torch.floor(row).to(torch.int64).clamp(0,self.edt.shape[0]-2);j=torch.floor(col).to(torch.int64).clamp(0,self.edt.shape[1]-2)
  a=row-i;b=col-j
  return self.edt[i,j]*(1-a)*(1-b)+self.edt[i+1,j]*a*(1-b)+self.edt[i,j+1]*(1-a)*b+self.edt[i+1,j+1]*a*b
 def values(self,z):
  p=self.p;K=self.K
  pts=self.ref+self.normal*z[:K,None]
  xy,d1,d2=[B@pts for B in self.b]
  norm=torch.linalg.vector_norm(d1,dim=1);tangent=d1/norm[:,None]
  curvature=(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])/norm**3
  ds=torch.linalg.vector_norm(torch.roll(xy,-1,0)-xy,dim=1)
  u=self.vbasis@z[K:]*p.v_max**2; un=torch.roll(u,-1);v=torch.sqrt(u);vn=torch.roll(v,-1)
  ax=(un-u)/(2*ds);dt=2*ds/(v+vn)
  delta=torch.atan((p.lf+p.lr)*curvature);dn=torch.roll(delta,-1)-delta
  c=[(p.s_max**2-delta**2)/p.s_max**2,1-(dn/(p.sv_max*dt))**2]
  for uu,kk in [(u,curvature),(un,torch.roll(curvature,-1))]:
   tire=ax+p.c_roll+p.c_drag*uu
   c += [(p.a_max-tire)/p.a_max,(p.a_max*p.v_switch-tire*torch.sqrt(uu))/(p.a_max*p.v_switch),(p.a_brake+tire)/p.a_brake]
   for share,load,tr,mu in [(1-p.drive_split_r,p.lr/(p.lf+p.lr),-p.h/(p.lf+p.lr),p.mu*p.mu_f_scale),(p.drive_split_r,p.lf/(p.lf+p.lr),p.h/(p.lf+p.lr),p.mu*p.mu_r_scale)]:
    fz=9.81*load+tr*ax;fx=share*tire;fy=load*uu*kk
    c += [fz/(9.81*load),((mu*fz)**2-fx**2-fy**2)/(mu*9.81*load)**2]
  for pos in [-p.length/3,0,p.length/3]: c += [self.field(xy+tangent*pos)-self.radius]
  return torch.cat([dt.sum().view(1),*c])
 def evaluate(self,z,jac=False):
  if self.last is None or not np.array_equal(self.last,z):
   self.last=z.copy();self.y=self.values(torch.tensor(z)).detach().numpy();self.J=None;self.calls+=1
  if jac and self.J is None:self.J=torch.func.jacfwd(self.values)(torch.tensor(z)).detach().numpy()
  return self.J if jac else self.y
 def run(self,maxiter=200):
  started=time.monotonic()
  r=minimize(lambda x:self.evaluate(x)[0],self.x0,jac=lambda x:self.evaluate(x,True)[0],method='SLSQP',bounds=self.bounds,constraints={'type':'ineq','fun':lambda x:self.evaluate(x)[1:],'jac':lambda x:self.evaluate(x,True)[1:]},options={'maxiter':maxiter,'ftol':1e-7,'disp':True})
  print('result',r.success,r.message,'time',time.monotonic()-started,'f',r.fun,'gmin',self.evaluate(r.x)[1:].min(),'calls',self.calls,flush=True)
  return r
if __name__=='__main__':
 theta=np.linspace(0,2*np.pi,240,endpoint=False);seed=6*np.c_[np.cos(theta),np.sin(theta)]; xx,yy=np.meshgrid(np.arange(-9,9,.05),np.arange(-9,9,.05));rad=np.hypot(xx,yy);t=Track.from_occupancy((rad<=4)|(rad>=8),.05,(-9,-9),seed)
 problem=Problem(seed,t);r=problem.run();print('offset',r.x[:problem.K].min(),r.x[:problem.K].max())
