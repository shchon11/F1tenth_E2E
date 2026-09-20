from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from f1sim import maps
from f1sim.raceline import Raceline
out=Path('/home/shchon11/Documents/Codex/2026-09-17/new-chat/outputs')
fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
for ax,name,file,title in zip(axes,['real:icra2022','scene:scene_0915_2026'],['verified-real_icra2022.csv','verified-scene_scene_0915_2026.csv'],['ICRA 2022','Authored scene: 3 asset obstacles']):
 t=maps.load(name);r=Raceline.load(Path('work/minimum-time-v4')/file);H,W=t.occupancy.shape;ox,oy=t.origin
 ax.imshow(np.ma.masked_where(~t.occupancy,t.occupancy),origin='lower',cmap='Greys',vmin=0,vmax=2,extent=(ox,ox+W*t.resolution,oy,oy+H*t.resolution),alpha=.6)
 for prop in t.props:
  c,s=np.cos(prop.yaw),np.sin(prop.yaw);xy=prop.build().envelope.footprint@np.array([[c,s],[-s,c]])+[prop.x,prop.y]
  ax.add_patch(Polygon(xy,facecolor='#ec9e37',edgecolor='#9a5a0a',lw=1.2,zorder=3))
 segments=np.stack([r.xy,np.roll(r.xy,-1,axis=0)],axis=1);line=LineCollection(segments,cmap='viridis',norm=plt.Normalize(0,10),linewidth=2.1,zorder=4);line.set_array(r.v);ax.add_collection(line)
 lo=r.xy.min(0)-1;hi=r.xy.max(0)+1;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
 m=r.optimization;ax.set_title(title+'\n'+f'Predicted lap {r.lap_time:.2f} s | KKT {m["kkt_stationarity"]:.1e}',fontsize=11)
 ax.spines[['top','right']].set_visible(False)
fig.colorbar(line,ax=axes,label='Planned speed [m/s]',shrink=.68)
fig.suptitle('Joint path + speed optimization; actual asset geometry included',fontsize=15)
fig.supxlabel('Converged quasi-steady local solutions. Full-body and axle constraints checked; not measured driving laps.',fontsize=10)
fig.savefig(out/'minimum-time-obstacles-v4.png',dpi=170)
