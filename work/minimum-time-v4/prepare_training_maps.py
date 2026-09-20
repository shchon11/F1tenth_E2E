"""Build audited training lines and cache a previously verified identical ICRA solve."""
from pathlib import Path
import hashlib,inspect,json,os,time,sys
import numpy as np
from f1sim import maps
from f1sim.raceline import Raceline
from f1sim.params import VehicleParams
from f1sim.learn.common import heldout_leakage,HELDOUT_TRACKS
root=Path('work/minimum-time-v4')
names=sys.argv[1:]
assert not heldout_leakage(names,HELDOUT_TRACKS)
for name in names:
 t=maps.load(name); start=time.monotonic(); print('PREPARE',name,flush=True)
 # Existing ICRA artifact was built with exactly these defaults and has full
 # converged+dense+export diagnostics. Do not spend another3min solving it again.
 if name in ['real:icra2022','real:icra2022+bare']:
  verified=root/'verified-real_icra2022.json'; csv=root/'verified-real_icra2022.csv'
  if verified.exists() and json.loads(verified.read_text()).get('ok'):
   planned=t.for_planning();params={k:v.default for k,v in inspect.signature(Raceline.build).parameters.items() if k!='track'};params['vehicle']=VehicleParams()
   cl=np.asarray(planned.centerline,dtype=np.float32).tobytes();geometry=repr((planned.occupancy.shape,planned.resolution,tuple(planned.origin))).encode()
   key=hashlib.md5(np.packbits(planned.occupancy).tobytes()+cl+geometry+repr(sorted(params.items())).encode()+b'rl10-joint-minimum-time-swept-props-v1').hexdigest()[:12]
   folder=Path.home()/'.cache/f1sim/racelines';folder.mkdir(parents=True,exist_ok=True)
   target=folder/f'{planned.name}_{key}.csv';line=Raceline.load(csv);assert line.optimization['status']=='converged';line.save(target)
 line=Raceline.build_cached(t)
 record=dict(map=name,track=t.name,props=len(t.props),length_m=line.length,lap_time=line.lap_time,optimization=line.optimization,elapsed_s=time.monotonic()-start)
 key=hashlib.sha256(name.encode()).hexdigest()[:12];(root/f'training-{key}.json').write_text(json.dumps(record,indent=2));print('READY',json.dumps(record),flush=True)
