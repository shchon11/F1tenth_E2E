import sys,json,time,traceback
from pathlib import Path
from f1sim import maps
from f1sim.raceline import Raceline
name=sys.argv[1];start=time.monotonic();out=Path('work/minimum-time-v4')/('verified-'+name.replace(':','_').replace('/','_'))
try:
 tr=maps.load(name);r=Raceline.build(tr);r.save(str(out)+'.csv');d=dict(map=name,ok=True,diagnostics=r.optimization,elapsed_s=time.monotonic()-start)
except Exception as e:
 d=dict(map=name,ok=False,error=str(e),elapsed_s=time.monotonic()-start);traceback.print_exc()
Path(str(out)+'.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
