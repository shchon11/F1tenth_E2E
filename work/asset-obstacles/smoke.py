import json,time
from f1sim import maps, tracks
out={}
for mode in ('edge','line','hard'):
    name=tracks.asset_scenario('real/icra22#'+mode+':44')
    start=time.perf_counter();t=maps.load(name)
    pp=t.for_planning()
    out[mode]={'scenario':name,'props':len(t.props),'styles':sorted({p.style for p in t.props}),
        'size_variants':len({(p.style,p.dims) for p in t.props}),
        'planning_added_cells':int((pp.occupancy & ~t.occupancy).sum()),
        'seconds':time.perf_counter()-start,
        'grid_unchanged':bool((t.occupancy==maps.load('real/icra22').occupancy).all())}
print(json.dumps(out,indent=2));open('work/asset-obstacles/real-map-smoke.json','w').write(json.dumps(out,indent=2))
