"""포팅본(AuRoRus)의 MAP 제어기 설정을 지도별 설정 파일에 적음. 인자: 지도 이름  덮어쓸 값 JSON  출처 한 줄  [scale]
기존 파일은 날짜를 붙여 복사해 둠. 값 = 그 지도의 raceline/solo MAP 설정에 덮어쓸 값을 얹은 전체."""
import sys, json, shutil, time
P = "/home/shchon11/f1sim_runs/baseline_settings.json"
name, ov, src = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
scale = float(sys.argv[4]) if len(sys.argv) > 4 else None
d = json.load(open(P))
m = d["maps"][name]
base = dict(m["raceline"]["solo"]["stack_params"]["map"])
base.update(ov)
bak = P.replace(".json", time.strftime("_%m%d_%H%M.bak.json"))
shutil.copy2(P, bak)
opt = {"follower": "map", "map_params": base}
if scale is not None:
    opt["scale"] = scale
m["aurorus"] = {"solo": {"options": dict(opt), "source": src}, "race": {"options": dict(opt), "source": src}}
json.dump(d, open(P, "w"), indent=1, ensure_ascii=False)
print("백업", bak); print(json.dumps(m["aurorus"]["solo"], ensure_ascii=False)[:600])
