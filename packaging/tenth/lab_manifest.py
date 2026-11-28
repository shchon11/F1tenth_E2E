#!/usr/bin/env python3
"""연구실 PC로 보낼 서버 자료 목록: 설치본 library/가 가리키는 것 + M1 서버가 여는 캐시(2026-11-29 strace로 확인한 목록).

  .venv/bin/python packaging/tenth/lab_manifest.py <dist/TENTH>     # 표준 출력: "원본<TAB>server_home 안의 상대 경로" 한 줄씩

연구실 PC의 서버는 HOME=<server_home>으로 돈다(실행기 설정). 그래서 이 PC의 ~/X 는 그쪽의 <server_home>/X 로 보낸다.
  - 드라이버 체크포인트(library/drivers의 checkpoint.path / host_checkpoint)        ~/f1sim_runs/...
  - 기준선 설정, 절차적 상자 묶음(prop_banks_box2)                                     ~/f1sim_runs/...
  - 코스 지도: 지도 등록(~/.f1sim/maps/<이름>/), 팀 지도(external/... -> <server_home>/external/...)
  - 캐시: 중심선(~/.cache/f1sim/centerlines/<지도>_*.csv), 최소 시간 주행선(racelines/*<지도>*), MAP 추종 표(map_lut/*)
  - 상자 배치(library/courses의 layouts.saved.bank, layouts.random.refline: 절대 경로) -> <server_home>/banks/<이름>
    (설치할 때 library 사본의 그 경로를 <server_home>/banks 로 바꾼다: install.sh --bank-root)
GPU마다 다른 컴파일 캐시(inductor, triton, ~/.nv)는 보내지 않는다(그 PC에서 처음 한 번 만든다).
"""
import glob
import os
import sys

import yaml

DIST = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(DIST, "server"))
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("F1SIM_EXTERNAL", os.path.join(REPO, "external"))     # 이 PC의 팀 지도(저장소 external/)
HOME = os.path.expanduser("~")
out = []


def add(src, dst):
    if os.path.exists(src):
        out.append((src, dst))
    else:
        print(f"없음(건너뜀): {src}", file=sys.stderr)


def home_rel(p):
    p = os.path.realpath(os.path.expanduser(p))
    if not p.startswith(HOME + "/"):
        raise SystemExit(f"홈 밖의 경로라 보낼 자리를 정할 수 없습니다: {p}")
    return os.path.relpath(p, HOME)


lib = os.path.join(DIST, "library")
for fn in sorted(glob.glob(os.path.join(lib, "drivers", "*.driver.yaml"))):
    d = yaml.safe_load(open(fn)) or {}
    for p in ((d.get("checkpoint") or {}).get("path"), d.get("host_checkpoint")):
        if p:
            add(os.path.expanduser(p), home_rel(p))
add(os.path.join(HOME, "f1sim_runs", "baseline_settings.json"), "f1sim_runs/baseline_settings.json")
add(os.path.join(HOME, "f1sim_runs", "prop_banks_box2"), "f1sim_runs/prop_banks_box2")

bases = {}
for fn in sorted(glob.glob(os.path.join(lib, "courses", "*.course.yaml"))):
    c = yaml.safe_load(open(fn)) or {}
    lay = c.get("layouts") or {}
    for p in ((lay.get("saved") or {}).get("bank"), (lay.get("random") or {}).get("refline")):
        if p and os.path.isabs(p):
            add(p, "banks/" + os.path.basename(p.rstrip("/")))
    tr = str(c.get("track", ""))
    if tr.startswith("real:"):
        bases.setdefault(tr[5:].split("+")[0], set()).add(tr)

from f1sim import maps   # noqa: E402  (설치본 사본)
maps.refresh_registered()
for b in sorted(bases):
    ent = maps.REAL.get(b)
    if not ent:
        print(f"지도 표에 없음: {b}", file=sys.stderr)
        continue
    y = os.path.realpath(ent[0])
    ydir = os.path.dirname(y)
    if y.startswith(os.path.realpath(maps.EXTERNAL) + "/"):
        rel = os.path.relpath(y, os.path.realpath(maps.EXTERNAL))
        img = (yaml.safe_load(open(y)) or {}).get("image", "")
        add(y, "external/" + rel)
        if img:
            add(os.path.join(ydir, img), "external/" + os.path.join(os.path.dirname(rel), img))
    elif y.startswith(os.path.join(HOME, ".f1sim") + "/"):
        add(ydir, home_rel(ydir))
    # 패키지 안(assets/maps)의 지도는 서버 코드 사본에 이미 있다
    for f in glob.glob(os.path.join(HOME, ".cache", "f1sim", "centerlines", f"{b}_*.csv")):
        add(f, home_rel(f))
    for tr in sorted(bases[b]):                   # 최소 시간 주행선: 코스의 트랙 이름으로 시작하는 것만(+duct, +duct+smooth ...)
        for f in glob.glob(os.path.join(HOME, ".cache", "f1sim", "racelines", glob.escape(tr) + "*")):
            add(f, home_rel(f))
for f in glob.glob(os.path.join(HOME, ".cache", "f1sim", "map_lut", "*")):
    add(f, home_rel(f))

seen = set()
for s, d in out:
    if d not in seen:
        seen.add(d)
        print(f"{s}\t{d}")
