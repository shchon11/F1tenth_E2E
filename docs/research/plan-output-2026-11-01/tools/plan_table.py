"""plan 학습 run의 고정 추이 표 (11-09, 사용자: "성능 표도 좀 띄엄띄엄 보내는것 같다? 표도 니 멋대로 만드는 것 같고"). 보고마다 이 표 두 개를
그대로 붙임 -- 행 = checkpoint(첫 행 = 시작점), 열은 언제나 같음.
 표 1 (25 update마다, analysis/<run>_summary.log): 학습 조건 빈 트랙 바퀴 / 같은 라인 최단과 차이 / 가속 사용(회전이 막지 않을 때) /
      브레이크 사용 / plan - 실제 0.5 s (자리, 속도) / 기억: 시야보다 더 맞힘 / 기억을 지우면 속도 출력 변화
 표 2 (100 update마다, out/one_<run>.log): 학습 조건 · 새 제어기 · MAP 각각 빈 트랙 바퀴 / 상자+상대 바퀴 / 세게 접촉 /분(네 경우 평균)
인자: run 이름"""
import re, sys
import numpy as np

D = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out"
run = sys.argv[1]


def analysis_rows(run):
    rows, cur = {}, None
    try:
        lines = []
        for fn in (f"{D}/analysis/{run}_start.log", f"{D}/analysis/{run}_summary.log"):     # the start point first
            try:
                lines += open(fn, errors="replace").read().splitlines()
            except OSError:
                pass
    except OSError:
        return rows
    for l in lines:
        m = re.match(r"== u(\d+)(.*)", l)
        if m:
            cur = int(m.group(1)); rows[cur] = {"tag": m.group(2).strip()}
            continue
        if cur is None:
            continue
        r = rows[cur]
        m = re.search(r"맞힘 기억 (\d+) % / 지금 특징 (\d+) %.*기억을 지우면 첫 단계 속도 출력 평균 ([\d.]+)", l)
        if m:
            r["mem_gain"] = int(m.group(1)) - int(m.group(2)); r["mem_drive"] = float(m.group(3))
        m = re.search(r"회전이 막지 않았는데.*사용 중앙 (\d+) %", l)
        if m:
            r["acc"] = int(m.group(1))
        m = re.search(r"내릴 때 브레이크 한계 대비: 중앙 (\d+) %", l)
        if m:
            r["brk"] = int(m.group(1))
        m = re.search(r"실제 바퀴 시간 중앙 ([\d.]+) s", l)
        if m:
            r["lap"] = float(m.group(1))
        m = re.search(r"-> ([\d.]+) s 빠름", l)
        if m:
            r["gap"] = float(m.group(1))
        m = re.search(r"0\.5 s: 속도 ([+-][\d.]+) .*?자리 ([\d.]+) cm", l)
        if m:
            r["t05v"], r["t05p"] = float(m.group(1)), float(m.group(2))
    return rows


def eval_rows(run):
    rows = {}
    cond = None
    try:
        lines = []
        for fn in (f"{D}/one_{run}_start.log", f"{D}/one_{run}.log"):
            try:
                lines += open(fn, errors="replace").read().splitlines()
            except OSError:
                pass
    except OSError:
        return rows
    for l in lines:
        m = re.search(r"u(\d+).*\[(ideal|track|map)\]", l)
        if m:
            cond = m.group(2); continue
        m = re.search(r"u(\d+)\s+(빈 트랙|상대 차|상자\+상대|상자) \d+대 \| .*한 바퀴 중앙 ([\d.]+) s \| 접촉 /분 벽 [\d.]+\(세게 ([\d.]+)\) 상자 [\d.]+\(세게 ([\d.]+)\) 차 [\d.]+\(세게 ([\d.]+)\)", l)
        if m and cond:
            u = int(m.group(1)); case = m.group(2)
            hard = float(m.group(4)) + float(m.group(5)) + float(m.group(6))
            rows.setdefault(u, {}).setdefault(cond, {})[case] = (float(m.group(3)), hard)
    return rows


f = lambda v, fmt: (fmt.format(v) if v is not None else "–")
A = analysis_rows(run)
print(f"**{run} 표 1** (25 update마다, 학습 조건 = plan대로 정확히, 원래 트랙)\n")
print("| checkpoint | 빈 트랙 바퀴 | 같은 라인 최단과 차이 | 가속 사용 | 브레이크 사용 | plan−실제 0.5초 | 기억: 시야보다 더 맞힘 | 기억이 속도 출력 바꾸는 크기 |")
print("|---|---|---|---|---|---|---|---|")
for u in sorted(A):
    r = A[u]
    lab = f"u{u}" + (f" ({r['tag']})" if r.get("tag") and not re.match(r"^\d\d:\d\d$", r["tag"]) else "")
    t05 = f"{r['t05p']:.0f} cm, {r['t05v']:+.2f} m/s" if "t05p" in r else "–"
    print(f"| {lab} | {f(r.get('lap'), '{:.2f}초')} | {f(r.get('gap'), '{:.2f}초')} | {f(r.get('acc'), '{} %')} | {f(r.get('brk'), '{} %')} | "
          f"{t05} | {f(r.get('mem_gain'), '{:+d} %p')} | {f(r.get('mem_drive'), '{:.3f}')} |")
E = eval_rows(run)
if E:
    print(f"\n**{run} 표 2** (100 update마다, 차 128대 × 40초: 빈 트랙 바퀴 / 상자+상대 바퀴 / 세게 접촉 /분 네 경우 평균)\n")
    print("| checkpoint | 학습 조건 | 물리 차 + 새 제어기 | 물리 차 + MAP |")
    print("|---|---|---|---|")
    for u in sorted(E):
        cells = []
        for c in ("ideal", "track", "map"):
            e = E[u].get(c, {})
            if not e:
                cells.append("–"); continue
            emp = e.get("빈 트랙", (None, None))[0]; both = e.get("상자+상대", (None, None))[0]
            hard = np.mean([v[1] for v in e.values()])
            cells.append(f"{f(emp, '{:.2f}')} / {f(both, '{:.2f}')}초 / 세게 {hard:.2f}")
        print(f"| u{u} | " + " | ".join(cells) + " |")
