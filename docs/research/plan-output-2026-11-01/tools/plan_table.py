"""지금 학습의 표 (11-09/10, 사용자: "뭐이리 표가 난잡해. 지금 돌리고 있는 학습만 보여줘. 그리고 읽기 좀 쉽게 해라", "학습 환경 평가나 잘해.
제어기까지 돌린건 서브로 두고"). 보고마다 그대로 붙임 -- 지금 run만, 행만 늘어남.
본 표 = 학습 조건(plan대로 정확히 실행) 평가: 학습 세계 그대로, 고정 시드, 학습 차 512대 x 40 s(out/one_RUN_i.log, analysis_loop6이
25 update마다). 칸: 빈 트랙 한 바퀴 / 상자+상대 한 바퀴(중앙) / 세게 닿음 /분(빈 트랙·상대·상자·상자+상대 네 경우 평균; 벽+상자+차) /
기억이 더 아는 것(기억으로 '이 트랙에 상자가 있나'를 맞힌 비율 - 지금 보이는 것으로 맞힌 비율, analysis/RUN_summary.log).
서브 표(--sub일 때만) = 실제 차(물리 + 새 제어기) 100 update 평가(out/one_RUN.log, 학습 차 128대) -- 같은 세 칸. 사용자 11-10:
"학습이 잘되고있고 안되고를 봐야지 궤적을 못따라가고있는건 별도로 분석해야하는거고" -> 정기 보고에는 본 표만.
첫 행 = 시작(이어 받은 checkpoint, 앞 run의 같은 기록에서).
인자: run 이름 [--sub]"""
import re, sys
import numpy as np
from run_chain import STAGE_NOTE, parent

D = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out"


def read_lines(*fns):
    lines = []
    for fn in fns:
        try:
            lines += open(fn, errors="replace").read().splitlines()
        except OSError:
            pass
    return lines


def memory_rows(lines):
    rows, cur = {}, None
    for l in lines:
        m = re.match(r"== u(\d+)", l)
        if m:
            cur = int(m.group(1)); continue
        m = re.search(r"맞힘 기억 (\d+) % / 지금 특징 (\d+) %", l)
        if m and cur is not None:
            rows[cur] = int(m.group(1)) - int(m.group(2))
    return rows


def eval_rows(lines, want):
    """one condition (ideal / track / map) of plan_validate lines: {u: {case: (lap median, hard contacts /min)}}"""
    rows, cond = {}, None
    for l in lines:
        m = re.search(r"u(\d+).*\[(ideal|track|map)\]", l)
        if m:
            cond = m.group(2); continue
        m = re.search(r"u(\d+)\b[^|]*?\s(빈 트랙|상대 차|상자\+상대|상자) \d+대 \| .*한 바퀴 중앙 ([\d.na]+) s \| 접촉 /분 벽 [\d.]+\(세게 ([\d.]+)\) "
                      r"상자 [\d.]+\(세게 ([\d.]+)\) 차 [\d.]+\(세게 ([\d.]+)\)", l)
        if m and cond == want:
            hard = float(m.group(4)) + float(m.group(5)) + float(m.group(6))
            rows.setdefault(int(m.group(1)), {})[m.group(2)] = (float(m.group(3)), hard)
    return rows


def cells(q):
    emp = q.get("빈 트랙", (None,))[0]; both = q.get("상자+상대", (None,))[0]
    hard = float(np.mean([v[1] for v in q.values()])) if q else None
    f = lambda v, fmt: (fmt.format(v) if v is not None and v == v else "–")
    return f"{f(emp, '{:.2f}초')} | {f(both, '{:.2f}초')} | {f(hard, '{:.2f}')}"


run = sys.argv[1]
p, cut = parent(run)
main, sub = [], []
if p is not None:
    q = eval_rows(read_lines(f"{D}/one_{p}_i.log"), "ideal").get(cut, {})
    mem = memory_rows(read_lines(f"{D}/analysis/{p}_summary.log")).get(cut)
    main.append(("시작", q, mem))
    t = eval_rows(read_lines(f"{D}/one_{p}.log"), "track").get(cut)
    if t:
        sub.append(("시작", t))
I = eval_rows(read_lines(f"{D}/one_{run}_i.log"), "ideal")
MEM = memory_rows(read_lines(f"{D}/analysis/{run}_summary.log"))
if 0 in I and main:
    # the start evaluated under this run's own conditions (11-10: plan25n8 changed the plans' tyre limit -- the parent's row was
    # measured under the old one)
    main[0] = ("시작", I.pop(0), main[0][2])
for u in sorted(set(I) | set(MEM)):
    main.append((f"u{u}", I.get(u, {}), MEM.get(u)))
for u, t in sorted(eval_rows(read_lines(f"{D}/one_{run}.log"), "track").items()):
    sub.append((f"u{u}", t))
print(f"**지금 학습 {run}** ({STAGE_NOTE.get(run, '')}{f', {p} u{cut}에서 이어 받음' if p else ''}) · 학습 조건(plan대로 정확히 실행), 학습 차 512대\n")
print("| checkpoint | 빈 트랙 한 바퀴 | 상자+상대 한 바퀴 | 세게 닿음 /분 | 기억이 더 아는 것 (목표 +10 %p) |")
print("|---|---|---|---|---|")
for lab, q, mem in main:
    print(f"| {lab} | {cells(q)} | {f'{mem:+d} %p' if mem is not None else '–'} |")
if sub and "--sub" in sys.argv:          # the controller's tracking is analysed separately (11-10), not in the routine table
    print("\n참고 · 실제 차(물리 + 새 제어기), 100 update마다\n")
    print("| checkpoint | 빈 트랙 한 바퀴 | 상자+상대 한 바퀴 | 세게 닿음 /분 |")
    print("|---|---|---|---|")
    for lab, t in sub:
        print(f"| {lab} | {cells(t)} |")
