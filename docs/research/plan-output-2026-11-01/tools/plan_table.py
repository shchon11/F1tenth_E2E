"""지금 학습의 고정 표 하나 (11-09, 사용자: "성능 표도 좀 띄엄띄엄 보내는것 같다? 표도 니 멋대로 만드는 것 같고", "뭐이리 씨발 이해가 안되냐
뭐이리 난잡해"). 보고마다 이 표 하나를 그대로 붙임 -- 열은 언제나 같고 행만 늘어남.
행 = checkpoint: 이어 받은 앞 단계들의 100 update 평가 행(run_chain: args.json의 init을 시작 run까지 거슬러) + 지금 run의 25 update마다 행.
지금 run의 시작점 분석(analysis/RUN_start.log, start_battery2)은 앞 단계의 그 checkpoint 행에 합침.
열: 빈 트랙 바퀴(plan대로 정확히 / 물리 차 + 새 제어기 / 물리 차 + MAP), 상자+상대 바퀴(새 제어기 / MAP), 세게 접촉/분(네 경우 평균; 새 제어기 /
MAP), plan - 실제(0.5초 / 1초 뒤 자리), 가속 · 브레이크 사용(회전이 막지 않을 때 모터 한계 대비 / 브레이크 한계 대비, 중앙), 기억(시야보다 더
맞힘 / 지우면 첫 단계 속도 출력이 바뀌는 크기). plan대로 = 25 update마다 분석(차 64대), 없으면 100 update 평가(차 12대).
인자: run 이름 [시작 run, 기본 plan25m]"""
import re, sys
import numpy as np
from run_chain import STAGE_NOTE, chain

D = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out"


def read_lines(*fns):
    lines = []
    for fn in fns:
        try:
            lines += open(fn, errors="replace").read().splitlines()
        except OSError:
            pass
    return lines


def analysis_rows(lines):
    rows, cur = {}, None
    for l in lines:
        m = re.match(r"== u(\d+)", l)
        if m:
            cur = int(m.group(1)); rows.setdefault(cur, {})      # a restarted loop repeats the header: merge
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
        m = re.search(r"0\.5 s: 속도 [+-][\d.]+ .*?자리 ([\d.]+) cm", l)
        if m:
            r["t05"] = float(m.group(1))
        m = re.search(r"1\.0 s: 속도 [+-][\d.]+ .*?자리 ([\d.]+) cm", l)
        if m:
            r["t10"] = float(m.group(1))
    return rows


def eval_rows(lines):
    rows, cond = {}, None
    for l in lines:
        m = re.search(r"u(\d+).*\[(ideal|track|map)\]", l)
        if m:
            cond = m.group(2); continue
        m = re.search(r"u(\d+)\b[^|]*?\s(빈 트랙|상대 차|상자\+상대|상자) \d+대 \| .*한 바퀴 중앙 ([\d.]+) s \| 접촉 /분 벽 [\d.]+\(세게 ([\d.]+)\) "
                      r"상자 [\d.]+\(세게 ([\d.]+)\) 차 [\d.]+\(세게 ([\d.]+)\)", l)
        if m and cond:
            hard = float(m.group(4)) + float(m.group(5)) + float(m.group(6))
            rows.setdefault(int(m.group(1)), {}).setdefault(cond, {})[m.group(2)] = (float(m.group(3)), hard)
    return {u: e for u, e in rows.items() if "track" in e and "map" in e}


run = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "plan25m"
stages = chain(run, root)
table = []                                   # (label, note, analysis row, eval row)
for si, (name, cut) in enumerate(stages):
    A = analysis_rows(read_lines(f"{D}/analysis/{name}_summary.log"))
    E = eval_rows(read_lines(f"{D}/one_{name}.log"))
    if cut is not None:                      # an earlier stage: its evaluated checkpoints up to the one passed on
        child = stages[si + 1][0]
        start = analysis_rows(read_lines(f"{D}/analysis/{child}_start.log")).get(0, {})
        us = sorted(u for u in E if u <= cut)
        if cut not in E and start:
            us.append(cut)
        for u in us:
            a = dict(A.get(u, {}))
            if u == cut:
                a.update(start)
            table.append((f"{name} u{u}", name, a, E.get(u, {})))
    else:
        for u in sorted(set(A) | set(E)):
            if A.get(u) or E.get(u):             # a checkpoint whose analysis has only just started has no numbers yet
                table.append((f"{name} u{u}", name, A.get(u, {}), E.get(u, {})))
f = lambda v, fmt: (fmt.format(v) if v is not None else "–")
print(f"**지금 학습 {run} 표** (행 = checkpoint, 25 update마다 학습 조건 분석 · 100 update마다 물리 차 평가, 위쪽 행 = 이어 받은 앞 단계)\n")
print("| checkpoint | 바꾼 것 | 빈 트랙 바퀴: plan대로 / 새 제어기 / MAP | 상자+상대 바퀴: 새 제어기 / MAP | 세게 접촉/분: 새 제어기 / MAP | "
      "plan−실제 자리: 0.5초 / 1초 | 가속 · 브레이크 사용 | 기억: 더 맞힘 / 출력 영향 |")
print("|---|---|---|---|---|---|---|---|")
seen = set()
for lab, name, a, e in table:
    note = STAGE_NOTE.get(name, "") if name not in seen else ""
    seen.add(name)
    lap_i = a.get("lap", e.get("ideal", {}).get("빈 트랙", (None,))[0])
    lap_t = e.get("track", {}).get("빈 트랙", (None,))[0]; lap_m = e.get("map", {}).get("빈 트랙", (None,))[0]
    both_t = e.get("track", {}).get("상자+상대", (None,))[0]; both_m = e.get("map", {}).get("상자+상대", (None,))[0]
    hard_t = float(np.mean([v[1] for v in e["track"].values()])) if "track" in e else None
    hard_m = float(np.mean([v[1] for v in e["map"].values()])) if "map" in e else None
    laps = f"{f(lap_i, '{:.2f}')} / {f(lap_t, '{:.2f}')} / {f(lap_m, '{:.2f}')}초"
    both = f"{f(both_t, '{:.2f}')} / {f(both_m, '{:.2f}')}초" if e else "–"
    hard = f"{f(hard_t, '{:.2f}')} / {f(hard_m, '{:.2f}')}" if e else "–"
    tail = f"{f(a.get('t05'), '{:.0f}')} / {f(a.get('t10'), '{:.0f}')} cm" if "t05" in a else "–"
    acc = f"{a['acc']} · {a['brk']} %" if "acc" in a and "brk" in a else "–"
    mem = f"{a['mem_gain']:+d} %p / {a['mem_drive']:.3f}" if "mem_gain" in a else "–"
    print(f"| {lab} | {note} | {laps} | {both} | {hard} | {tail} | {acc} | {mem} |")
