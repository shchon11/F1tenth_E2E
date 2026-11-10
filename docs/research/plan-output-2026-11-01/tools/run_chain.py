"""이어 온 학습 단계 (11-09, 사용자: "아니 씨발 지금 그럼 세개를 학습하고 있는거야?" -- run 이름이 단계마다 바뀌어도 한 학습으로 보이게).
run의 args.json(args.init = .../앞run/plan_uN.pt)을 거슬러 시작 run까지: [(run, 이어 준 update)] 오래된 것부터, 마지막 = 지금 run(None = 전부).
STAGE_NOTE = 단계마다 바꾼 것(새 단계를 시작할 때 여기에 한 줄 추가)."""
import os, re, json

R = os.path.expanduser("~/f1sim_runs")
STAGE_NOTE = {"plan25m": "기억 + 상자 치우기 + 가속 탐색", "plan25n": "+ plan 뒷부분 예측", "plan25n2": "+ plan 충돌 감점 끔",
              "plan25n3": "+ 기억 학습 구간 2바퀴(12.8초)",
              "plan25n4": "+ 기억 보조 문제(직선 12 m, 시작 직후 멈춤)",
              "plan25n5": "+ 기억 보조 문제(트랙을 따라 앞 12 m 상자 위치, 안 보이는 것 포함)",
              "plan25n6": "+ plan 지그재그 비용(무게 0.005)",
              "plan25n7": "+ plan 지그재그 비용 무게 0.02"}


def run_args(name):
    try:
        return json.load(open(os.path.join(R, name, "args.json")))["args"]
    except (OSError, KeyError, ValueError):
        return {}


def parent(name):
    """the run and update this run was initialised from"""
    ini = run_args(name).get("init") or ""
    m = re.search(r"plan_u(\d+)\.pt$", ini)
    return (os.path.basename(os.path.dirname(ini)), int(m.group(1))) if m else (None, None)


def chain(run, root):
    out, cur = [(run, None)], run
    while cur != root:
        p, u = parent(cur)
        if p is None or not os.path.exists(os.path.join(R, p, "progress.jsonl")):
            break
        out.append((p, u)); cur = p
    return out[::-1]
