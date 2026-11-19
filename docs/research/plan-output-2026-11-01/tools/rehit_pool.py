"""(11-19) 같은 상자 다시 박기를 여러 시험 기록에 걸쳐 모아 봄 -- 한 시험에는 '지난번에 세게 닿은 통과'가 6-32번뿐이라 하나로는 못 가름.
행 파일 = exam_boxes.py 를 EXAM_ROWS=파일 로 돌려 남긴 통과별 행. 길 위의 상자(길에서 0.40 m 안, exam_boxes의 (라)와 같음)만.
인자: 이름=행npy,행npy,...  [이름=...]
이름마다: 첫 통과 세게 닿음 | 둘째 통과부터, 지난번에 세게 닿았던 상자를 또 세게 닿음 | 지난번 안 닿거나 약하게였던 상자를 세게 닿음 (Wilson 95 %)"""
import sys
import numpy as np

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV = range(10)


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def cell(m, hd):
    k, n = int((m & hd).sum()), int(m.sum())
    lo, hi = wilson(k, n)
    return f"{100 * k / max(n, 1):5.1f} % [{lo:4.1f}-{hi:4.1f}] ({k}/{n})"


for arg in sys.argv[1:]:
    name, fs = arg.split("=", 1)
    rows = np.concatenate([np.load(f) for f in fs.split(",")])
    way = rows[:, IGAP] < 0.40
    hd = rows[:, IHARD] > 0
    first = way & (rows[:, IN] == 1)
    again = way & (rows[:, IN] >= 2) & (rows[:, IPREV] > 0)
    clean = way & (rows[:, IN] >= 2) & (rows[:, IPREV] == 0)
    print(f"{name:24s} 기록 {len(fs.split(',')):2d}개 | 첫 통과 {cell(first, hd)} | 지난번 세게 닿은 상자 또 {cell(again, hd)} | "
          f"지난번 깨끗 -> 세게 {cell(clean, hd)}")
