#!/usr/bin/env python3
"""TENTH 아이콘: 흰 바탕에 IFAC 2026(korea26_scan) 코스 중심선 윤곽 한 줄(검정). 글자·로고 없음.

  .venv/bin/python packaging/tenth/make_icon.py [중심선 csv]   # -> packaging/tenth/icon.svg, icon.png(256)
중심선은 서버가 만든 캐시(~/.cache/f1sim/centerlines/korea26_scan_*.csv, x,y 한 줄씩)를 읽는다. 한 번 만들어 저장소에 둔다.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.expanduser("~/.cache/f1sim/centerlines/korea26_scan_*.csv")))[0]
xy = np.loadtxt(src, delimiter=",")
# 매끄럽게(원형 이동 평균) + 점 줄이기
k = 9
pad = np.vstack([xy[-k:], xy, xy[:k]])
ker = np.ones(2 * k + 1) / (2 * k + 1)
xy = np.stack([np.convolve(pad[:, i], ker, mode="same")[k:-k] for i in range(2)], 1)[::4]
xy[:, 1] *= -1                                     # 화면 y는 아래로
# 긴 축을 대각선(45°)으로 돌린다: 길쭉한 코스도 정사각형을 채우도록
c = xy - xy.mean(0)
ev = np.linalg.eigh(np.cov(c.T))[1][:, -1]
a = np.arctan2(ev[1], ev[0])
r = np.deg2rad(-45.0) - a
xy = c @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
lo, hi = xy.min(0), xy.max(0)
S, M = 256.0, 40.0                                 # 캔버스, 여백
scale = (S - 2 * M) / float((hi - lo).max())
p = (xy - lo) * scale + M + ((S - 2 * M) - (hi - lo) * scale) / 2
d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in p) + " Z"
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">\n'
       f'<rect x="8" y="8" width="240" height="240" rx="40" fill="#ffffff" stroke="#d0d0d0" stroke-width="2"/>\n'
       f'<path d="{d}" fill="none" stroke="#111111" stroke-width="14" stroke-linejoin="round" stroke-linecap="round"/>\n'
       f'</svg>\n')
open(os.path.join(HERE, "icon.svg"), "w").write(svg)

from PIL import Image, ImageDraw                    # noqa: E402  (venv에 있음)
F = 4
im = Image.new("RGBA", (256 * F, 256 * F), (0, 0, 0, 0))
dr = ImageDraw.Draw(im)
dr.rounded_rectangle([8 * F, 8 * F, 248 * F, 248 * F], radius=40 * F, fill=(255, 255, 255, 255), outline=(208, 208, 208, 255), width=2 * F)
pts = [(float(a) * F, float(b) * F) for a, b in p] + [(float(p[0, 0]) * F, float(p[0, 1]) * F)]
dr.line(pts, fill=(17, 17, 17, 255), width=14 * F, joint="curve")
for a, b in pts:                                    # 둥근 이음
    r = 7 * F
    dr.ellipse([a - r, b - r, a + r, b + r], fill=(17, 17, 17, 255))
im.resize((256, 256), Image.LANCZOS).save(os.path.join(HERE, "icon.png"))
print("icon.svg, icon.png <-", src)
