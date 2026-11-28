#!/usr/bin/env python3
"""side.py out.jpg ours.png ref.jpg [ref2.jpg]: our screen next to real game screens, same height"""
import sys
from PIL import Image
out, files = sys.argv[1], sys.argv[2:]
H = 620
ims = []
for f in files:
    im = Image.open(f).convert("RGB")
    ims.append(im.resize((round(im.width * H / im.height), H), Image.LANCZOS))
W = Image.new("RGB", (sum(i.width for i in ims) + 12 * (len(ims) - 1), H), (255, 255, 255))
x = 0
for im in ims:
    W.paste(im, (x, 0)); x += im.width + 12
W.save(out, quality=88)
