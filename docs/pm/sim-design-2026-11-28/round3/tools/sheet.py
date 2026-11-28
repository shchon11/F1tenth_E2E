#!/usr/bin/env python3
"""contact sheet of PNGs: sheet.py out.png cols cellw cellh file..."""
import os, sys
from PIL import Image, ImageDraw
out, cols, cw, ch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
files = sys.argv[5:]
rows = (len(files) + cols - 1) // cols
W = Image.new("RGB", (cw * cols, ch * rows), (24, 24, 26))
d = ImageDraw.Draw(W)
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB").resize((cw, ch), Image.LANCZOS)
    W.paste(im, ((i % cols) * cw, (i // cols) * ch))
    if "--label" in os.environ.get("SHEET", ""):
        d.text(((i % cols) * cw + 8, (i // cols) * ch + 6), os.path.basename(f), fill=(255, 255, 0))
W.save(out)
