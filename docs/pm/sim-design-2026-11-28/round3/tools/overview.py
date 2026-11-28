#!/usr/bin/env python3
"""overview.png: every screen, small, in flow order, file name under each."""
import glob, os
from PIL import Image, ImageDraw, ImageFont
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(R, "out", "[0-9]*.png")))
cw, ch, cap, gap, cols = 640, 400, 44, 16, 6
rows = (len(files) + cols - 1) // cols
W = Image.new("RGB", (cols * cw + (cols + 1) * gap, rows * (ch + cap) + (rows + 1) * gap), (14, 15, 17))
d = ImageDraw.Draw(W)
font = ImageFont.truetype(os.path.join(R, "..", "round2", "fonts", "PretendardVariable.ttf"), 22)
try:
    font.set_variation_by_name("SemiBold")
except Exception:
    pass
for i, f in enumerate(files):
    x = gap + (i % cols) * (cw + gap)
    y = gap + (i // cols) * (ch + cap + gap)
    W.paste(Image.open(f).convert("RGB").resize((cw, ch), Image.LANCZOS), (x, y))
    d.text((x + 2, y + ch + 10), os.path.basename(f)[:-4], fill=(244, 244, 241), font=font)
W.save(os.path.join(R, "overview.png"))
print(W.size)
