#!/usr/bin/env python3
"""디자인 토큰(tokens.css) -> Godot 테마 두 파일을 만든다. 손으로 고치지 말고 이 스크립트를 다시 돌린다.

  game/scripts/ui/tokens.gd      class_name Tokens: 색·간격·글자 단계 상수 (GDScript에서 직접 씀)
  game/theme/tenth_theme.tres    Theme: 기본 글꼴, 글자 단계별 형 변형(ReadoutXL ...), 판 StyleBox

규칙(02_direction.md, 05_components.md)
- 컴포넌트는 의미 토큰(--c-*, --t-*, --s-*)만 쓴다. 원색(--p-*)은 Tokens.PALETTE에만 남긴다.
- 글꼴 묶음(--font-*)은 글꼴 파일로 바꾼다. 라틴·숫자는 Saira, 한글은 Plex KR로 떨어지도록 fallbacks를 건다.
- CSS의 px = Godot의 px(1920×1200 기준 화면). 행간(line-height)은 Godot Label에 직접 대응이 없어 상수로만 남긴다.

사용: python3 game/tools/tokens_to_theme.py [--css docs/pm/sim-design-2026-11-28/tokens.css]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GAME = os.path.dirname(HERE)
REPO = os.path.dirname(GAME)
DEFAULT_CSS = os.path.join(REPO, "docs", "pm", "sim-design-2026-11-28", "tokens.css")

# 글꼴 묶음 -> (주 글꼴 파일 접두, {굵기: 파일 접미}, 한글 대체 여부)
FONT_FAMILIES = {
    "font-ui": ("IBMPlexSansKR", {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"}, False),
    "font-display": ("SairaCondensed", {500: "Medium", 600: "SemiBold", 700: "Bold"}, True),
    "font-label": ("SairaSemiCondensed", {500: "Medium", 600: "SemiBold", 700: "Bold"}, True),
    "font-data": ("IBMPlexMono", {400: "Regular", 500: "Medium", 600: "SemiBold"}, True),
}
KR_FALLBACK = ("IBMPlexSansKR", {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"})
TYPE_NAMES = {  # --t-* -> Godot 테마 형 변형 이름(Label 기반)
    "t-readout-xl": "ReadoutXL", "t-readout-l": "ReadoutL", "t-readout-m": "ReadoutM", "t-readout-s": "ReadoutS",
    "t-title-l": "TitleL", "t-title-m": "TitleM", "t-title-s": "TitleS", "t-body": "Body", "t-body-s": "BodyS",
    "t-label": "Caps", "t-micro": "Micro", "t-data": "Data", "t-data-s": "DataS",
}
TYPE_COLOR = {  # 형 변형의 기본 글자색(의미 토큰)
    "Caps": "c-text-4", "Micro": "c-text-4", "BodyS": "c-text-3", "DataS": "c-text-3", "Data": "c-text-2",
}


def parse_root(css: str) -> dict:
    """:root { --a: b; } 블록의 사용자 속성만(시트 변형·미디어 질의는 무시)."""
    m = re.search(r":root\s*\{(.*?)\n\}", css, re.S)
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    out = {}
    for name, val in re.findall(r"--([a-z0-9-]+)\s*:\s*([^;]+);", body):
        out[name] = val.strip()
    return out


def resolve(tok: dict, v: str, depth: int = 0) -> str:
    if depth > 10:
        raise ValueError(f"var() 순환: {v}")
    return re.sub(r"var\(--([a-z0-9-]+)\)", lambda m: resolve(tok, tok[m.group(1)], depth + 1), v)


def color(v: str):
    v = v.strip()
    if v.startswith("#"):
        h = v[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        a = int(h[6:8], 16) / 255.0 if len(h) == 8 else 1.0
        return (r, g, b, a)
    m = re.match(r"rgba?\(([^)]*)\)", v)
    if m:
        p = [x.strip() for x in m.group(1).split(",")]
        r, g, b = (float(x) / 255.0 for x in p[:3])
        a = float(p[3]) if len(p) > 3 else 1.0
        return (r, g, b, a)
    return None


def gd_color(c) -> str:
    return "Color(%.6g, %.6g, %.6g, %.6g)" % c


def font_file(prefix: str, weights: dict, w: int) -> str:
    best = min(weights, key=lambda k: abs(k - w))
    return f"res://assets/fonts/{prefix}-{weights[best]}.woff2"


def parse_font(v: str):
    """'600 104px/0.9 "Saira Condensed", ...' (var 풀린 뒤) -> (굵기, 크기 px, 행간 px|배수 문자열)."""
    m = re.match(r"(\d+)\s+(\d+)px/([\d.]+)(px)?", v)
    w, size, lh, unit = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    line_px = float(lh) if unit else float(lh) * size
    return w, size, line_px


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--css", default=DEFAULT_CSS)
    a = ap.parse_args(argv)
    raw = parse_root(open(a.css, encoding="utf-8").read())
    tok = {k: resolve(raw, v) for k, v in raw.items()}

    colors = {k: color(v) for k, v in tok.items() if k.startswith(("c-",)) and color(v)}
    palette = {k: color(v) for k, v in tok.items() if k.startswith("p-") and color(v)}
    sizes = {k: int(v[:-2]) for k, v in tok.items() if re.fullmatch(r"\d+px", v)}
    durs = {k: int(v[:-2]) for k, v in tok.items() if re.fullmatch(r"\d+ms", v)}
    # 글자 단계: 원래 값(var 풀기 전)에서 어떤 묶음을 썼는지 본다
    types = {}
    for k, v in raw.items():
        if k.startswith("t-"):
            fam = re.search(r"var\(--(font-[a-z]+)\)", v).group(1)
            w, size, line_px = parse_font(resolve(raw, re.sub(r"var\(--font-[a-z]+\)", "", v)))
            types[k] = (fam, w, size, line_px)

    # ---------------------------------------------------------------- tokens.gd
    L = ["# 자동 생성: game/tools/tokens_to_theme.py <- docs/pm/sim-design-2026-11-28/tokens.css",
         "# 손으로 고치지 말 것. 컴포넌트는 C_* · S_* · T_* 만 쓴다(원색 PALETTE 직접 사용 금지).",
         "class_name Tokens", "extends RefCounted", ""]
    for k, c in sorted(colors.items()):
        L.append(f"const {k.upper().replace('-', '_')} := {gd_color(c)}")
    L.append("")
    for k, s in sorted(sizes.items()):
        L.append(f"const {k.upper().replace('-', '_')} := {s}")
    for k, d in sorted(durs.items()):
        L.append(f"const {k.upper().replace('-', '_')}_MS := {d}")
    L.append("")
    L.append("## 글자 단계: [테마 형 변형 이름, 글꼴 묶음, 굵기, 크기 px, 행간 px]")
    L.append("const TYPE := {")
    for k, (fam, w, size, line_px) in types.items():
        L.append(f'\t"{k}": ["{TYPE_NAMES[k]}", "{fam}", {w}, {size}, {line_px:g}],')
    L.append("}")
    L.append("")
    L.append("const PALETTE := {")
    for k, c in sorted(palette.items()):
        L.append(f'\t"{k}": {gd_color(c)},')
    L.append("}")
    L.append("")
    gd_path = os.path.join(GAME, "scripts", "ui", "tokens.gd")
    os.makedirs(os.path.dirname(gd_path), exist_ok=True)
    open(gd_path, "w", encoding="utf-8").write("\n".join(L))

    # ---------------------------------------------------------------- tenth_theme.tres
    ext = {}       # path -> id

    def ext_id(path):
        if path not in ext:
            ext[path] = f"{len(ext) + 1}_font"
        return ext[path]

    subs = []      # (id, type, body lines)
    fonts_for = {}  # (fam, w) -> sub id

    def font_res(fam, w):
        key = (fam, w)
        if key in fonts_for:
            return fonts_for[key]
        prefix, weights, kr = FONT_FAMILIES[fam]
        main_path = font_file(prefix, weights, w)
        sid = f"fv_{fam.split('-')[1]}_{w}"
        body = [f'base_font = ExtResource("{ext_id(main_path)}")']
        if kr:
            body.append(f'fallbacks = Array[Font]([ExtResource("{ext_id(font_file(*KR_FALLBACK, w))}")])')
        subs.append((sid, "FontVariation", body))
        fonts_for[key] = sid
        return sid

    def stylebox(sid, bg, border=None, widths=(0, 0, 0, 0), margins=(0, 0, 0, 0)):
        body = [f"bg_color = {gd_color(bg)}"]
        if border:
            body.append(f"border_color = {gd_color(border)}")
            for side, wv in zip(("left", "top", "right", "bottom"), widths):
                if wv:
                    body.append(f"border_width_{side} = {wv}")
        for side, mv in zip(("left", "top", "right", "bottom"), margins):
            body.append(f"content_margin_{side} = {mv}")
        subs.append((sid, "StyleBoxFlat", body))
        return sid

    C = colors
    body_font = font_res("font-ui", 400)
    sb_plate = stylebox("sb_plate", C["c-bg-overlay"], C["c-line"], (1, 1, 1, 1), (16, 12, 16, 12))
    sb_panel = stylebox("sb_panel", C["c-bg-panel"], C["c-line"], (1, 1, 1, 1), (16, 16, 16, 16))
    sb_strip = stylebox("sb_strip", C["c-bg-void"], C["c-line-strong"], (0, 0, 0, 1), (16, 0, 16, 0))
    sb_hint = stylebox("sb_hint", C["c-bg-void"], C["c-line"], (0, 1, 0, 0), (16, 0, 16, 0))
    sb_key = stylebox("sb_key", C["c-bg-well"], C["c-line-strong"], (1, 1, 1, 2), (6, 1, 6, 1))
    sb_chip = stylebox("sb_chip_accent", C["c-accent"], None, margins=(8, 2, 8, 2))

    R = [f'default_font = SubResource("{body_font}")', f"default_font_size = {types['t-body'][2]}",
         f"Label/colors/font_color = {gd_color(C['c-text'])}",
         f"Panel/styles/panel = SubResource(\"{sb_panel}\")",
         f"PanelContainer/styles/panel = SubResource(\"{sb_panel}\")"]
    for k, (fam, w, size, line_px) in types.items():
        n = TYPE_NAMES[k]
        R.append(f'{n}/base_type = &"Label"')
        R.append(f'{n}/fonts/font = SubResource("{font_res(fam, w)}")')
        R.append(f"{n}/font_sizes/font_size = {size}")
        R.append(f"{n}/colors/font_color = {gd_color(C[TYPE_COLOR.get(n, 'c-text')])}")
    for n, sb in (("Plate", sb_plate), ("Strip", sb_strip), ("HintBar", sb_hint), ("KeyCap", sb_key),
                  ("ChipAccent", sb_chip)):
        R.append(f'{n}/base_type = &"PanelContainer"')
        R.append(f'{n}/styles/panel = SubResource("{sb}")')

    T = [f'[gd_resource type="Theme" format=3]', ""]
    for path, i in ext.items():
        T.append(f'[ext_resource type="FontFile" path="{path}" id="{i}"]')
    T.append("")
    for sid, typ, body in subs:
        T.append(f'[sub_resource type="{typ}" id="{sid}"]')
        T += body
        T.append("")
    T.append("[resource]")
    T += R
    T.append("")
    tres = os.path.join(GAME, "theme", "tenth_theme.tres")
    os.makedirs(os.path.dirname(tres), exist_ok=True)
    open(tres, "w", encoding="utf-8").write("\n".join(T))
    print(f"tokens.gd: 색 {len(colors)}, 크기 {len(sizes)}, 글자 단계 {len(types)} | theme: 글꼴 {len(ext)}, 하위 리소스 {len(subs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
