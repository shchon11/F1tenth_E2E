#!/usr/bin/env python3
"""Writes the round-3 mockup pages (static HTML, no script) from recorded data.

    python3 -I tools/build.py

Every number on the pages comes from a file: the recorded session (result json + steps), the
training status and progress logs, the measured maps, the bag_profile output. See SPEC.md §9.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import world  # noqa: E402

A = os.path.join(ROOT, "assets")
SID = "20261129-000708-quali-a-1124"
CAR = "Traxxas 4-Tec / Hokuyo UST-10LX / Jetson AGX Orin"
COURSE = "IFAC 2026"
M = "−"          # minus sign


def ov(name):
    return json.load(open(os.path.join(A, "ov_%s.json" % name)))


def pts(p):
    return " ".join("%.1f,%.1f" % (x, y) for x, y in p)


def fnum(v, nd):
    return ("%." + str(nd) + "f") % v


def page(name, title, body):
    html = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>%s</title>\n'
            '<link rel="stylesheet" href="assets/base.css"></head><body>\n%s\n</body></html>\n') % (title, body.strip())
    with open(os.path.join(ROOT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", name + ".html")


def now(*pairs):
    return '<dl class="now">' + "".join("<div><dt>%s</dt><dd>%s</dd></div>" % p for p in pairs) + "</dl>"


def row(label, value, cls="", more=None, mid=None):
    m = ""
    if more:
        m = '<dl class="more">' + "".join("<div><dt>%s</dt><dd>%s</dd></div>" % p for p in more) + "</dl>"
    e = "<em>%s</em>" % mid if mid else ""
    return '<div class="row %s"><label>%s</label>%s<b>%s</b>%s</div>' % (cls, label, e, value, m)


def sub(*words):
    return '<div class="sub">' + "".join("<span>%s</span>" % w for w in words) + "</div>"


# ----------------------------------------------------------------- data
J, Z = world.session(SID)
LAPS = J["laps"]
DRIVER = "%s u%d" % (J["detail"]["driver"]["run"], J["detail"]["driver"]["update"])
DUR, RM = J["duration_s"], J["remove_at_s"]


def hud_state(t):
    i = int(round(t / 0.025))
    done = [l for l in LAPS if l["t_end"] <= t]
    cur = next((l for l in LAPS if l["t_start"] <= t < l["t_end"]), None)
    streak = 0
    for l in reversed(done):
        if not l["clean"]:
            break
        streak += 1
    if cur and any(c["t"] <= t and c["lap"] == cur["n"] for c in J["contacts"]):
        streak = 0
    t0 = cur["t_start"] if cur else (done[-1]["t_end"] if done else 0.0)
    return dict(t=t, v=float(Z["v"][i]), steer=float(Z["steer"][i]),
                acc=float((Z["v"][i + 4] - Z["v"][i - 4]) / 0.2), x=float(Z["x"][i]), y=float(Z["y"][i]),
                streak=streak, best=min((l["time"] for l in done), default=None), lap_t=t - t0, done=done,
                left=DUR - t, boxes_on=bool(Z["boxes_on"][i]))


def mmss(s):
    s = int(math.ceil(s - 1e-6))
    return "%d:%02d" % (s // 60, s % 60)


def records():
    snap = os.path.join(A, "records.json")
    if os.path.exists(snap):
        return json.load(open(snap))
    names = {"kr26-hall": "IFAC 2026", "lab-12x16": "map12x16"}
    out = []
    for j in world.all_results():
        d = j["detail"].get("driver") or {}
        drv = "%s u%d" % (d["run"], d["update"]) if d.get("run") else "Follow-the-Gap"
        sid = j["id"]
        out.append(dict(id=sid, when="%s-%s %s:%s" % (sid[4:6], sid[6:8], sid[9:11], sid[11:13]),
                        course=names[j["spec"]["course"]], driver=drv, streak=j["score"]["best_clean_streak"],
                        best=j["score"]["best_lap"], dur=j["duration_s"]))
    out.sort(key=lambda r: r["id"], reverse=True)
    json.dump(out, open(snap, "w"), ensure_ascii=False, indent=1)
    return out


def training():
    snap = os.path.join(A, "training.json")
    if os.path.exists(snap):
        return json.load(open(snap))
    st = json.load(open(os.path.expanduser("~/f1sim_runs/_train_status.json")))
    runs = []
    for k, r in st["runs"].items():
        p = r["progress"]
        runs.append(dict(name=k, update=p["update"], updates=r["updates"], laps_per_min=p["laps_per_min"],
                         exams=[dict(u=e["u"], lap=e["lap"], boxes=e["box_boxes"], walls=e["box_walls"]) for e in r["exams"]]))
    runs.sort(key=lambda r: (r["update"] >= r["updates"], -r["update"] / r["updates"]))
    L = [json.loads(l) for l in open(os.path.expanduser("~/f1sim_runs/hs43b/progress.jsonl"))]
    series = dict(update=[l["update"] for l in L], laps_per_min=[l["laps_per_min"] for l in L],
                  boxes_per_min=[l["contacts_prop_per_min"] for l in L])
    out = dict(runs=runs, hs43b=series)
    json.dump(out, open(snap, "w"), ensure_ascii=False)
    return out


# ----------------------------------------------------------------- pieces
def minimap(st, boxes_on):
    c = world.Course(COURSE)
    lo, hi = c.lo - 0.4, c.hi + 0.4
    k = 400 / (hi[1] - lo[1])

    def T(p):                                  # north to the right
        p = np.asarray(p, float)
        return np.stack([(p[..., 1] - lo[1]) * k, (p[..., 0] - lo[0]) * k], -1)
    h = (hi[0] - lo[0]) * k
    s = '<svg class="hud-map" viewBox="0 0 400 %.0f" preserveAspectRatio="xMaxYMin meet">' % h
    for w in c.walls:
        s += '<polygon points="%s" fill="none" stroke="#f4f4f1" stroke-opacity=".8" stroke-width="3" stroke-linejoin="round"/>' % pts(T(w[::4]))
    if boxes_on:
        for b in J["boxes"]:
            x, y = T([b["x"], b["y"]])
            s += '<rect x="%.1f" y="%.1f" width="9" height="9" fill="#f4f4f1" fill-opacity=".8"/>' % (x - 4.5, y - 4.5)
    x, y = T([st["x"], st["y"]])
    s += '<circle cx="%.1f" cy="%.1f" r="8" fill="#f4f4f1" stroke="#0e0f11" stroke-width="3"/></svg>' % (x, y)
    return s


def bar(label, value, lo, hi, left, unit):
    """a zero-anchored bar: value between lo and hi, fill from zero."""
    z = (0 - lo) / (hi - lo) * 100
    v = (min(max(value, lo), hi) - lo) / (hi - lo) * 100
    a, b = min(z, v), max(z, v)
    return ('<dl class="hud-bar" style="left:%dpx"><dt>%s</dt><div class="line"><i style="left:%.1f%%;width:%.1f%%"></i>'
            '<u style="left:%.1f%%"></u></div></dl>') % (left, label, a, b - a, z)


def hud(t, layers=False, msg=None, cls="", only_map=False):
    st = hud_state(t)
    best = fnum(st["best"], 3) if st["best"] else "–"
    s = '<div class="hud %s">' % cls
    if not only_map:
        s += ('<dl class="hud-tl"><div><dt>무충돌 연속</dt><dd>%d</dd></div><div><dt>최고 랩</dt><dd>%s</dd></div>'
              '<div><dt>이번 랩</dt><dd>%s</dd></div></dl>') % (st["streak"], best, fnum(st["lap_t"], 2))
    prog = t / DUR * 100
    s += ('<dl class="hud-tc"><dt>남은 시간</dt><dd>%s</dd><div class="line"><i style="width:%.1f%%"></i>'
          '<u style="left:%.1f%%"></u><span style="left:%.1f%%">상자 치움</span></div></dl>') % (mmss(st["left"]), prog, RM / DUR * 100, RM / DUR * 100)
    s += '<dl class="hud-bc"><dd>%s<small>m/s</small></dd></dl>' % fnum(st["v"], 1)
    if msg:
        s += '<div class="hud-msg">%s</div>' % msg
    if only_map:
        s += minimap(st, st["boxes_on"])
    if layers:
        s += minimap(st, st["boxes_on"])
        s += '<div class="hud-laps">'
        for l in st["done"]:
            s += "<div><span>%d</span><span>%s</span><span>%s</span></div>" % (
                l["n"], fnum(l["time"], 3), ("접촉 %d" % l["contacts"]) if l["contacts"] else "")
        s += "</div>"
        # steering: the servo window is not centred (left +0.412, right -0.254 rad); left on the left
        s += bar("조향", -st["steer"], -0.412, 0.254, 470, "rad")
        s += bar("가감속", st["acc"], -4.8, 5.6, 1150, "m/s²")
    return s + "</div>"


# ----------------------------------------------------------------- pages
def build():
    rec = records()
    tr = training()
    courses = {k: world.Course(k) for k in world.MAPS}

    # 01 main
    page("01_main", "메인", '''
<img class="bg" src="assets/hero.png" alt=""><div class="film-l"></div><div class="film-t"></div>
<div class="mark">TENTH</div>
<div class="col" style="top:286px"><div class="menu fit"><div class="on">달리기</div><div>차고</div><div>코스</div><div>훈련</div><div>기록</div></div>
%s</div>
%s''' % (sub("설정", "종료"), now(("차", CAR), ("코스", COURSE), ("드라이버", DRIVER))))

    # 02 race setup (the competition's qualifying format: 120 s, three boxes taken away at 60 s)
    page("02_race", "레이스 준비", '''
<img class="bg" src="assets/venue.png" alt=""><div class="film-l"></div>
<div class="title">달리기</div>
<div class="col"><div class="rows">%s</div><div class="go">출발</div>%s</div>''' % (
        row("코스", COURSE, "on lr") + row("드라이버", DRIVER) + row("방식", "예선") + row("시간", "120초")
        + row("상자", "3개") + row("상자 치움", "60초"), sub("뒤로")))

    # 03-05 driving
    page("03_hud", "주행 HUD 기본", '<img class="bg" src="assets/drive.png" alt=""><div class="film-t"></div><div class="film-b"></div>' + hud(26.8))
    page("04_hud_all", "주행 HUD 전부 켬", '<img class="bg" src="assets/drive_layers.png" alt=""><div class="film-t"></div><div class="film-b"></div>' + hud(26.8, layers=True))
    page("05_hud_clear", "주행 HUD 상자 치움", '<img class="bg" src="assets/drive_clear.png" alt=""><div class="film-t"></div><div class="film-b"></div>' + hud(30.0, msg="상자 치움"))

    # 06 pause, 07 layers
    paused = '<img class="bg" src="assets/drive_pause.png" alt=""><div class="film-t"></div><div class="film-b"></div>' + hud(44.0, cls="faded") + '<div class="film-all"></div>'
    page("06_pause", "일시정지", paused + '''
<div class="col" style="top:286px"><div class="menu fit"><div class="on">계속</div><div>다시 달리기</div><div>표시</div><div>설정</div><div>나가기</div></div></div>''')
    live = ('<img class="bg" src="assets/drive_pause.png" alt=""><div class="film-t"></div><div class="film-b"></div>'
            + hud(44.0, cls="menu-open", only_map=True) + '<div class="film-l"></div>')
    page("07_layers", "표시", live + '''
<div class="title">표시</div>
<div class="col"><div class="rows">%s</div>%s</div>''' % (
        row("미니맵", "켬", "on lr") + row("LiDAR", "끔") + row("드라이버 궤적", "끔") + row("조향·가감속", "끔") + row("바퀴 기록", "끔"), sub("뒤로")))

    # 08 result
    sc = J["score"]
    page("08_result", "결과", '''
<img class="bg" src="assets/finish.png" alt=""><div class="film-l"></div><div class="film-t"></div>
<div class="title">결과</div>
<dl class="score" style="top:292px"><div><dt>무충돌 연속</dt><dd>%d<small>바퀴</small></dd></div><div><dt>최고 랩</dt><dd>%s<small>초</small></dd></div></dl>
<div class="col" style="top:512px"><div class="go focus">다시 달리기</div>%s</div>
%s''' % (sc["best_clean_streak"], fnum(sc["best_lap"], 3), sub("바퀴 기록", "다시 보기", "나가기"),
         now(("코스", COURSE), ("드라이버", DRIVER))))

    # 09 result laps: focus on lap 2, its line and its contacts in the scene
    o = ov("top")
    FOC = 2
    rows = ""
    for l in LAPS:
        cs = [c for c in J["contacts"] if c["lap"] == l["n"]]
        mid = ("접촉 %d" % len(cs)) if cs else None
        if l["n"] == FOC:
            more = [("상자", "%d" % sum(c["kind"] == "box" for c in cs)), ("벽", "%d" % sum(c["kind"] == "wall" for c in cs)),
                    ("세게", "%d" % sum(c["hard"] for c in cs))]
            rows += row("%d" % l["n"], fnum(l["time"], 3), "s on", more=more, mid=mid)
        else:
            rows += row("%d" % l["n"], fnum(l["time"], 3), "s", mid=mid)
    svg = '<svg class="ov" viewBox="0 0 1920 1200">'
    svg += '<polyline points="%s" fill="none" stroke="#f4f4f1" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>' % pts(o["laps"][str(FOC)])
    for c, (x, y) in zip(J["contacts"], o["contacts"]):
        if c["lap"] != FOC:
            continue
        if c["hard"]:
            svg += '<circle cx="%.1f" cy="%.1f" r="12" fill="#f4f4f1" stroke="#0e0f11" stroke-width="3"/>' % (x, y)
        else:
            svg += '<circle cx="%.1f" cy="%.1f" r="11" fill="#0e0f11" stroke="#f4f4f1" stroke-width="3.5"/>' % (x, y)
    svg += "</svg>"
    page("09_result_laps", "바퀴 기록", '''
<img class="bg" src="assets/top.png" alt="">%s<div class="film-l"></div>
<div class="title">바퀴 기록</div>
<div class="col"><div class="rows">%s</div>%s</div>''' % (svg, rows, sub("뒤로")))

    # 10 garage, 11 steering, 12 sensor
    page("10_garage", "차고", '''
<img class="bg" src="assets/garage.png" alt=""><div class="film-l"></div>
<div class="title">차고</div>
<div class="col" style="top:270px"><div class="menu m fit"><div class="on">섀시</div><div>타이어</div><div>구동</div><div>조향</div><div>센서</div><div>컴퓨터</div><div>외관</div></div>%s</div>
%s''' % (sub("뒤로"), now(("섀시", "Traxxas 4-Tec"), ("LiDAR", "Hokuyo UST-10LX"), ("컴퓨터", "Jetson AGX Orin"))))
    page("11_garage_steer", "차고 조향", '''
<img class="bg" src="assets/garage_steer.png" alt=""><div class="film-l"></div>
<div class="title">조향</div>
<div class="col"><div class="rows">%s</div>%s</div>''' % (
        row("서보 창", "%s0.254 / +0.412 rad" % M) + row("지연", "0.10 s", "on lr", more=[("범위", "0.09 – 0.11 s"), ("출처", "추정")])
        + row("트림", "%s0.021" % M), sub("뒤로")))
    page("12_garage_sensor", "차고 센서", '''
<img class="bg" src="assets/garage_sensor.png" alt=""><div class="film-l"></div>
<div class="title">센서</div>
<div class="col"><div class="rows">%s</div>%s</div>''' % (
        row("센서", "Hokuyo UST-10LX") + row("시야", "270°", "on lr", more=[("출처", "규격")]) + row("빔", "1081")
        + row("주기", "40 Hz") + row("거리", "10 m"), sub("뒤로")))

    # 13 course select (focus ICCAS 2025)
    c25 = courses["ICCAS 2025"]
    page("13_course", "코스", '''
<img class="bg" src="assets/course_iccas.png" alt=""><div class="film-l"></div>
<div class="title">코스</div>
<div class="col" style="top:286px"><div class="menu fit"><div>IFAC 2026</div><div class="on">ICCAS 2025</div><div>ICRA 2022</div><div>map12x16</div><div>map16x07</div></div>%s</div>
%s''' % (sub("편집", "새로 만들기", "bag에서 가져오기", "뒤로"),
         now(("길이", "%.1f m" % c25.length), ("최소 폭", "%.2f m" % c25.min_width), ("크기", "%.1f × %.1f m" % tuple(c25.size)))))

    # 14 course edit (map16x07)
    c16 = courses["map16x07"]
    o = ov("edit")
    svg = '<svg class="ov" viewBox="0 0 1920 1200">'
    svg += '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#f4f4f1" stroke-width="5"/>' % (o["gate"][0][0], o["gate"][0][1], o["gate"][1][0], o["gate"][1][1])
    sel = o["outer"][9]
    for x, y in o["outer"] + o["inner"]:
        if [x, y] == sel:
            continue
        svg += '<circle cx="%.1f" cy="%.1f" r="7" fill="#0e0f11" stroke="#f4f4f1" stroke-width="3"/>' % (x, y)
    svg += '<circle cx="%.1f" cy="%.1f" r="22" fill="none" stroke="#f4f4f1" stroke-width="3"/><circle cx="%.1f" cy="%.1f" r="10" fill="#f4f4f1"/>' % (sel[0], sel[1], sel[0], sel[1])
    svg += "</svg>"
    page("14_course_edit", "코스 편집", '''
<img class="bg" src="assets/edit.png" alt="">%s<div class="film-l"></div>
<div class="title">map16x07</div>
<div class="col" style="top:270px"><div class="menu m fit"><div class="on">호스</div><div>상자</div><div>출발선</div></div>
<dl class="pairs" style="margin-top:34px"><div><dt>길이</dt><dd>%.1f m</dd></div><div><dt>최소 폭</dt><dd>%.2f m</dd></div></dl>
<div class="go" style="margin-top:46px">시험 주행</div>%s</div>''' % (svg, c16.length, c16.min_width, sub("저장", "뒤로")))

    # 15 bag import: values bag_profile wrote for this bag (sysid examples/20260827-115713)
    page("15_course_bag", "bag에서 가져오기", '''
<img class="bg" src="assets/bag.png" alt=""><div class="film-l"></div><div class="film-t"></div>
<div class="title">가져오기</div>
<div class="col"><div class="rows">%s</div><div class="go focus">저장</div>%s</div>
%s''' % (row("지도", "8.35 × 22.55 m") + row("서보 창", "%s0.254 / +0.412 rad" % M) + row("지연", "0.11 s")
         + row("제동 한계", "4.85 m/s²") + row("ERPM 이득", "4204"), sub("뒤로"),
         now(("bag", "20260827-115713_map08x23_v9.2_161s_fastest-longest"))))

    # 16 training list, 17 one run
    runs = tr["runs"]
    items = ""
    for i, r in enumerate(runs):
        cls = "on" if r["name"] == "hs43b" else ("dim" if r["update"] >= r["updates"] else "")
        items += '<div class="%s">%s<i>%d / %d</i></div>' % (cls, r["name"], r["update"], r["updates"])
    hs = next(r for r in runs if r["name"] == "hs43b")
    strip = now(("코스", COURSE), ("update", "%d / %d" % (hs["update"], hs["updates"])), ("분당 바퀴", fnum(hs["laps_per_min"], 2)))
    page("16_train", "훈련", '''
<img class="bg" src="assets/train.png" alt=""><div class="film-l"></div><div class="film-t"></div>
<div class="title">훈련</div>
<div class="col" style="top:270px"><div class="menu m" style="width:636px">%s</div>%s</div>
%s''' % (items, sub("새 훈련", "뒤로"), strip))

    S = tr["hs43b"]
    u = np.array(S["update"], float)

    def smooth(a, k=25):
        a = np.asarray(a, float)
        pad = np.concatenate([np.full(k // 2, a[:k].mean()), a, np.full(k // 2, a[-k:].mean())])
        return np.convolve(pad, np.ones(k) / k, "valid")
    X0, X1 = 800, 1790
    xs = X0 + (u - u[0]) / (u[-1] - u[0]) * (X1 - X0)
    FU = 900
    fx = float(np.interp(FU, u, xs))
    svg = '<svg class="ov" viewBox="0 0 1920 1200" font-family="Pretendard" fill="#f4f4f1">'
    svg += '<line x1="%.1f" y1="262" x2="%.1f" y2="716" stroke="#f4f4f1" stroke-opacity=".45" stroke-width="2"/>' % (fx, fx)
    svg += '<text x="%.1f" y="244" font-size="28" font-weight="700" text-anchor="middle">u%d</text>' % (fx, FU)
    for key, label, y0, y1, nd in (("laps_per_min", "분당 바퀴", 310, 470, 2), ("boxes_per_min", "분당 상자 접촉", 590, 700, 2)):
        a = smooth(S[key], 15)
        lo_, hi_ = a.min(), a.max()
        ys = y1 - (a - lo_) / (hi_ - lo_) * (y1 - y0)
        svg += '<polyline points="%s" fill="none" stroke="#f4f4f1" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>' % pts(np.stack([xs, ys], 1)[::2])
        svg += '<text x="%d" y="%d" font-size="22" font-weight="500" fill-opacity=".6">%s</text>' % (X0, y0 - 30, label)
        svg += '<text x="%d" y="%.0f" font-size="30" font-weight="700" text-anchor="start">%s</text>' % (X1 + 18, ys[-1] + 10, fnum(S[key][-1], nd))
        for e in hs["exams"]:
            ex = float(np.interp(e["u"], u, xs))
            ey = float(np.interp(e["u"], u, ys))
            if e["u"] == FU:
                svg += '<circle cx="%.1f" cy="%.1f" r="11" fill="#f4f4f1" stroke="#0e0f11" stroke-width="3"/>' % (ex, ey)
            elif key == "laps_per_min":
                svg += '<circle cx="%.1f" cy="%.1f" r="6" fill="#0e0f11" stroke="#f4f4f1" stroke-width="2.5"/>' % (ex, ey)
    svg += '<text x="%d" y="756" font-size="22" font-weight="500" fill-opacity=".6">u%d</text>' % (X0, u[0])
    svg += '<text x="%d" y="756" font-size="22" font-weight="500" fill-opacity=".6" text-anchor="end">u%d / %d</text>' % (X1, u[-1], hs["updates"])
    svg += "</svg>"
    ex = next(e for e in hs["exams"] if e["u"] == FU)
    page("17_train_run", "훈련 실행", '''
<img class="bg" src="assets/train.png" alt=""><div class="film-l"></div><div class="film-t" style="height:820px"></div>%s
<div class="title">hs43b</div>
<div class="col"><div class="rows">%s</div><div class="go">달려 보기</div>%s</div>
%s''' % (svg, row("지점", "u%d" % FU, "on lr") + row("시험 랩", "%.1f s" % ex["lap"]) + row("시험 상자 접촉", fnum(ex["boxes"], 2))
         + row("시험 벽 접촉", fnum(ex["walls"], 2)), sub("멈춤", "뒤로"), ""))

    # 18 records
    i0 = next(i for i, r in enumerate(rec) if r["id"] == SID)
    show = rec[max(0, i0 - 2):max(0, i0 - 2) + 7]
    rows = ""
    for r in show:
        rows += row(r["when"], fnum(r["best"], 3), "s on" if r["id"] == SID else "s", mid=r["driver"]).replace("<em>", '<em class="l">').replace("<label>", '<label class="w">')
    f = rec[i0]
    page("18_records", "기록", '''
<img class="bg" src="assets/replay.png" alt=""><div class="film-l"></div><div class="film-t"></div>
<div class="title">기록</div>
<div class="col"><div class="rows">%s</div><div class="go">다시 보기</div>%s</div>
%s''' % (rows, sub("뒤로"), now(("코스", f["course"]), ("시간", "%d초" % f["dur"]), ("무충돌 연속", "%d" % f["streak"]), ("최고 랩", fnum(f["best"], 3)))))


if __name__ == "__main__":
    build()
