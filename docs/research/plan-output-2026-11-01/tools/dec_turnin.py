"""(11-27) Does the policy's state know that the other car will TURN IN? On the held-out sessions of an exam4 decoder
(state_decoder_train.py's test split, out/decoder/<record>_decin.npz), at every decision with the other car visible 1-4 m
ahead of us, the decoder's 'other car in 1.0 s' (our frame at the decision) against (a) the truth and (b) a straight-line
guess (its heading and speed now, 1 s on), split into: approaching the left hairpin (lap fraction 0.45-0.58, where 56 % of
rc42a u950's closing rear-ends were) and elsewhere. If the decoder is no better than the straight line at the hairpin, the
turn-in is not in the state.  usage: dec_turnin.py RECORD_NAME (e.g. exam4_rc42a_u900_ideal_race4_n256)"""
import sys

import numpy as np

import dec_replay as DR

rec = sys.argv[1]
z = np.load(DR.record_path(rec)); d = DR.decoder(rec); di = DR.decin(rec)
L = float(z["lapL"][0]); dt = float(z["dt"]); K = int(round(1.0 / dt))
sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))
rows = []
for k, s in enumerate(di["sessions"]):
    s = int(s)
    feat, mem = di["feat"][:, k].astype(np.float32), di["mem"][:, k].astype(np.float32)
    T = feat.shape[0]
    o = d.run(feat, mem, "feat+mem"); q = o["opp"]; S = d.opp_scale
    st = z["st"][:T + 1, s].astype(float); op = z["op"][:T + 1, s].astype(float); ors = z["ors"][:T, s]
    for t in range(0, T - K):
        if ors[t:t + K + 1].any() or not z["on"][t, s]:
            continue
        x, y, yw = st[t, :3]; c, sn = np.cos(yw), np.sin(yw)
        to_ego = lambda X, Y: ((X - x) * c + (Y - y) * sn, -(X - x) * sn + (Y - y) * c)
        ox, oy = to_ego(op[t, 0], op[t, 1])
        if not (1.0 <= ox <= 4.0 and abs(oy) < 1.0):
            continue
        tx, ty = to_ego(op[t + K, 0], op[t + K, 1])                     # truth in 1 s, our frame now
        sx, sy = to_ego(op[t, 0] + np.cos(op[t, 2]) * op[t, 3] * 1.0, op[t, 1] + np.sin(op[t, 2]) * op[t, 3] * 1.0)
        dx, dy = q[t, 8] * S, q[t, 9] * S                                  # the decoder's 1.0 s (outputs: there, x, y, cos, sin, v, x0.5, y0.5, x1.0, y1.0)
        p = sig(q[t, 0])
        lapq = np.mod(z["orem"][t, s], L) / L
        turn = np.unwrap([op[t, 2], op[t + K, 2]])[1] - op[t, 2]
        rows.append((0.45 <= lapq < 0.58, np.hypot(dx - tx, dy - ty), np.hypot(sx - tx, sy - ty), dy - ty, sy - ty, turn, p))
a = np.array(rows, float)
print(f"== {rec}: held-out sessions {len(di['sessions'])}, decisions with the other car visible 1-4 m ahead {len(a)}")
for m, nm in ((a[:, 0] > 0, "왼쪽 헤어핀 진입 (바퀴 0.45-0.58)"), (a[:, 0] == 0, "그 밖")):
    b = a[m]
    if not len(b):
        continue
    print(f"  {nm}: {len(b)} 결정 | 1 s 동안 상대가 꺾는 각 중앙값 {np.degrees(np.median(np.abs(b[:, 5]))):.0f}° | "
          f"1 s 뒤 위치 오차 중앙값: 디코더 {np.median(b[:, 1]):.2f} m · 직진 짐작 {np.median(b[:, 2]):.2f} m | "
          f"옆(꺾는 쪽) 오차 중앙값: 디코더 {np.median(np.abs(b[:, 3])):.2f} m · 직진 {np.median(np.abs(b[:, 4])):.2f} m | "
          f"디코더가 직진보다 나은 결정 {100 * np.mean(b[:, 1] < b[:, 2]):.0f} % | 상대 있음 P 중앙값 {np.median(b[:, 6]):.2f}")
