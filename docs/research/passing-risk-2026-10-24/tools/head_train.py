"""Research (2026-10-24): retrain the v2 actor's nearest-car head on its own frozen features (head_data.py), for the
deployable shield. Same architecture (aux_v2.NearestCarHead, 384 -> 256 -> 5), started from the checkpoint's head.
Loss: presence BCE + where a car is seen W_POS (dx^2 + dy^2) + W_CL closing^2 + W_VL v_lat^2 (normalised units, as
`nearest_car_loss`; the original weights were 10 / 1 / 0.1), with samples in the last 0.5 s before a hard car contact
weighted x PRE. Reports, on the validation file, R2 / mean abs error of each reading for a seen car ahead and for the
pre-contact samples, old head vs new, and writes a copy of the checkpoint with only `actor.near_car.*` replaced (the
action path is untouched: the head does not feed the action).
usage: head_train.py CKPT TRAIN.pt VAL.pt OUT_CKPT [W_POS=10] [W_CL=5] [W_VL=5] [PRE=5] [EPOCHS=30]"""
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CK, TR, VA, OUT = sys.argv[1:5]
W_POS = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
W_CL = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0
W_VL = float(sys.argv[7]) if len(sys.argv) > 7 else 5.0
PRE = float(sys.argv[8]) if len(sys.argv) > 8 else 5.0
EPOCHS = int(sys.argv[9]) if len(sys.argv) > 9 else 30
torch.manual_seed(0)


def load(p):
    d = torch.load(p)
    return (torch.cat([v["z"] for v in d.values()]).float(), torch.cat([v["tgt"] for v in d.values()]),
            torch.cat([v["gt"] for v in d.values()]), torch.cat([v["y"] for v in d.values()]).float())


ck = torch.load(CK, map_location="cpu", weights_only=True)
sd = ck["state_dict"]
head = nn.Sequential(nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 5))
head.load_state_dict({k.replace("actor.near_car.net.", ""): v for k, v in sd.items() if k.startswith("actor.near_car.net.")})
old = nn.Sequential(nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 5)); old.load_state_dict(head.state_dict())
zt, tt, gt_t, yt = load(TR)
zv, tv, gt_v, yv = load(VA)
print(f"train {len(zt)} samples ({int((tt[:, 0] > 0).sum())} seen, {int(yt.sum())} pre-contact); val {len(zv)}", flush=True)


def loss_fn(pred, tgt, w):
    ex = tgt[:, 0]
    bce = F.binary_cross_entropy_with_logits(pred[:, 0], ex, reduction="none")
    reg = (W_POS * ((pred[:, 1] - tgt[:, 1]) ** 2 + (pred[:, 2] - tgt[:, 2]) ** 2) + W_CL * (pred[:, 3] - tgt[:, 3]) ** 2
           + W_VL * (pred[:, 4] - tgt[:, 4]) ** 2)
    return ((bce + ex * reg) * w).sum() / w.sum()


def report(net, z, tgt, y, tag):
    with torch.no_grad():
        p = net(z)
    seen = tgt[:, 0] > 0
    out = {}
    for name, m in (("seen_ahead", seen & (tgt[:, 1] > 0)), ("pre_contact_seen", seen & (y > 0))):
        row = {"n": int(m.sum()), "exists_acc": float(((p[:, 0] > 0) == seen).float().mean())}
        for i, (k, sc) in enumerate((("dx", 5.0), ("dy", 2.0), ("closing", 5.0), ("v_lat", 2.0)), start=1):
            a, b = p[m, i] * sc, tgt[m, i] * sc
            row[k] = {"r2": float(1 - ((a - b) ** 2).mean() / b.var().clamp_min(1e-9)), "mae": float((a - b).abs().mean())}
        out[name] = row
    fp = float(((p[:, 0] > 0) & ~seen).float().mean()); fn = float(((p[:, 0] <= 0) & seen).float().mean())
    out["exists_fp"], out["exists_fn"] = fp, fn
    print(tag, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv["r2"], 3), round(vv["mae"], 3)) if isinstance(vv, dict) else vv
                                                                   for kk, vv in v.items()}) for k, v in out.items()}), flush=True)
    return out


r_old = report(old, zv, tv, yv, "OLD head (val)")
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
w_t = 1.0 + PRE * yt
n = len(zt)
best, best_sd = float("inf"), None                      # early stopping on the validation loss (the first try overfit:
for ep in range(EPOCHS):                                # val 0.218 at epoch 5 -> 0.259 at 30)
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, 4096):
        idx = perm[i:i + 4096]
        loss = loss_fn(head(zt[idx]), tt[idx], w_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()) * len(idx)
    with torch.no_grad():
        lv = float(loss_fn(head(zv), tv, 1.0 + PRE * yv))
    if lv < best:
        best, best_sd = lv, {k: v.clone() for k, v in head.state_dict().items()}
    print(f"epoch {ep + 1}: train {tot / n:.4f} | val {lv:.4f}{' *' if lv == best else ''}", flush=True)
head.load_state_dict(best_sd)
r_new = report(head, zv, tv, yv, "NEW head (val)")
new_sd = dict(sd)
for k, v in head.state_dict().items():
    new_sd["actor.near_car.net." + k] = v.clone()
ck2 = dict(ck); ck2["state_dict"] = new_sd
extra = dict(ck2.get("extra") or {})
extra["near_car_head_retrained"] = {"from": CK, "train": TR, "val": VA, "w_pos": W_POS, "w_closing": W_CL, "w_vlat": W_VL,
                                    "pre_weight": PRE, "epochs": EPOCHS, "note": "head only; the action path is untouched"}
ck2["extra"] = extra
torch.save(ck2, OUT)
json.dump({"old": r_old, "new": r_new}, open(OUT + ".report.json", "w"), indent=1)
print("saved", OUT, flush=True)
