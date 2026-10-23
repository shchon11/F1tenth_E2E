"""Analysis half of contact_probe.py, in torch (the venv's sklearn 0.23 does not import against its scipy).
Fits L2 logistic probes for "a hard car-contact onset within the next 0.5 s" on the first seed's cells, scores ROC AUC on
the other seed's: on the TRUE relative state, on the actor's features [e | y], on both; and reads the nearest-car head
against the truth (closing speed, the opponent's sideways speed), in close situations and in the last 0.5 s before hits.
usage: contact_probe_analyze.py OUT_DIR [TRAIN_SEED]"""
import json
import os
import sys

import torch

OUT = sys.argv[1]
data = torch.load(os.path.join(OUT, "probe_data.pt"))
seeds = sorted({k[0] for k in data})
tr_seed = int(sys.argv[2]) if len(sys.argv) > 2 else seeds[0]
tr = [k for k in data if k[0] == tr_seed]
te = [k for k in data if k[0] != tr_seed]
cat = lambda keys, f: torch.cat([data[k][f].float() for k in keys])


def auc(score, y):
    """ROC AUC by ranks (ties averaged)."""
    y = y.bool()
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = score.argsort()
    ranks = torch.empty_like(score); ranks[order] = torch.arange(1, len(score) + 1, dtype=score.dtype)
    # average ranks over ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe(Xtr, ytr, Xte, l2=1e-2, steps=400):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    pos = ytr.float().mean().clamp(1e-4, 1 - 1e-4)
    wt = torch.where(ytr.bool(), 0.5 / pos, 0.5 / (1 - pos))            # balanced classes
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xtr @ w + b
        loss = (wt * torch.nn.functional.binary_cross_entropy_with_logits(z, ytr.float(), reduction="none")).mean() + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return Xte @ w + b


res = {"train_seed": tr_seed, "n_train": int(cat(tr, "y").numel()), "pos_train": int(cat(tr, "y").sum()),
       "n_test": int(cat(te, "y").numel()), "pos_test": int(cat(te, "y").sum())}
ytr, yte = cat(tr, "y"), cat(te, "y")
for name, f in (("truth", lambda k: cat(k, "gt")), ("features", lambda k: cat(k, "z")),
                ("both", lambda k: torch.cat([cat(k, "gt"), cat(k, "z")], 1)),
                ("distance_only", lambda k: cat(k, "gt")[:, :1])):
    s = probe(f(tr), ytr, f(te))
    res[f"auc_{name}"] = round(auc(s, yte), 4)
    print(f"probe on {name}: AUC {res[f'auc_{name}']}", flush=True)
# does the state ENCODE the two quantities that set a contact's severity? ridge regression features -> truth, R^2 on the
# other seed (closing speed: gt[:, 3]; the opponent's sideways speed: gt[:, 4])
def ridge_r2(Xtr, ttr, Xte, tte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    A, Ate = (Xtr - mu) / sd, (Xte - mu) / sd
    A1, Ate1 = torch.cat([A, torch.ones(len(A), 1)], 1), torch.cat([Ate, torch.ones(len(Ate), 1)], 1)
    reg = lam * torch.eye(A1.shape[1]); reg[-1, -1] = 0
    w = torch.linalg.solve(A1.T @ A1 + reg, A1.T @ ttr)
    p = Ate1 @ w
    return float(1 - ((p - tte) ** 2).sum() / ((tte - tte.mean()) ** 2).sum())


Ztr, Zte, Gtr, Gte = cat(tr, "z").double(), cat(te, "z").double(), cat(tr, "gt").double(), cat(te, "gt").double()
for col, nm in ((3, "closing"), (4, "v_lat"), (0, "distance")):
    res[f"ridge_r2_{nm}"] = round(ridge_r2(Ztr, Gtr[:, col], Zte, Gte[:, col]), 3)
    print(f"features -> true {nm}: R^2 {res[f'ridge_r2_{nm}']}", flush=True)
if all(data[k]["near"] is not None for k in data):
    keys = list(data)
    near, gt, y = cat(keys, "near"), cat(keys, "gt"), cat(keys, "y").bool()
    pc, pv, tc, tv = near[:, 3] * 5.0, near[:, 4] * 2.0, gt[:, 3], gt[:, 4]
    r2 = lambda p, t: float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum().clamp_min(1e-9))
    res["near_head"] = {
        "closing_r2_close": round(r2(pc, tc), 3), "v_lat_r2_close": round(r2(pv, tv), 3),
        "closing_mae_close": round(float((pc - tc).abs().mean()), 3), "v_lat_mae_close": round(float((pv - tv).abs().mean()), 3),
        "closing_true_pre_hit": round(float(tc[y].mean()), 3), "closing_pred_pre_hit": round(float(pc[y].mean()), 3),
        "closing_mae_pre_hit": round(float((pc - tc)[y].abs().mean()), 3),
        "v_lat_true_abs_pre_hit": round(float(tv[y].abs().mean()), 3), "v_lat_pred_abs_pre_hit": round(float(pv[y].abs().mean()), 3),
        "v_lat_mae_pre_hit": round(float((pv - tv)[y].abs().mean()), 3),
        "closing_true_other": round(float(tc[~y].mean()), 3), "closing_pred_other": round(float(pc[~y].mean()), 3)}
    # the head's own reading as a contact predictor: its closing-speed estimate alone, vs the true one alone
    res["auc_head_closing_alone"] = round(auc(pc, y.float()), 4)
    res["auc_true_closing_alone"] = round(auc(tc, y.float()), 4)
    print("nearest-car head:", json.dumps(res["near_head"]), flush=True)
json.dump(res, open(os.path.join(OUT, "probe_result.json"), "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
