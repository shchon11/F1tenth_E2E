"""Average the weights of several checkpoints into one policy.

    python3 scripts/average_checkpoints.py out.pt ~/f1sim_runs/ppo_v17/ppo_u{1000,1100,1200}.pt

Late in a racing run the policy oscillates between cautious and aggressive from checkpoint to
checkpoint (ppo_v17 swung between 0.26 and 0.38 contacts per car per 20 s with no trend), so the last
checkpoint is a lottery ticket. Averaging a few late ones lands between those extremes and in practice
beats each of them: for ppo_v17 the average of updates 1000/1100/1200 had both the fewest contacts and
the most overtakes of the three. Buffers (running statistics) are taken from the last checkpoint, not
averaged; integer tensors are copied.
"""
import argparse, copy, os
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    cks = [torch.load(os.path.expanduser(p), map_location="cpu", weights_only=False) for p in a.ckpts]
    out = copy.deepcopy(cks[-1])
    key = "model" if "model" in out else "state_dict"
    ref = out[key]
    if any(set(c[key]) != set(ref) for c in cks):
        raise SystemExit("checkpoints have different parameter sets")
    out[key] = {k: (sum(c[key][k].float() for c in cks) / len(cks)).to(v.dtype) if v.is_floating_point() else v
                for k, v in ref.items()}
    meta = out.get("extra") or {}
    meta["averaged_from"] = [os.path.basename(p) for p in a.ckpts]
    out["extra"] = meta
    torch.save(out, os.path.expanduser(a.out))
    print(f"wrote {a.out} = mean of {len(cks)} checkpoints: {meta['averaged_from']}")


if __name__ == "__main__":
    main()
