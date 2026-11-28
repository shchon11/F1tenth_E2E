import numpy as np, pickle, windows as WN
fit, val, pre = WN.splits()
out = {}
for name, fs in (("fit", fit), ("val", val), ("pre", pre)):
    print("==", name, flush=True)
    out[name] = WN.build(fs)
    print(name, out[name]["r"].shape, flush=True)
pickle.dump(out, open("windows.pkl", "wb"))
