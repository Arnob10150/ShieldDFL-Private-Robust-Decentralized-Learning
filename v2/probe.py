"""Quick diagnostic sweep (not part of the paper grids): every attack at 30% on one
dataset and seed, for ShieldDFL and the strongest decentralized baselines.

Run: python -I v2/probe.py [dataset] [workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R  # noqa: E402

if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "cicids2018"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    attacks = ["ipm", "sign_flip", "scaling", "label_flip", "alie", "gaussian"]
    methods = ["shield", "median", "clipped", "gossip", "balance"]
    cfgs = [dict(R.BASE, dataset=ds, seed=0, method=m, attack=a, ratio=0.3)
            for a in attacks for m in methods]
    with ProcessPoolExecutor(w) as ex:
        for c, r in zip(cfgs, ex.map(R.run_one, cfgs)):
            print(f"{c['attack']:11s} {c['method']:8s} acc {r['accuracy']:.4f} "
                  f"worst {r['acc_worst']:.4f}", flush=True)
