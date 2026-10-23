"""Check the extreme-skew fix (alpha = 0.1) before rerunning every ShieldDFL run."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R  # noqa: E402

if __name__ == "__main__":
    cfgs = [dict(R.BASE, dataset=d, seed=s, method="shield", attack="alie", ratio=0.2, alpha=0.1)
            for d in ("cicids2018", "cicddos2019") for s in (0, 1)]
    with ProcessPoolExecutor(4) as ex:
        for c, r in zip(cfgs, ex.map(R.run_one, cfgs)):
            print(c["dataset"], c["seed"], f"acc {100 * r['accuracy']:.2f} worst {100 * r['acc_worst']:.1f}",
                  flush=True)
