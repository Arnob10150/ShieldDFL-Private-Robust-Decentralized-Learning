"""One experiment = (dataset, method, attack, malicious ratio, DP epsilon, seed, ablation).
Each finished run is appended as one JSON line to v2/results/runs.jsonl and skipped on
re-launch, so the grid can be interrupted and resumed.

Run a grid:  python -I v2/run.py --grid main|poison|privacy|ablation|scale [--workers 6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RES = HERE / "results"
RES.mkdir(exist_ok=True)
OUT = RES / "runs.jsonl"
DATASETS = ["cicddos2019", "cicids2018", "edgeiiot"]
DFL = ["gossip", "median", "trimmed", "krum", "clipped", "balance", "shield"]
# six generic attacks, two amplified variants, two written against ShieldDFL's own gates
ATTACKS = ["label_flip", "sign_flip", "sign_flip10", "scaling", "gaussian", "alie", "ipm",
           "ipm5", "adaptive", "adaptive_dist", "adaptive_rep", "adaptive_col"]
SEEDS = [0, 1, 2, 3, 4]
BASE = dict(clients=20, degree=6, rounds=30, alpha=0.5, eps=None, attack="none", ratio=0.0,
            abl="")


def key(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def done_keys():
    if not OUT.exists():
        return set()
    return {json.loads(l)["key"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}


# ------------------------------------------------------------------ data
def load_split(name, seed, n_clients, alpha):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    z = np.load(HERE / "data" / f"{name}.npz", allow_pickle=False)
    X, y = z["X"], z["y"]
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    sc = StandardScaler().fit(xtr)
    f = lambda a: np.clip(sc.transform(a), -10, 10).astype(np.float32)
    xtr, xte = f(xtr), f(xte)
    rng = np.random.default_rng(seed)
    # every client must hold enough rows to train and to keep a held-out split; the floor
    # scales with the federation size so that large n does not make the draw unsatisfiable
    floor = max(40, min(100, int(0.25 * len(ytr) / n_clients)))
    best = None
    for attempt in range(200):
        parts = [[] for _ in range(n_clients)]
        for c in (0, 1):
            idx = np.where(ytr == c)[0]; rng.shuffle(idx)
            cuts = (np.cumsum(rng.dirichlet(np.full(n_clients, alpha))) * len(idx)).astype(int)[:-1]
            for i, part in enumerate(np.split(idx, cuts)):
                parts[i].extend(part.tolist())
        lo = min(len(p) for p in parts)
        if best is None or lo > best[0]:
            best = (lo, parts)
        if lo >= floor:
            break
    parts = best[1]
    # a Dirichlet draw over many clients can leave one with too few rows to train on; top
    # the smallest up from the largest so that every client can still hold out a split
    need = max(60, floor)
    order = sorted(range(n_clients), key=lambda i: len(parts[i]))
    for i in order:
        while len(parts[i]) < need:
            j = max(range(n_clients), key=lambda k: len(parts[k]))
            if len(parts[j]) <= need:
                break
            parts[i].append(parts[j].pop())
    clients = []
    for p in parts:
        p = np.array(p); rng.shuffle(p)
        nv = max(20, int(0.15 * len(p)))            # local held-out split, never shared
        clients.append((p[nv:], p[:nv]))
    return xtr, ytr, xte, yte, clients


# ------------------------------------------------------------------ one run
def run_one(cfg):
    import torch
    torch.set_num_threads(1)
    import shield as S
    t0 = time.time()
    seed = cfg["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xtr, ytr, xte, yte, clients = load_split(cfg["dataset"], seed, cfg["clients"], cfg["alpha"])
    Xt, Yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    Xte = torch.from_numpy(xte)
    d = xtr.shape[1]
    tc = S.TrainCfg()
    T, n = cfg["rounds"], cfg["clients"]
    out = dict(cfg)

    def evaluate(model):
        return S.metrics(yte, S.predict(model, Xte))

    m = cfg["method"]
    # ---------------- reference settings that do not use the gossip protocol
    if m == "centralized":
        model = S.FlowMLP(d)
        idx = np.concatenate([c[0] for c in clients])
        S.train_local(model, Xt[idx], Yt[idx], S.TrainCfg(epochs=10), gen)
        out.update(evaluate(model), seconds=time.time() - t0, key=key(cfg))
        return out
    if m == "local":
        rows = []
        for tr, _ in clients:
            model = S.FlowMLP(d)
            S.train_local(model, Xt[tr], Yt[tr], S.TrainCfg(epochs=T), gen)
            rows.append(evaluate(model))
        out.update({k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
                   seconds=time.time() - t0, key=key(cfg))
        return out

    # ---------------- DP calibration (per-client noise for a common target epsilon)
    # every client meets the same (eps, delta) on its own records; the noise multiplier
    # therefore depends on its own dataset size (record-level DP, RDP accounting)
    tcs = [S.TrainCfg() for _ in clients]
    if cfg["eps"] is not None:
        sig_eps = [S.sigma_for_epsilon(float(cfg["eps"]), len(c[0]), tc, T) for c in clients]
        for t_, (s_, _) in zip(tcs, sig_eps):
            t_.sigma = s_
        out["sigma_mean"] = float(np.mean([s for s, _ in sig_eps]))
        out["eps_accounted_max"] = float(max(e for _, e in sig_eps))

    rng = np.random.default_rng(seed + 7)
    n_bad = int(round(cfg["ratio"] * n))
    bad = set(rng.permutation(n)[:n_bad].tolist())
    benign = [i for i in range(n) if i not in bad]
    model = S.FlowMLP(d)
    x0 = S.flat(model)
    xs = [x0.clone() for _ in range(n)]
    states = [S.ShieldState() for _ in range(n)]
    abl = cfg["abl"]
    sh_kw = dict(use_val="val" not in abl, use_dist="dist" not in abl,
                 use_clip="clip" not in abl, use_rep="rep" not in abl)
    xv = [(torch.from_numpy(xtr[c[1]]), torch.from_numpy(ytr[c[1]])) for c in clients]
    # the gate-aware attackers collude and pool their own held-out records; this surrogate
    # stands in for the private validation split each receiver screens with
    xsur = ysur = None
    if cfg["attack"] in S.ADAPTIVE and bad:
        sur = np.concatenate([clients[i][1] for i in sorted(bad)])
        xsur, ysur = torch.from_numpy(xtr[sur]), torch.from_numpy(ytr[sur])
    curve, comm = [], 0
    nparam = x0.numel()
    for t in range(T):
        new = []
        for i, (tr, _) in enumerate(clients):
            S.load_flat(model, xs[i])
            S.train_local(model, Xt[tr], Yt[tr], tcs[i], gen,
                          flip_labels=(i in bad and cfg["attack"] in
                                       ("label_flip",) + S.ADAPTIVE))
            new.append(S.flat(model))
        if m == "fedavg":                                   # server reference (clean only)
            w = np.array([len(c[0]) for c in clients], float); w /= w.sum()
            g = sum(float(wi) * v for wi, v in zip(w, new))
            xs = [g.clone() for _ in range(n)]
            comm += 2 * n * nparam * 4
        else:
            bstack = torch.stack([new[i] for i in benign])
            bupd = torch.stack([new[i] - xs[i] for i in benign]).mean(0)
            if cfg["attack"] in S.ADAPTIVE and bad:
                # one crafted model serves every attacker: they collude, so the model that
                # is best against the screening rule is best for all of them
                kw_a = S.adaptive_kwargs(cfg["attack"], t, T)
                xadv = None
                if kw_a is not None:
                    x_bad = torch.stack([new[i] for i in sorted(bad)]).mean(0)
                    xadv = S.adaptive_model(bstack, x_bad, model, xsur, ysur, **kw_a)
                sent = [(xadv if xadv is not None else new[i]) if i in bad else new[i]
                        for i in range(n)]
            else:
                sent = [S.malicious_model(cfg["attack"], xs[i], new[i], bstack, gen, benign_upd=bupd)
                        if i in bad else new[i] for i in range(n)]
            nb = S.neighbours(n, cfg["degree"], rng)
            nxt = []
            for i in range(n):
                if i in bad:          # attackers keep a normal training state and only
                    nxt.append(new[i]); continue      # *send* crafted models

                peers = [sent[j] for j in nb[i]]
                comm += len(peers) * nparam * 4
                kw = dict(t=t, T=T, peer_ids=nb[i], state=states[i], model=model,
                          xv=xv[i][0], yv=xv[i][1], f=max(1, n_bad * cfg["degree"] // n + 1))
                if m == "shield":
                    kw.update(sh_kw)
                nxt.append(S.RULES[m](new[i], peers, **kw))
            xs = nxt
        if (t + 1) % 5 == 0 or t == T - 1:
            accs = []
            for i in benign:
                S.load_flat(model, xs[i]); accs.append(evaluate(model)["accuracy"])
            curve.append([t + 1, float(np.mean(accs))])
    rows = []
    for i in benign:
        S.load_flat(model, xs[i]); rows.append(evaluate(model))
    out.update({k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]})
    out["acc_worst"] = float(min(r["accuracy"] for r in rows))
    out["acc_std_clients"] = float(np.std([r["accuracy"] for r in rows]))
    out.update(curve=curve, comm_mb=comm / 2 ** 20, seconds=time.time() - t0, key=key(cfg))
    return out


# ------------------------------------------------------------------ grids
def grid(name):
    seeds = SEEDS
    cfgs = []
    add = lambda **kw: cfgs.append({**BASE, **kw})
    if name == "main":
        for ds in DATASETS:
            for s in seeds:
                for m in ["centralized", "local", "fedavg"] + DFL:
                    add(dataset=ds, seed=s, method=m)
    elif name == "poison":
        # every attack, including the two gate-aware ones, up to 40% malicious peers
        for ds in DATASETS:
            for s in seeds:
                for a in ATTACKS:
                    for r in (0.2, 0.3, 0.4):
                        for m in DFL:
                            add(dataset=ds, seed=s, method=m, attack=a, ratio=r)
    elif name == "breaking":
        # the regime in which a rule is expected to fail: half the peers malicious, and
        # sparse graphs in which a neighbourhood is easily dominated by attackers
        for ds in DATASETS:
            for s in seeds[:3]:
                for a in ("label_flip", "ipm5", "adaptive", "adaptive_col"):
                    for m in DFL:
                        add(dataset=ds, seed=s, method=m, attack=a, ratio=0.5)
                        for dg in (3, 10):
                            add(dataset=ds, seed=s, method=m, attack=a, ratio=0.3, degree=dg)
    elif name == "privacy":
        for ds in DATASETS:
            for s in seeds:
                for e in (1, 2, 4, 8):
                    for m in ("gossip", "shield"):
                        add(dataset=ds, seed=s, method=m, eps=e)
                    for a in ("alie", "adaptive"):
                        add(dataset=ds, seed=s, method="shield", eps=e, attack=a, ratio=0.2)
    elif name == "ablation":
        # ablated under the gate-aware attacks as well, where each component should matter
        for ds in DATASETS:
            for s in seeds:
                for a in ("scaling", "alie", "label_flip", "adaptive", "adaptive_dist",
                          "adaptive_rep", "adaptive_col"):
                    # single removals cannot separate components that substitute for one
                    # another, so the grid also removes them in pairs and as a group
                    for ab in ("", "noval", "nodist", "noclip", "norep",
                               "nodistnoclip", "novalnodist", "novalnodistnoclip",
                               "novalnodistnoclipnorep"):
                        add(dataset=ds, seed=s, method="shield", attack=a, ratio=0.3, abl=ab)
    elif name == "scale":
        for ds in DATASETS:
            for s in seeds:
                for al in (0.1, 1.0, 100.0):
                    for m in ("gossip", "median", "clipped", "balance", "shield"):
                        for a in ("alie", "adaptive"):
                            add(dataset=ds, seed=s, method=m, alpha=al, attack=a, ratio=0.2)
    elif name == "nodes":
        # larger federations at a fixed neighbourhood size
        for ds in DATASETS:
            for s in seeds[:3]:
                for nc in (40, 60):
                    for m in ("gossip", "median", "clipped", "balance", "shield"):
                        add(dataset=ds, seed=s, method=m, clients=nc, attack="adaptive", ratio=0.3)
    else:
        raise SystemExit(f"unknown grid {name}")
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs.jsonl")
    a = ap.parse_args()
    global OUT
    OUT = RES / a.out
    have = done_keys()
    todo = [c for c in grid(a.grid) if key(c) not in have]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.grid}: {len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, c): c for c in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            c = futs[fu]
            try:
                r = fu.result()
            except Exception as e:   # one failed run must not stop the grid
                print(f"FAIL {c}: {type(e).__name__}: {e}", flush=True)
                continue
            with OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"[{k}/{len(todo)}] {c['dataset']} {c['method']} {c['attack']} {c['ratio']} "
                  f"eps={c['eps']} s{c['seed']} {c['abl']} acc {r['accuracy']:.4f} "
                  f"({r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
