"""ShieldDFL v2: trust-aware, poisoning-resilient decentralized federated learning with
record-level differential privacy, plus every baseline and attack used in the paper.

Protocol (one round t, every client i in parallel):
  1. local training from x_i (optionally DP-SGD with per-sample clipping) -> x_i'
  2. Byzantine clients replace x_i' by a malicious model
  3. i receives x_j' from its neighbours N_i(t) on a random d-regular-style graph
  4. i aggregates {x_i'} U {x_j'} with the method under test -> x_i
Raw data never leaves a client; there is no server. Metrics are computed for benign
clients only, on a common held-out test set, with a fixed 0.5 decision threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap


# ------------------------------------------------------------------ model
class FlowMLP(nn.Module):
    """Flow classifier; LayerNorm instead of BatchNorm so per-sample DP gradients exist."""

    def __init__(self, d_in: int, h=(128, 64), emb=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h[0]), nn.LayerNorm(h[0]), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h[0], h[1]), nn.ReLU(), nn.Linear(h[1], emb), nn.ReLU(),
            nn.Linear(emb, 2))

    def forward(self, x):
        return self.net(x)


def flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def load_flat(model: nn.Module, vec: torch.Tensor) -> None:
    off = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[off:off + n].view_as(p))
            off += n


# ------------------------------------------------------------------ local training
@dataclass
class TrainCfg:
    epochs: int = 1
    batch: int = 64
    lr: float = 1e-3
    wd: float = 1e-4
    clip: float = 1.0          # DP per-sample clipping norm C
    sigma: float = 0.0         # DP noise multiplier (0 = no DP)


def train_local(model, x, y, cfg: TrainCfg, gen: torch.Generator, flip_labels=False):
    if flip_labels:
        y = 1 - y
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n = len(y)
    if cfg.sigma > 0:
        params = {k: v.detach() for k, v in model.named_parameters()}

        def loss_one(p, xi, yi):
            out = functional_call(model, p, (xi.unsqueeze(0),))
            return F.cross_entropy(out, yi.unsqueeze(0))
        ps_grad = vmap(grad(loss_one), in_dims=(None, 0, 0), randomness="different")
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, cfg.batch):
            idx = perm[s:s + cfg.batch]
            xb, yb = x[idx], y[idx]
            opt.zero_grad(set_to_none=True)
            if cfg.sigma <= 0:
                F.cross_entropy(model(xb), yb).backward()
            else:   # DP-SGD: clip every example's gradient to C, sum, add N(0, sigma^2 C^2)
                params = {k: v.detach() for k, v in model.named_parameters()}
                g = ps_grad(params, xb, yb)
                norms = torch.sqrt(sum(v.reshape(len(idx), -1).pow(2).sum(1) for v in g.values()))
                scale = (cfg.clip / (norms + 1e-12)).clamp(max=1.0)
                for k, p in model.named_parameters():
                    gs = (g[k] * scale.view(-1, *([1] * (g[k].dim() - 1)))).sum(0)
                    gs = gs + torch.randn(gs.shape, generator=gen) * cfg.sigma * cfg.clip
                    p.grad = gs / cfg.batch
            opt.step()


def dp_epsilon(sigma, n_local, cfg: TrainCfg, rounds, delta=1e-5):
    from opacus.accountants import RDPAccountant
    q = min(1.0, cfg.batch / n_local)
    steps = rounds * cfg.epochs * math.ceil(n_local / cfg.batch)
    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)]
    return float(acc.get_epsilon(delta=delta))


def sigma_for_epsilon(eps, n_local, cfg: TrainCfg, rounds, delta=1e-5):
    lo, hi = 0.3, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if dp_epsilon(mid, n_local, cfg, rounds, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi, dp_epsilon(hi, n_local, cfg, rounds, delta)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.softmax(model(x), 1)[:, 1].numpy()


@torch.no_grad()
def val_loss(model, x, y):
    model.eval()
    return float(F.cross_entropy(model(x), y))


def metrics(y, p):
    from sklearn.metrics import roc_auc_score
    yh = (p >= 0.5).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    prec1 = tp / max(1, tp + fp); rec1 = tp / max(1, tp + fn)
    prec0 = tn / max(1, tn + fn); rec0 = tn / max(1, tn + fp)
    f1 = lambda a, b: 2 * a * b / max(1e-12, a + b)
    try:
        auc = float(roc_auc_score(y, p)) if np.isfinite(p).all() else float("nan")
    except ValueError:
        auc = float("nan")
    return {"accuracy": (tp + tn) / len(y), "f1_macro": 0.5 * (f1(prec1, rec1) + f1(prec0, rec0)),
            "precision": prec1, "recall": rec1, "fpr": fp / max(1, fp + tn), "auc": auc,
            "tpr1": tpr_at_fpr(y, p, 0.01)}


# ------------------------------------------------------------------ topology
def neighbours(n, degree, rng):
    """Random undirected graph, re-drawn every round, each node with >= degree peers."""
    adj = [set() for _ in range(n)]
    for i in range(n):
        while len(adj[i]) < degree:
            j = int(rng.integers(n))
            if j != i:
                adj[i].add(j); adj[j].add(i)
    return [sorted(a) for a in adj]


# ------------------------------------------------------------------ attacks (model level)
def malicious_model(kind, x_old, x_new, benign_new, rng_t, scale=10.0, benign_upd=None):
    """x_old: attacker's model before training, x_new: after (honest or flipped training).
    benign_new: stacked benign models of this round, benign_upd: their mean update
    (used only by the omniscient ALIE and IPM attackers)."""
    upd = x_new - x_old
    if kind in ("none", "label_flip"):
        return x_new
    if kind == "sign_flip":
        return x_old - upd
    if kind == "sign_flip10":     # strong variant: reversed and amplified update
        return x_old - 10.0 * upd
    if kind == "ipm5" and benign_upd is not None:   # strong IPM, eps = 5
        return x_old - 5.0 * benign_upd
    if kind == "scaling":
        return x_old + scale * upd
    if kind == "gaussian":
        return x_new + torch.randn(x_new.shape, generator=rng_t) * 3.0 * upd.std()
    mu, sd = benign_new.mean(0), benign_new.std(0)
    if kind == "alie":        # "A little is enough" (Baruch et al., 2019), z for n=20, f=6
        return mu - 1.0 * sd
    if kind == "ipm":         # inner-product manipulation (Xie et al., 2020), eps = 0.5
        return x_old - 0.5 * benign_upd
    raise ValueError(kind)


# ------------------------------------------------------------------ aggregation rules
def agg_mean(own, peers, **_):
    return torch.stack([own] + peers).mean(0)


def agg_median(own, peers, **_):
    return torch.stack([own] + peers).median(0).values


def agg_trimmed(own, peers, beta=0.2, **_):
    s = torch.stack([own] + peers).sort(0).values
    k = int(beta * s.shape[0])
    return s[k:s.shape[0] - k].mean(0) if s.shape[0] > 2 * k else s.mean(0)


def agg_krum(own, peers, f=2, **_):
    s = torch.stack([own] + peers)
    d = torch.cdist(s, s) ** 2
    m = max(1, s.shape[0] - f - 2)
    score = d.sort(1).values[:, 1:m + 1].sum(1)
    return s[int(score.argmin())]


def agg_clipped_gossip(own, peers, **_):
    """ClippedGossip (He, Karimireddy & Jaggi, 2022) with an adaptive radius: the
    median neighbour distance, so honest peers are rarely clipped."""
    if not peers:
        return own
    diffs = torch.stack([p - own for p in peers])
    norms = diffs.norm(dim=1)
    tau = norms.median()
    clipped = diffs * (tau / (norms + 1e-12)).clamp(max=1.0).unsqueeze(1)
    return own + clipped.sum(0) / (len(peers) + 1)


def agg_balance(own, peers, t=0, T=1, gamma=0.3, kappa=1.0, alpha=0.5, **_):
    """BALANCE (Fang et al., 2024): accept a neighbour only if it lies within a shrinking
    ball around the client's own model."""
    thr = gamma * math.exp(-kappa * t / max(1, T)) * float(own.norm())
    ok = [p for p in peers if float((p - own).norm()) <= thr]
    return alpha * own + (1 - alpha) * torch.stack(ok).mean(0) if ok else own


@dataclass
class ShieldState:
    """Per-client reputation of every other client (EMA of instantaneous trust)."""
    rep: dict = field(default_factory=dict)


def agg_shield(own, peers, peer_ids, state: ShieldState, model, xv, yv, beta=0.8, lam=2.0,
               kappa=2.5, use_val=True, use_dist=True, use_clip=True, use_rep=True, **_):
    """ShieldDFL screening of neighbour models.

    For each neighbour model x_j (all statistics are over this round's neighbourhood, so a
    client whose own data is skewed does not reject honest peers for being different):
      validation gate  l_j <= med(l) + lam * MAD(l)       (loss on the client's held-out split)
      distance gate    ||x_j - x_own|| <= kappa * med(dist)
      reputation       r_j <- beta r_j + (1 - beta) 1[gates]   (soft weight, never a ban)
      aggregation      x_own + sum_j r_j 1[gates] clip(x_j - x_own, med(dist)) / (1 + sum r)
    """
    if not peers:
        return own
    # a validation split that lacks a class cannot rank models: the node's own one-class
    # model always looks best and every honest neighbour would be rejected
    if use_val and int(torch.bincount(yv, minlength=2).min()) < 5:
        use_val = False
    diffs = [p - own for p in peers]
    dist = torch.tensor([float(d.norm()) for d in diffs])
    # the client's own model is the one point it can trust; including it (distance 0, its
    # own validation loss) keeps the medians honest while at most half the peers are bad
    tau = float(torch.cat([dist, torch.zeros(1)]).median()) or float(dist.median())
    if use_val:
        load_flat(model, own)
        l_own = val_loss(model, xv, yv)
        losses = []
        for p in peers:
            load_flat(model, p); losses.append(val_loss(model, xv, yv))
        losses = torch.tensor(losses)
        ref = torch.cat([losses, torch.tensor([l_own])])
        med_l = float(ref.median())
        mad_l = float((ref - med_l).abs().median()) + 1e-3
    trust = []
    for j, pid in enumerate(peer_ids):
        ok = True
        if use_val and float(losses[j]) > med_l + lam * mad_l:
            ok = False
        if use_dist and float(dist[j]) > kappa * tau + 1e-12:
            ok = False
        s = 1.0 if ok else 0.0
        if use_rep:
            r = beta * state.rep.get(pid, 0.5) + (1 - beta) * s
            state.rep[pid] = r
            w = r if ok else 0.0
        else:
            w = s
        trust.append(w)
    w = torch.tensor(trust)
    if float(w.sum()) <= 0:
        return own
    acc = torch.zeros_like(own)
    for wj, d, nd in zip(w, diffs, dist):
        if float(wj) <= 0:
            continue
        c = min(1.0, tau / (float(nd) + 1e-12)) if use_clip else 1.0
        acc += float(wj) * c * d
    # mixing weight: own model counts as one fully trusted peer
    return own + acc / (float(w.sum()) + 1.0)


RULES = {"gossip": agg_mean, "median": agg_median, "trimmed": agg_trimmed, "krum": agg_krum,
         "clipped": agg_clipped_gossip, "balance": agg_balance, "shield": agg_shield}


# ------------------------------------------------------------------ adaptive attacks
# The attacks above are generic: none of them knows that the receiver screens models. The
# three below are written against ShieldDFL's own rule. The attacker is assumed to know the
# rule and its constants (lambda, kappa, beta), to observe the honest models of the current
# round (the omniscient setting already used for ALIE and IPM), and to hold its own
# labelled records, which it uses as a surrogate for the receiver's private validation
# split. It cannot read that split, forge an identity or change the graph.
@torch.no_grad()
def adaptive_model(benign_new, x_bad, model, xs_sur, ys_sur, lam=2.0, kappa=2.5,
                   dominate=False, slack=0.9, steps=16, use_val=True):
    """The most damaging model that still passes both of ShieldDFL's gates.

    The attacker interpolates from the coordinate-wise mean of the honest models, the point
    of the honest cloud that is hardest to reject, towards `x_bad`, a model it has trained
    on inverted labels, and takes the largest step for which both gates still admit it:

      distance   the receiver's radius is kappa * med({r_ij} U {0}). While the attackers are
                 a minority of a neighbourhood that median is an honest distance, estimated
                 by the median distance of the honest models to their mean, and a model at
                 that mean is already one such distance from any receiver, so the usable
                 radius is (kappa - 1) * tau. A majority can set the median itself, and the
                 whole kappa * tau is then open to it.
      validation the receiver admits a model whose loss is at most med + lam * MAD over the
                 neighbourhood. The attacker evaluates that threshold on its own split.
                 With `dominate` it counts its colluding copies into the reference set,
                 which is what lets a majority move the median onto itself.
    """
    mu = benign_new.mean(0)
    tau = float((benign_new - mu).norm(dim=1).median()) + 1e-12
    radius = slack * (kappa if dominate else max(0.0, kappa - 1.0)) * tau
    losses = []
    for h in benign_new:
        load_flat(model, h)
        losses.append(val_loss(model, xs_sur, ys_sur))
    losses = torch.tensor(losses)

    def passes(cand):
        load_flat(model, cand)
        l = float(val_loss(model, xs_sur, ys_sur))
        extra = torch.full((len(losses),), l) if dominate else torch.tensor([l])
        ref = torch.cat([losses, extra])
        med = float(ref.median())
        mad = float((ref - med).abs().median()) + 1e-3
        return l <= med + lam * mad

    d = x_bad - mu
    d = d / (float(d.norm()) + 1e-12)
    if not use_val:
        # only the geometric budget binds: spend all of it towards the poisoned model
        return mu + radius * d
    lo, hi, best = 0.0, radius, mu
    for _ in range(steps):                       # bisection on the largest admitted step
        mid = 0.5 * (lo + hi)
        cand = mu + mid * d
        if passes(cand):
            lo, best = mid, cand
        else:
            hi = mid
    return best


ADAPTIVE = ("adaptive", "adaptive_rep", "adaptive_col", "adaptive_dist")


def adaptive_kwargs(kind, t, T):
    """Per-round settings of the three gate-aware attackers.

    adaptive      the constrained attacker above, admitted by both gates in every round.
    adaptive_rep  trust-then-strike: it transmits its honest model while its reputation
                  climbs towards one, then spends that trust on a step far larger than the
                  gates allow, because the smoothed reputation decays only at rate beta per
                  round and still weights it after the instantaneous gates have closed.
    adaptive_col  colluding majority: every attacker sends the same model, so inside a
                  neighbourhood they control they are the median of both statistics.
    """
    if kind == "adaptive":
        return dict(dominate=False, kappa=2.5)
    if kind == "adaptive_dist":
        # geometry only: the crafted model stays inside the distance budget but makes no
        # attempt to keep its loss down, so the validation gate is the only defence left
        return dict(dominate=False, kappa=2.5, use_val=False)
    if kind == "adaptive_col":
        return dict(dominate=True, kappa=2.5)
    if kind == "adaptive_rep":
        if t < 0.5 * T:
            return None                           # send the honest model this round
        return dict(dominate=False, kappa=10.0, slack=1.0)
    raise ValueError(kind)


def tpr_at_fpr(y, p, target=0.01):
    """Recall at the largest threshold whose false-positive rate is at most `target`.
    Reported because accuracy at a fixed 0.5 threshold saturates on these datasets."""
    from sklearn.metrics import roc_curve
    try:
        fpr, tpr, _ = roc_curve(y, p)
    except ValueError:
        return float("nan")
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0
