# src/metrics.py
"""
Fairness metrics for the misinformation-containment experiments (Contribution 2).

The efficiency metrics (total/peak infected) say HOW MUCH spread a strategy
stops. They do not say WHO is left exposed. A strategy can be efficient overall
yet protect a few central communities while leaving peripheral ones fully
infected. This module measures that inequality with the Gini coefficient of
per-community infection rates.

  per-community infection rate = (members ever infected) / (community size)
  Gini = 0  -> every community infected at the same rate (perfectly fair)
  Gini -> 1 -> infection concentrated in a few communities (maximally unfair)

Communities are detected once with Louvain (fixed seed) and cached. The SIR
here is the validated push-based engine (statistically equivalent to
src/sir_model.run_sir) and returns the set of ever-infected nodes so we can
break infections down by community.
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    GAMMA, T_MAX, INITIAL_INFECTED, BUDGET_LEVELS, STRATEGIES,
    MONTE_CARLO_RUNS, BETA_BASE,
)
from graph_loader import load_graph, compute_clustering, compute_personalized_beta
from strategies import get_immunized_nodes
from sir_model import run_sir_fast

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "raw")


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------

def gini(values):
    """
    Gini coefficient of a list of non-negative values.

    Uses the rank-weighted formula
        G = sum_i (2*rank_i - n - 1) * x_i  /  (n * sum_i x_i)
    with values sorted ascending and rank starting at 1.
    Returns 0.0 for an empty input or an all-zero vector.
    """
    x = sorted(float(v) for v in values)
    n = len(x)
    total = sum(x)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((2 * (i + 1) - n - 1) * xi for i, xi in enumerate(x))
    return cum / (n * total)


# ---------------------------------------------------------------------------
# Community detection (cached)
# ---------------------------------------------------------------------------

def _partition_cache_path(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    return os.path.join(CACHE_DIR, f"louvain_partition_n{n}_m{m}.json")


def get_communities(G):
    """
    Return (partition, communities) where partition maps node -> community id
    and communities maps community id -> set of member nodes. Uses the same
    fixed Louvain seed (42) as the community_bridge strategy, and caches the
    partition so fairness and strategy selection use the SAME community split.
    """
    path = _partition_cache_path(G)
    if os.path.exists(path):
        with open(path) as f:
            partition = {int(k): v for k, v in json.load(f).items()}
    else:
        # random_state pins the partition without mutating the global RNG
        # (which would perturb the Monte-Carlo seed stream on a cache miss).
        # Same mechanism as strategies._load_or_compute_partition.
        partition = community_louvain.best_partition(G, random_state=42)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in partition.items()}, f)

    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, set()).add(node)
    return partition, communities


# ---------------------------------------------------------------------------
# SIR that returns the set of ever-infected nodes
# ---------------------------------------------------------------------------

def run_sir_states(adj, nodes, beta, gamma, immunized, t_max, init_count):
    """Set of ever-infected nodes (excludes immunized). Thin wrapper around
    sir_model.run_sir_fast, kept for the fairness pipeline."""
    ever, _peak, _dur = run_sir_fast(adj, nodes, beta, gamma, immunized, t_max, init_count)
    return ever


# ---------------------------------------------------------------------------
# Per-run fairness
# ---------------------------------------------------------------------------

def community_infection_rates(communities, ever_infected, immunized):
    """
    Infection rate per community = (ever-infected members) / (community size).
    Immunized members are excluded from the infected count (they were never
    infected) but kept in the denominator so a community that was protected by
    spending budget on it still counts as 'covered'.
    """
    rates = []
    for members in communities.values():
        size = len(members)
        if size == 0:
            continue
        infected_here = len(members & ever_infected)
        rates.append(infected_here / size)
    return rates


def run_fairness(G, beta, strategy, budget, communities, n_runs=MONTE_CARLO_RUNS,
                 adj=None, nodes=None):
    """
    Run n_runs simulations for one (strategy, budget) pair and return the mean
    and std of the per-run Gini of community infection rates, plus the mean
    overall community infection rate.
    """
    if nodes is None:
        nodes = list(G.nodes())
    if adj is None:
        adj = {n: list(G.neighbors(n)) for n in nodes}

    is_random = (strategy == "random")
    if not is_random:
        immun = get_immunized_nodes(G, strategy, budget)

    ginis, mean_rates = [], []
    for _ in range(n_runs):
        if is_random:
            immun = get_immunized_nodes(G, strategy, budget)
        ever = run_sir_states(adj, nodes, beta, GAMMA, immun, T_MAX, INITIAL_INFECTED)
        rates = community_infection_rates(communities, ever, immun)
        ginis.append(gini(rates))
        mean_rates.append(float(np.mean(rates)) if rates else 0.0)

    return {
        "gini_mean": float(np.mean(ginis)),
        "gini_std": float(np.std(ginis)),
        "infection_rate_mean": float(np.mean(mean_rates)),
    }


def run_fairness_experiments(beta_mode="uniform", strategies=None, n_runs=MONTE_CARLO_RUNS):
    """
    Compute fairness for every (strategy, budget) and save to
    results/raw/fairness_<beta_mode>.csv. Resumable: skips rows already present.
    """
    strategies = strategies or STRATEGIES
    G = load_graph()
    nodes = list(G.nodes())
    adj = {n: list(G.neighbors(n)) for n in nodes}
    partition, communities = get_communities(G)
    print(f"{len(communities)} communities; {G.number_of_nodes()} nodes", flush=True)

    if beta_mode == "personalized":
        beta = compute_personalized_beta(G, compute_clustering(G))
    else:
        beta = BETA_BASE

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"fairness_{beta_mode}.csv")
    cols = ["strategy", "budget", "gini_mean", "gini_std", "infection_rate_mean"]
    done = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        done = set(zip(prev["strategy"], prev["budget"]))
    else:
        pd.DataFrame(columns=cols).to_csv(csv_path, index=False)

    for strategy in strategies:
        for budget in BUDGET_LEVELS:
            if (strategy, budget) in done:
                continue
            t0 = time.time()
            r = run_fairness(G, beta, strategy, budget, communities, n_runs, adj, nodes)
            pd.DataFrame([{"strategy": strategy, "budget": budget,
                           "gini_mean": round(r["gini_mean"], 4),
                           "gini_std": round(r["gini_std"], 4),
                           "infection_rate_mean": round(r["infection_rate_mean"], 4)}]
                         ).to_csv(csv_path, mode="a", header=False, index=False)
            print(f"{strategy} @ {budget}: Gini={r['gini_mean']:.3f}"
                  f"+/-{r['gini_std']:.3f}  inf_rate={r['infection_rate_mean']:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    print(f"Saved {csv_path}", flush=True)
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute community fairness (Gini) per strategy/budget")
    ap.add_argument("--beta", choices=["uniform", "personalized"], default="uniform")
    ap.add_argument("--runs", type=int, default=MONTE_CARLO_RUNS)
    args = ap.parse_args()
    df = run_fairness_experiments(beta_mode=args.beta, n_runs=args.runs)
    print("\n" + df.to_string(index=False))
