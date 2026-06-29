# src/run_twitter.py
# Simulation pipeline for the Twitter ego network dataset.
# Uses approximate betweenness (k=500) instead of exact computation.

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import networkx as nx

# Ensure project root and src/ are on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MONTE_CARLO_RUNS, T_MAX, INITIAL_INFECTED,
    BUDGET_LEVELS, GAMMA, BETA_BASE, ALPHA,
)
from graph_loader import compute_clustering, compute_personalized_beta
from sir_model import run_sir
from strategies import get_immunized_nodes

# ---------------------------------------------------------------------------
# Twitter-specific config
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TWITTER_PATH = os.path.join(PROJECT_ROOT, "data", "twitter_combined.txt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "raw")
CSV_PATH = os.path.join(RESULTS_DIR, "results_twitter.csv")

# Use betweenness_approx instead of exact betweenness for this large graph
TWITTER_STRATEGIES = ["random", "degree", "betweenness_approx", "community_bridge"]


# ---------------------------------------------------------------------------
# Load Twitter graph (directed edges → undirected)
# ---------------------------------------------------------------------------

def load_twitter_graph():
    """Load the Twitter ego network as an undirected graph."""
    print(f"Loading Twitter graph from {TWITTER_PATH}...")
    G = nx.read_edgelist(TWITTER_PATH, nodetype=int, create_using=nx.Graph())
    return G


def twitter_graph_summary(G):
    """Print basic graph properties."""
    degrees = [d for _, d in G.degree()]
    print(f"  Nodes:    {G.number_of_nodes():,}")
    print(f"  Edges:    {G.number_of_edges():,}")
    print(f"  Avg deg:  {np.mean(degrees):.2f}")
    print(f"  Max deg:  {max(degrees):,}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    print()


# ---------------------------------------------------------------------------
# Monte Carlo runner (same logic as simulation.py, self-contained)
# ---------------------------------------------------------------------------

def load_completed(csv_path):
    """Load already-completed experiments to skip on resume."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["strategy"], df["budget"]))


def run_monte_carlo(G, beta, strategy_name, budget, n_runs=MONTE_CARLO_RUNS):
    """Run n_runs SIR simulations for one (strategy, budget) pair."""
    is_random = (strategy_name == "random")

    if not is_random:
        immunized = get_immunized_nodes(G, strategy_name, budget)

    raw = []
    for i in range(n_runs):
        if is_random:
            immunized = get_immunized_nodes(G, strategy_name, budget)

        result = run_sir(G, beta, GAMMA, immunized, T_MAX, INITIAL_INFECTED)
        raw.append({
            "total_infected": result["total_infected"],
            "peak_infected": result["peak_infected"],
            "duration": result["duration"],
        })

    totals = [r["total_infected"] for r in raw]
    peaks = [r["peak_infected"] for r in raw]
    durations = [r["duration"] for r in raw]

    return {
        "total_infected_mean": np.mean(totals),
        "total_infected_std": np.std(totals),
        "peak_infected_mean": np.mean(peaks),
        "peak_infected_std": np.std(peaks),
        "duration_mean": np.mean(durations),
        "duration_std": np.std(durations),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load and summarize graph
    G = load_twitter_graph()
    print("Twitter ego network summary:")
    twitter_graph_summary(G)

    # Compute personalized beta
    print("Computing clustering coefficients...")
    clustering = compute_clustering(G)
    print("Computing personalized beta...")
    beta = compute_personalized_beta(G, clustering)

    # Set up incremental CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    completed = load_completed(CSV_PATH)
    total_combos = len(TWITTER_STRATEGIES) * len(BUDGET_LEVELS)

    columns = [
        "strategy", "budget", "immunized_count",
        "total_infected_mean", "total_infected_std",
        "peak_infected_mean", "peak_infected_std",
        "duration_mean", "duration_std",
    ]

    if completed:
        print(f"Resuming — {len(completed)}/{total_combos} already done.\n")
    else:
        pd.DataFrame(columns=columns).to_csv(CSV_PATH, index=False)

    n_nodes = G.number_of_nodes()
    print(f"Running {total_combos} experiments "
          f"({MONTE_CARLO_RUNS} runs each, {n_nodes:,} nodes)\n")

    combo = 0
    for strategy in TWITTER_STRATEGIES:
        for budget in BUDGET_LEVELS:
            combo += 1
            k = max(1, int(budget * n_nodes))

            if (strategy, budget) in completed:
                print(f"[{combo}/{total_combos}] {strategy} @ {budget} — skipping")
                continue

            print(f"[{combo}/{total_combos}] {strategy} @ budget={budget} "
                  f"({k:,} nodes)...", end=" ", flush=True)

            t0 = time.time()
            result = run_monte_carlo(G, beta, strategy, budget)
            elapsed = time.time() - t0

            row = {
                "strategy": strategy,
                "budget": budget,
                "immunized_count": k,
                "total_infected_mean": round(result["total_infected_mean"], 2),
                "total_infected_std": round(result["total_infected_std"], 2),
                "peak_infected_mean": round(result["peak_infected_mean"], 2),
                "peak_infected_std": round(result["peak_infected_std"], 2),
                "duration_mean": round(result["duration_mean"], 2),
                "duration_std": round(result["duration_std"], 2),
            }

            pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)

            print(f"done ({elapsed:.1f}s) — "
                  f"infected={result['total_infected_mean']:,.0f} "
                  f"± {result['total_infected_std']:,.0f}")

    print(f"\nAll results saved to {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print("\n" + df.to_string(index=False))
