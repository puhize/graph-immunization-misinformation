# src/simulation.py

import os
import sys
import time
import random
import numpy as np
import pandas as pd

# Ensure project root is on the path (for config.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Ensure src/ is on the path (for sibling modules)
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MONTE_CARLO_RUNS, T_MAX, INITIAL_INFECTED,
    BUDGET_LEVELS, STRATEGIES, GAMMA, BETA_BASE,
)
from graph_loader import load_graph, compute_clustering, compute_personalized_beta
from sir_model import run_sir
from strategies import get_immunized_nodes

# Resolve paths relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "raw")


def run_monte_carlo(G, beta, strategy_name, budget, n_runs=MONTE_CARLO_RUNS):
    """
    Run n_runs SIR simulations for a single (strategy, budget) pair.

    For deterministic strategies (degree, betweenness, community_bridge),
    the immunized set is computed once. For random strategy, it varies per run.

    Parameters:
        G: NetworkX graph
        beta: dict or float, infection rates
        strategy_name: str, one of the strategy names
        budget: float (0–1), fraction of nodes to immunize
        n_runs: int, number of Monte Carlo repetitions

    Returns:
        dict with keys: total_infected_mean, total_infected_std,
                        peak_infected_mean, peak_infected_std,
                        duration_mean, duration_std,
                        raw_results (list of per-run dicts)
    """
    is_random = (strategy_name == "random")

    # For deterministic strategies, compute immunized set once
    if not is_random:
        immunized = get_immunized_nodes(G, strategy_name, budget)

    raw = []
    for i in range(n_runs):
        # Random strategy picks a new set each run
        if is_random:
            immunized = get_immunized_nodes(G, strategy_name, budget)

        result = run_sir(
            G, beta, GAMMA, immunized, T_MAX, INITIAL_INFECTED
        )
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
        "raw_results": raw,
    }


def _load_completed(csv_path):
    """Load already-completed experiments from CSV to skip on resume."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["strategy"], df["budget"]))


def run_all_experiments(G=None, beta=None, beta_mode="personalized"):
    """
    Run Monte Carlo simulations for every (strategy, budget) combination
    defined in config.py. Saves results incrementally to a CSV file.
    Skips already-completed experiments on resume.

    Parameters:
        G: NetworkX graph (loaded from config if None)
        beta: dict or float (computed from graph if None)
        beta_mode: "personalized" or "uniform" — determines beta and output filename

    Returns:
        pandas DataFrame with all results
    """
    if G is None:
        print("Loading graph...")
        G = load_graph()

    if beta is None:
        if beta_mode == "personalized":
            print("Computing personalized beta...")
            clustering = compute_clustering(G)
            beta = compute_personalized_beta(G, clustering)
        else:
            print(f"Using uniform beta = {BETA_BASE}")
            beta = BETA_BASE

    n_nodes = G.number_of_nodes()
    total_combos = len(STRATEGIES) * len(BUDGET_LEVELS)

    # Set up CSV path based on beta mode
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"facebook_{beta_mode}.csv")
    completed = _load_completed(csv_path)

    if completed:
        print(f"Resuming — {len(completed)}/{total_combos} experiments "
              f"already done, skipping them.")

    columns = [
        "strategy", "budget", "immunized_count",
        "total_infected_mean", "total_infected_std",
        "peak_infected_mean", "peak_infected_std",
        "duration_mean", "duration_std",  # always T_MAX at R0=5, not informative
    ]

    # Raw runs CSV for statistical testing (paired t-tests)
    raw_csv_path = csv_path.replace(".csv", "_raw_runs.csv")

    raw_columns = ["strategy", "budget", "run_id",
                   "total_infected", "peak_infected", "duration"]

    # Write headers if starting fresh
    if not completed:
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        pd.DataFrame(columns=raw_columns).to_csv(raw_csv_path, index=False)
    elif not os.path.exists(raw_csv_path):
        # Aggregated CSV exists but raw runs was never created
        pd.DataFrame(columns=raw_columns).to_csv(raw_csv_path, index=False)

    print(f"\nRunning {total_combos} experiments "
          f"({MONTE_CARLO_RUNS} runs each, {n_nodes} nodes)\n")

    combo = 0
    for strategy in STRATEGIES:
        for budget in BUDGET_LEVELS:
            combo += 1
            k = max(1, int(budget * n_nodes))

            # Skip if already completed in a previous run
            if (strategy, budget) in completed:
                print(f"[{combo}/{total_combos}] {strategy} @ budget={budget} "
                      f"— already done, skipping")
                continue

            print(f"[{combo}/{total_combos}] {strategy} @ budget={budget} "
                  f"({k} nodes)...", end=" ", flush=True)

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

            # Append aggregated result to CSV
            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)

            # Append raw per-run results for statistical testing
            raw_rows = [
                {
                    "strategy": strategy,
                    "budget": budget,
                    "run_id": i,
                    "total_infected": r["total_infected"],
                    "peak_infected": r["peak_infected"],
                    "duration": r["duration"],
                }
                for i, r in enumerate(result["raw_results"])
            ]
            pd.DataFrame(raw_rows).to_csv(
                raw_csv_path, mode="a", header=False, index=False
            )

            print(f"done ({elapsed:.1f}s) — "
                  f"infected={result['total_infected_mean']:.0f} "
                  f"± {result['total_infected_std']:.0f}")

    # Load and return the full results
    df = pd.read_csv(csv_path)
    print(f"\nAll results saved to {csv_path}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Facebook Monte Carlo SIR simulations")
    parser.add_argument("--beta", choices=["uniform", "personalized"],
                        default="uniform",
                        help="'uniform' = fixed beta_base for all nodes, "
                             "'personalized' = beta_i formula. Default: uniform")
    args = parser.parse_args()

    print(f"Beta mode: {args.beta}")
    df = run_all_experiments(beta_mode=args.beta)
    print("\n" + df.to_string(index=False))
