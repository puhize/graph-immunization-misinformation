"""
Re-run ONLY the corrected community_bridge strategy and save to NEW csv files.
The existing facebook_uniform.csv / facebook_personalized.csv are NOT touched.

Outputs (results/raw/):
    community_bridge_fixed_uniform.csv            (aggregated mean/std per budget)
    community_bridge_fixed_uniform_raw_runs.csv   (per-run rows, for t-tests)
    community_bridge_fixed_personalized.csv
    community_bridge_fixed_personalized_raw_runs.csv

Run on your own machine (no time limit):  python rerun_bridge.py
Uses MONTE_CARLO_RUNS from config.py (500) and the original SIR engine.
"""
import sys, os, time
sys.path.insert(0, "src")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from graph_loader import load_graph, compute_clustering, compute_personalized_beta
from sir_model import run_sir
from strategies import get_immunized_nodes
from config import GAMMA, T_MAX, INITIAL_INFECTED, BUDGET_LEVELS, MONTE_CARLO_RUNS

STRAT = "community_bridge"
RAW_DIR = "results/raw"


def run_mode(G, beta, mode):
    n_nodes = G.number_of_nodes()
    agg_rows, raw_rows = [], []

    for budget in BUDGET_LEVELS:
        k = max(1, int(budget * n_nodes))
        immun = get_immunized_nodes(G, STRAT, budget)   # deterministic, computed once
        totals, peaks, durs = [], [], []
        t0 = time.time()
        for i in range(MONTE_CARLO_RUNS):
            r = run_sir(G, beta, GAMMA, immun, T_MAX, INITIAL_INFECTED)
            totals.append(r["total_infected"])
            peaks.append(r["peak_infected"])
            durs.append(r["duration"])
            raw_rows.append({"strategy": STRAT, "budget": budget, "run_id": i,
                             "total_infected": r["total_infected"],
                             "peak_infected": r["peak_infected"],
                             "duration": r["duration"]})
        agg_rows.append({"strategy": STRAT, "budget": budget, "immunized_count": k,
                         "total_infected_mean": round(np.mean(totals), 2),
                         "total_infected_std": round(np.std(totals), 2),
                         "peak_infected_mean": round(np.mean(peaks), 2),
                         "peak_infected_std": round(np.std(peaks), 2),
                         "duration_mean": round(np.mean(durs), 2),
                         "duration_std": round(np.std(durs), 2)})
        print(f"[{mode}] {STRAT} @ {budget}: "
              f"infected={np.mean(totals):.0f}+/-{np.std(totals):.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    agg_path = os.path.join(RAW_DIR, f"community_bridge_fixed_{mode}.csv")
    raw_path = os.path.join(RAW_DIR, f"community_bridge_fixed_{mode}_raw_runs.csv")
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    print(f"[{mode}] wrote {agg_path} and {raw_path}", flush=True)


def main():
    G = load_graph()
    print(f"Graph loaded: {G.number_of_nodes()} nodes, "
          f"{MONTE_CARLO_RUNS} runs/budget", flush=True)

    run_mode(G, 0.05, "uniform")                       # uniform beta = float

    clustering = compute_clustering(G)
    beta_p = compute_personalized_beta(G, clustering)  # personalized beta = per-node dict
    run_mode(G, beta_p, "personalized")

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
