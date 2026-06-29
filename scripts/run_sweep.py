"""
Robustness sweeps for Contribution 1: run the full strategy comparison under a
chosen susceptibility model and a chosen R0, writing to TAGGED CSVs so the main
results (R0 = 5) are never overwritten.

Examples
  python run_sweep.py --beta uniform      --r0 2     # low-virality sensitivity
  python run_sweep.py --beta echo         --r0 5     # echo-chamber direction
  python run_sweep.py --beta personalized --r0 2

beta modes
  uniform       constant beta_base for all nodes
  personalized  skeptic direction:  beta_i = beta_base * (1 - alpha*CC_i)
  echo          echo-chamber direction, mean-centred (R0 preserved):
                beta_i = beta_base * (1 + alpha*(CC_i - <CC>))

R0 is set via beta_base = r0 * gamma (gamma fixed). Output:
  results/raw/facebook_<beta>_r<r0>.csv  (+ _raw_runs.csv). Resumable.
Uses a fast push-based SIR, validated equivalent to src/sir_model.run_sir.
"""
import sys, os, time, random, argparse
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from graph_loader import load_graph, compute_clustering, compute_echo_beta
from strategies import get_immunized_nodes
from sir_model import run_sir_fast
from config import GAMMA, ALPHA, T_MAX, INITIAL_INFECTED, BUDGET_LEVELS, STRATEGIES, MONTE_CARLO_RUNS

RAW_DIR = "results/raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", choices=["uniform", "personalized", "echo"], default="uniform")
    ap.add_argument("--r0", type=float, default=5.0)
    ap.add_argument("--runs", type=int, default=MONTE_CARLO_RUNS)
    ap.add_argument("--strategies", default=",".join(STRATEGIES),
                    help="comma-separated subset, e.g. betweenness,community_bridge")
    a = ap.parse_args()

    beta_base = a.r0 * GAMMA
    G = load_graph()
    nodes = list(G.nodes())
    adj = {n: list(G.neighbors(n)) for n in nodes}

    if a.beta == "uniform":
        beta = beta_base
    else:
        cc = compute_clustering(G)
        if a.beta == "personalized":
            beta = {n: beta_base * (1 - ALPHA * cc[n]) for n in nodes}
        else:  # echo
            beta = compute_echo_beta(G, cc, beta_base=beta_base, alpha=ALPHA)

    r0tag = (f"{a.r0:g}").replace(".", "_")
    os.makedirs(RAW_DIR, exist_ok=True)
    agg_path = os.path.join(RAW_DIR, f"facebook_{a.beta}_r{r0tag}.csv")
    raw_path = os.path.join(RAW_DIR, f"facebook_{a.beta}_r{r0tag}_raw_runs.csv")
    cols = ["strategy", "budget", "immunized_count", "total_infected_mean", "total_infected_std",
            "peak_infected_mean", "peak_infected_std", "duration_mean", "duration_std"]
    rawcols = ["strategy", "budget", "run_id", "total_infected", "peak_infected", "duration"]
    done = set()
    if os.path.exists(agg_path):
        done = set(zip(pd.read_csv(agg_path)["strategy"], pd.read_csv(agg_path)["budget"]))
    else:
        pd.DataFrame(columns=cols).to_csv(agg_path, index=False)
        pd.DataFrame(columns=rawcols).to_csv(raw_path, index=False)

    print(f"beta={a.beta}  R0={a.r0}  beta_base={beta_base:.4f}  -> {agg_path}", flush=True)
    strat_list = [s.strip() for s in a.strategies.split(",") if s.strip()]
    for strat in strat_list:
        for b in BUDGET_LEVELS:
            if (strat, b) in done:
                continue
            is_rand = (strat == "random")
            if not is_rand:
                immun = get_immunized_nodes(G, strat, b)
            t0 = time.time(); tot, pk, du, raws = [], [], [], []
            for i in range(a.runs):
                if is_rand:
                    immun = get_immunized_nodes(G, strat, b)
                ever, pi, di = run_sir_fast(adj, nodes, beta, GAMMA, immun, T_MAX, INITIAL_INFECTED)
                ti = len(ever)
                tot.append(ti); pk.append(pi); du.append(di)
                raws.append({"strategy": strat, "budget": b, "run_id": i,
                             "total_infected": ti, "peak_infected": pi, "duration": di})
            pd.DataFrame([{"strategy": strat, "budget": b, "immunized_count": max(1, int(b*len(nodes))),
                           "total_infected_mean": round(np.mean(tot), 2), "total_infected_std": round(np.std(tot), 2),
                           "peak_infected_mean": round(np.mean(pk), 2), "peak_infected_std": round(np.std(pk), 2),
                           "duration_mean": round(np.mean(du), 2), "duration_std": round(np.std(du), 2)}]
                         ).to_csv(agg_path, mode="a", header=False, index=False)
            pd.DataFrame(raws).to_csv(raw_path, mode="a", header=False, index=False)
            print(f"  {strat} @ {b}: inf={np.mean(tot):.0f}  ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
