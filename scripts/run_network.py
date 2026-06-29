"""
Four-strategy comparison on a large network, using the vectorized sparse SIR
engine (sir_model.run_sir_vectorized). Built for the Twitter ego network
(~81k nodes), where exact betweenness and the pure-Python engines are
infeasible — betweenness uses the validated k=500 approximation.

  python run_network.py --tag twitter --runs 100
  python run_network.py --tag twitter --runs 100 --strategies community_bridge   # resume one

Output: results/raw/<tag>_uniform.csv (+ _raw_runs.csv). Resumable (skips
completed strategy/budget cells). First community_bridge run computes and caches
the Louvain partition (one-time, a few minutes at 81k nodes).
"""
import sys, os, time, argparse
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import networkx as nx
from strategies import get_immunized_nodes
from sir_model import build_adjacency, run_sir_vectorized
from config import BETA_BASE, GAMMA, T_MAX, INITIAL_INFECTED, BUDGET_LEVELS
from strategies import _load_or_compute_partition


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float)); n = len(x); tot = x.sum()
    if n == 0 or tot == 0:
        return 0.0
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * tot))

# betweenness -> approximate (k=500) for large graphs
STRATEGIES = ["random", "degree", "betweenness_approx", "community_bridge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/twitter_combined.txt")
    ap.add_argument("--tag", default="twitter")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--fairness", action="store_true", help="also compute Gini of per-community infection rates")
    a = ap.parse_args()
    strat_list = [s.strip() for s in a.strategies.split(",") if s.strip()]

    print(f"Loading {a.graph} ...", flush=True)
    G = nx.read_edgelist(a.graph, nodetype=int)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    A, nodes, index_of = build_adjacency(G)
    print(f"  CSR built (nnz={A.nnz})", flush=True)

    comm_idx = None; fair_path = None; fdone = set()
    if a.fairness:
        partition = _load_or_compute_partition(G)
        comm_idx = np.fromiter((partition[x] for x in nodes), dtype=np.int64)
        sizes = np.bincount(comm_idx)
        valid = sizes > 0
        fair_path = f"results/raw/{a.tag}_fairness.csv"
        fcols = ["strategy", "budget", "gini_mean", "gini_std", "infection_rate_mean", "n_runs"]
        if os.path.exists(fair_path):
            fdone = set(zip(pd.read_csv(fair_path).strategy, pd.read_csv(fair_path).budget))
        else:
            pd.DataFrame(columns=fcols).to_csv(fair_path, index=False)
        print(f"  fairness on: {int(valid.sum())} communities", flush=True)

    os.makedirs("results/raw", exist_ok=True)
    agg = f"results/raw/{a.tag}_uniform.csv"
    raw = f"results/raw/{a.tag}_uniform_raw_runs.csv"
    cols = ["strategy", "budget", "immunized_count", "total_infected_mean", "total_infected_std",
            "peak_infected_mean", "peak_infected_std", "duration_mean", "duration_std"]
    rawcols = ["strategy", "budget", "run_id", "total_infected", "peak_infected", "duration"]
    done = set()
    if os.path.exists(agg):
        d = pd.read_csv(agg); done = set(zip(d.strategy, d.budget))
    else:
        pd.DataFrame(columns=cols).to_csv(agg, index=False)
        pd.DataFrame(columns=rawcols).to_csv(raw, index=False)

    rng = np.random.default_rng(42)
    N = len(nodes)
    for strat in strat_list:
        for b in BUDGET_LEVELS:
            if (strat, b) in done:
                print(f"skip {strat} @ {b} (done)", flush=True); continue
            is_rand = (strat == "random")
            if not is_rand:
                immS = get_immunized_nodes(G, strat, b)
                imm_idx = np.fromiter((index_of[x] for x in immS), dtype=np.int64)
            t0 = time.time(); tot, pk, du, rows = [], [], [], []
            ginis, irates = [], []
            for i in range(a.runs):
                if is_rand:
                    immS = get_immunized_nodes(G, strat, b)
                    imm_idx = np.fromiter((index_of[x] for x in immS), dtype=np.int64)
                if a.fairness:
                    ti, pi, di, ever = run_sir_vectorized(A, BETA_BASE, GAMMA, imm_idx, T_MAX, INITIAL_INFECTED, rng, return_ever=True)
                    inf_c = np.bincount(comm_idx, weights=ever.astype(float), minlength=sizes.size)
                    rates = inf_c[valid] / sizes[valid]
                    ginis.append(_gini(rates)); irates.append(float(rates.mean()))
                else:
                    ti, pi, di = run_sir_vectorized(A, BETA_BASE, GAMMA, imm_idx, T_MAX, INITIAL_INFECTED, rng)
                tot.append(ti); pk.append(pi); du.append(di)
                rows.append({"strategy": strat, "budget": b, "run_id": i,
                             "total_infected": ti, "peak_infected": pi, "duration": di})
            pd.DataFrame([{"strategy": strat, "budget": b, "immunized_count": max(1, int(b * N)),
                           "total_infected_mean": round(float(np.mean(tot)), 2), "total_infected_std": round(float(np.std(tot)), 2),
                           "peak_infected_mean": round(float(np.mean(pk)), 2), "peak_infected_std": round(float(np.std(pk)), 2),
                           "duration_mean": round(float(np.mean(du)), 2), "duration_std": round(float(np.std(du)), 2)}]
                         ).to_csv(agg, mode="a", header=False, index=False)
            pd.DataFrame(rows).to_csv(raw, mode="a", header=False, index=False)
            if a.fairness and (strat, b) not in fdone:
                pd.DataFrame([{"strategy": strat, "budget": b,
                               "gini_mean": round(float(np.mean(ginis)), 4),
                               "gini_std": round(float(np.std(ginis)), 4),
                               "infection_rate_mean": round(float(np.mean(irates)), 4),
                               "n_runs": len(ginis)}]).to_csv(fair_path, mode="a", header=False, index=False)
            print(f"{strat} @ {b}: infected={np.mean(tot):.0f}"
                  + (f"  gini={np.mean(ginis):.3f}" if a.fairness else "")
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
