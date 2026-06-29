"""
Leiden robustness check for the community-bridge strategy.

Re-detects communities on the Facebook graph with the Leiden algorithm and
compares the resulting inter-community-degree (bridge) node selection against
the Louvain-based selection used throughout the thesis. Because every SIR
outcome depends only on *which* nodes are immunized, a high overlap of the
top-k bridge set means the headline results are robust to the community-
detection algorithm.

Reproduces the figures cited in the thesis (≈88-93% top-k overlap, Spearman
rho ≈ 0.88, modularity ~0.835 vs ~0.836).

Run from the repository root:
    python scripts/leiden_robustness.py

Requires: python-igraph, leidenalg  (pip install igraph leidenalg)
"""
import os
import sys
import json
import csv

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import networkx as nx
from scipy.stats import spearmanr
import community as community_louvain

from graph_loader import load_graph
from config import BUDGET_LEVELS

try:
    import igraph as ig
    import leidenalg
except ImportError:
    sys.exit("This script needs python-igraph and leidenalg:\n"
             "    pip install igraph leidenalg")

CACHE = "results/cache/louvain_partition_n4039_m88234.json"
OUT = "results/leiden_overlap.csv"


def inter_community_degree(G, part):
    """Bridge score = number of a node's neighbours in a different community."""
    return {v: sum(1 for u in G.neighbors(v) if part[u] != part[v]) for v in G.nodes()}


def top_k(scores, k):
    return set(sorted(scores, key=lambda v: (scores[v], v), reverse=True)[:k])


def main():
    G = load_graph()
    N = G.number_of_nodes()
    nodes = list(G.nodes())

    # Louvain partition: reuse the cached one if present (identical to the thesis),
    # otherwise recompute with the same fixed seed.
    if os.path.exists(CACHE):
        louvain = {int(k): v for k, v in json.load(open(CACHE)).items()}
    else:
        louvain = community_louvain.best_partition(G, random_state=42)

    # Leiden partition on the same graph (modularity objective, fixed seed).
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=N, edges=[(idx[u], idx[v]) for u, v in G.edges()])
    membership = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, seed=42).membership
    leiden = {nodes[i]: c for i, c in enumerate(membership)}

    s_louv = inter_community_degree(G, louvain)
    s_leid = inter_community_degree(G, leiden)

    print("Communities  -> Louvain: %d | Leiden: %d"
          % (len(set(louvain.values())), len(set(leiden.values()))))
    print("Modularity   -> Louvain: %.4f | Leiden: %.4f"
          % (community_louvain.modularity(louvain, G),
             community_louvain.modularity(leiden, G)))
    rho, _ = spearmanr([s_louv[v] for v in nodes], [s_leid[v] for v in nodes])
    print("Bridge-score Spearman (Louvain vs Leiden): %.3f\n" % rho)

    rows = []
    print("budget   k     top-k bridge-set overlap")
    for b in BUDGET_LEVELS:
        k = max(1, int(b * N))
        overlap = len(top_k(s_louv, k) & top_k(s_leid, k)) / k * 100
        print("%5.0f%%  %4d   %5.1f%%" % (b * 100, k, overlap))
        rows.append({"budget": b, "k": k, "overlap_pct": round(overlap, 1)})

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["budget", "k", "overlap_pct"])
        w.writeheader()
        w.writerows(rows)
    print("\nSaved %s" % OUT)


if __name__ == "__main__":
    main()
