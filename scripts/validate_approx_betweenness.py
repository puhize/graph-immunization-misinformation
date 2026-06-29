# validate_approx_betweenness.py
# Compare exact vs approximate (k=500) betweenness on the Facebook graph.
# Measures node set overlap at each budget level to validate the approximation.

import sys
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, _R)

import json
import os
import networkx as nx
from config import BUDGET_LEVELS
from graph_loader import load_graph
from strategies import (
    _load_or_compute_betweenness,
    _load_or_compute_approx_betweenness,
    _top_k_nodes,
    _budget_to_count,
)

# Load graph
G = load_graph()
n = G.number_of_nodes()
print(f"Facebook graph: {n} nodes\n")

# Load exact betweenness (from cache)
print("Loading exact betweenness (cached)...")
bc_exact = _load_or_compute_betweenness(G)

# Compute approximate betweenness (k=500) — will cache for future use
print("Loading/computing approximate betweenness (k=500)...")
bc_approx = _load_or_compute_approx_betweenness(G, sample_k=500)

# Compare at each budget level
print(f"\n{'Budget':>8} {'k nodes':>8} {'Overlap':>8} {'Overlap %':>10}")
print("-" * 40)

for budget in BUDGET_LEVELS:
    k = _budget_to_count(G, budget)
    exact_set = _top_k_nodes(bc_exact.items(), k)
    approx_set = _top_k_nodes(bc_approx.items(), k)
    overlap = len(exact_set & approx_set)
    pct = 100 * overlap / k

    print(f"{budget:>8.2f} {k:>8} {overlap:>8} {pct:>9.1f}%")

# Rank correlation (Spearman) across all nodes
from scipy.stats import spearmanr

nodes = sorted(bc_exact.keys())
exact_vals = [bc_exact[n] for n in nodes]
approx_vals = [bc_approx[n] for n in nodes]
rho, pval = spearmanr(exact_vals, approx_vals)

print(f"\nSpearman rank correlation: rho={rho:.4f}, p={pval:.2e}")
print("(rho > 0.95 means the ranking is highly reliable)")
