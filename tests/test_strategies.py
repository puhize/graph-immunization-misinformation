# test_strategies.py — Validate all four immunization strategies
import sys
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, _R)

import random
from graph_loader import load_graph
from strategies import get_immunized_nodes, STRATEGY_MAP

random.seed(42)
G = load_graph()
n = G.number_of_nodes()
budget = 0.05  # 5% of nodes

print(f"Graph: {n} nodes, {G.number_of_edges()} edges")
print(f"Budget: {budget} = {int(budget * n)} nodes\n")

for name in STRATEGY_MAP:
    print(f"--- {name} ---")
    nodes = get_immunized_nodes(G, name, budget)
    print(f"  Immunized: {len(nodes)} nodes")

    # Sanity checks
    assert len(nodes) == int(budget * n), f"Expected {int(budget * n)}, got {len(nodes)}"
    assert all(n in G.nodes() for n in nodes), "Immunized node not in graph!"

    # Show sample
    sample = sorted(list(nodes))[:5]
    print(f"  Sample nodes: {sample}")

    # For degree strategy, verify we got high-degree nodes
    if name == "degree":
        degrees = [G.degree(n) for n in nodes]
        print(f"  Min degree in set: {min(degrees)}, Max: {max(degrees)}")

    print()

print("All strategies validated!")
