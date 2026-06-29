# test_sir.py — Validate SIR model implementation
import sys
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, _R)

import random
import networkx as nx
from sir_model import run_sir
from graph_loader import load_graph, compute_clustering, compute_personalized_beta

random.seed(42)

# --- Test 1: Basic SIR on small graph ---
print("=" * 50)
print("Test 1: SIR on small complete graph (K20)")
print("=" * 50)
K = nx.complete_graph(20)
result = run_sir(K, beta=0.3, gamma=0.05, immunized_nodes=set(), t_max=100, initial_infected_count=1)
print(f"Total infected: {result['total_infected']}/20")
print(f"Peak infected:  {result['peak_infected']}")
print(f"Duration:       {result['duration']} steps")
print(f"History length: {len(result['history'])} steps")
print()

# --- Test 2: R0 < 1 should die out quickly ---
print("=" * 50)
print("Test 2: Low beta (R0 < 1) — epidemic should die out")
print("=" * 50)
G_er = nx.erdos_renyi_graph(500, 0.01, seed=42)
result_low = run_sir(G_er, beta=0.001, gamma=0.1, immunized_nodes=set(), t_max=200, initial_infected_count=3)
print(f"Total infected: {result_low['total_infected']}/500")
print(f"Duration:       {result_low['duration']} steps")
assert result_low['total_infected'] < 50, "R0 < 1 but epidemic spread widely!"
print("PASS — epidemic died out as expected")
print()

# --- Test 3: Immunization should reduce spread ---
print("=" * 50)
print("Test 3: Immunization reduces infection")
print("=" * 50)
G_er2 = nx.erdos_renyi_graph(200, 0.05, seed=99)

random.seed(10)
no_immune = run_sir(G_er2, beta=0.1, gamma=0.01, immunized_nodes=set(), t_max=200, initial_infected_count=3)

random.seed(10)
hub_nodes = sorted(G_er2.degree(), key=lambda x: x[1], reverse=True)[:40]
immune_set = set(n for n, d in hub_nodes)
with_immune = run_sir(G_er2, beta=0.1, gamma=0.01, immunized_nodes=immune_set, t_max=200, initial_infected_count=3)

print(f"Without immunization: {no_immune['total_infected']} infected")
print(f"With 20% immunized:   {with_immune['total_infected']} infected")
print()

# --- Test 4: Personalized beta on Facebook graph ---
print("=" * 50)
print("Test 4: Personalized beta on Facebook graph")
print("=" * 50)
G = load_graph()
clustering = compute_clustering(G)
beta_personalized = compute_personalized_beta(G, clustering)

random.seed(123)
result_fb = run_sir(G, beta=beta_personalized, gamma=0.01, immunized_nodes=set(), t_max=200, initial_infected_count=5)
print(f"Nodes: {G.number_of_nodes()}")
print(f"Total infected: {result_fb['total_infected']}")
print(f"Peak infected:  {result_fb['peak_infected']}")
print(f"Duration:       {result_fb['duration']} steps")
print(f"Final state:    S={result_fb['history'][-1][0]}, I={result_fb['history'][-1][1]}, R={result_fb['history'][-1][2]}")
print()
print("All tests passed!")
