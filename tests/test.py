# test.py
import sys
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, _R)
from graph_loader import load_graph, graph_summary, compute_clustering, compute_personalized_beta

G = load_graph()
graph_summary(G)

clustering = compute_clustering(G)
beta = compute_personalized_beta(G, clustering)

sample = sorted(beta.items(), key=lambda x: x[1])
print("\nLowest β nodes (most skeptical):", sample[:3])
print("Highest β nodes (most susceptible):", sample[-3:])