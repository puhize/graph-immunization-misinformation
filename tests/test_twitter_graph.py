# test_twitter_graph.py — Verify Twitter graph properties and clustering coefficient
import sys
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, _R)

import networkx as nx
import numpy as np

# Load Twitter graph as undirected
print("Loading Twitter graph...")
G = nx.read_edgelist("data/twitter_combined.txt", nodetype=int, create_using=nx.Graph())

# Basic properties
degrees = [d for _, d in G.degree()]
print(f"\nTwitter ego network summary:")
print(f"  Nodes:                  {G.number_of_nodes():,}")
print(f"  Edges:                  {G.number_of_edges():,}")
print(f"  Average degree:         {np.mean(degrees):.2f}")
print(f"  Max degree:             {max(degrees):,}")
print(f"  Connected components:   {nx.number_connected_components(G)}")

# Clustering coefficient — key metric for topology hypothesis
print("\nComputing average clustering coefficient (may take a few minutes)...")
avg_cc = nx.average_clustering(G)
print(f"  Avg clustering (Twitter):   {avg_cc:.4f}")
print(f"  Avg clustering (Facebook):  0.6055")
print(f"  Twitter < Facebook:         {avg_cc < 0.6055}")

if avg_cc < 0.6055:
    print("\n  Topology hypothesis supported: Twitter is less clustered,")
    print("  so personalized beta should have less effect on Twitter.")
else:
    print("\n  WARNING: Twitter clustering >= Facebook.")
    print("  The topology hypothesis may not hold.")
