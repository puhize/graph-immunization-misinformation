# src/precompute_twitter_betweenness.py
# Pre-compute and cache approximate betweenness centrality for the Twitter graph.
# This avoids computing it during the simulation pipeline.

import os
import sys
import json
import time
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TWITTER_PATH = os.path.join(PROJECT_ROOT, "data", "twitter_combined.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")

SAMPLE_K = 500

if __name__ == "__main__":
    # Load graph
    print("Loading Twitter graph...")
    G = nx.read_edgelist(TWITTER_PATH, nodetype=int, create_using=nx.Graph())
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # Cache path (must match what strategies.py expects)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    cache_path = os.path.join(CACHE_DIR,
                              f"betweenness_approx_k{SAMPLE_K}_n{n}_m{m}.json")

    if os.path.exists(cache_path):
        print(f"\nCache already exists: {cache_path}")
        print("Delete it first if you want to recompute.")
    else:
        print(f"\nComputing approximate betweenness centrality "
              f"(k={SAMPLE_K} samples)...")
        t0 = time.time()
        bc = nx.betweenness_centrality(G, k=SAMPLE_K, seed=42)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s")

        # Save to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in bc.items()}, f)
        print(f"Cached to {cache_path}")

        # Quick sanity check
        top5 = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 nodes by approx betweenness:")
        for node, score in top5:
            print(f"  Node {node}: {score:.6f}")
