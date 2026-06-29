# src/graph_loader.py

import os
import networkx as nx
import numpy as np
from config import GRAPH_PATH, ALPHA, BETA_BASE

# Resolve paths relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

def load_graph():
    path = os.path.join(PROJECT_ROOT, GRAPH_PATH)
    G = nx.read_edgelist(path, nodetype=int)
    return G

def compute_clustering(G):
    return nx.clustering(G)

def compute_personalized_beta(G, clustering):
    """
    β_i = β_base × (1 − α × CC_i)
    """
    beta = {}
    for node in G.nodes():
        cc = clustering[node]
        beta[node] = BETA_BASE * (1 - ALPHA * cc)
    return beta

def graph_summary(G):
    degrees = [d for n, d in G.degree()]
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Is connected: {nx.is_connected(G)}")
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Avg clustering coefficient: {nx.average_clustering(G):.4f}")

def compute_echo_beta(G, clustering, beta_base=BETA_BASE, alpha=ALPHA):
    """
    Echo-chamber susceptibility (Contribution-1 robustness variant).

    Opposite direction to compute_personalized_beta: nodes embedded in tight
    communities (high local clustering) are MORE susceptible, consistent with
    the echo-chamber / network-segregation literature (Del Vicario et al.;
    Stein et al.). The formula is MEAN-CENTRED so the network-average beta
    stays beta_base — i.e. the overall R0 is preserved, and any change in
    outcomes reflects the DIRECTION of the susceptibility-structure
    relationship rather than a change in virality:

        beta_i = beta_base * (1 + alpha * (CC_i - mean_CC)),  clipped to [0, 1]
    """
    mean_cc = float(np.mean(list(clustering.values())))
    beta = {}
    for node in G.nodes():
        b = beta_base * (1.0 + alpha * (clustering[node] - mean_cc))
        beta[node] = min(1.0, max(0.0, b))
    return beta
