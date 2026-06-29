"""
analyze_vitality.py
-------------------
Analyzes the modularity vitality cache to understand which nodes were
selected by the community bridge strategy and why it underperforms.

Produces:
  1. Top-N vitality nodes with their degree, betweenness, and clustering
  2. Overlap analysis: which vitality nodes also appear in betweenness top-N
  3. Vitality score distribution plot
  4. Scatter plots: vitality vs degree, vitality vs clustering
  5. Community size distribution

Usage:
    python analyze_vitality.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys

import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, _R)
from graph_loader import load_graph, compute_clustering

VITALITY_CACHE    = "results/cache/vitality_n4039_m88234.json"
BETWEENNESS_CACHE = "results/cache/betweenness_n4039_m88234.json"
FIGURES_DIR       = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

BUDGETS = [0.01, 0.05, 0.10, 0.15, 0.20]


# ─── Load caches ──────────────────────────────────────────────────────────────
def load_caches():
    print("Loading graph...")
    G = load_graph()

    print("Loading vitality cache...")
    with open(VITALITY_CACHE) as f:
        vitality = {int(k): v for k, v in json.load(f).items()}

    print("Loading betweenness cache...")
    with open(BETWEENNESS_CACHE) as f:
        betweenness = {int(k): v for k, v in json.load(f).items()}

    print("Computing clustering coefficients...")
    clustering = compute_clustering(G)

    return G, vitality, betweenness, clustering


# ─── Build node dataframe ─────────────────────────────────────────────────────
def build_node_df(G, vitality, betweenness, clustering):
    degree = dict(G.degree())
    rows = []
    for node in G.nodes():
        rows.append({
            'node': node,
            'vitality': vitality.get(node, 0.0),
            'betweenness': betweenness.get(node, 0.0),
            'degree': degree[node],
            'clustering': clustering[node],
        })
    df = pd.DataFrame(rows)
    df['vitality_rank']    = df['vitality'].rank(ascending=False).astype(int)
    df['betweenness_rank'] = df['betweenness'].rank(ascending=False).astype(int)
    df['degree_rank']      = df['degree'].rank(ascending=False).astype(int)
    return df


# ─── Top-N analysis ───────────────────────────────────────────────────────────
def print_top_nodes(df, n=30):
    print("\n" + "="*85)
    print(f"TOP {n} NODES BY MODULARITY VITALITY")
    print("="*85)
    print(f"{'Node':>8} {'Vitality':>12} {'Vit rank':>9} {'BW rank':>9} "
          f"{'Deg rank':>9} {'Degree':>7} {'Clustering':>12}")
    print("-"*85)
    for _, row in df.nsmallest(n, 'vitality_rank').iterrows():
        print(f"{int(row['node']):>8} {row['vitality']:>12.6f} {int(row['vitality_rank']):>9} "
              f"{int(row['betweenness_rank']):>9} {int(row['degree_rank']):>9} "
              f"{int(row['degree']):>7} {row['clustering']:>12.4f}")


# ─── Overlap analysis ─────────────────────────────────────────────────────────
def print_overlap_analysis(df, G):
    print("\n" + "="*65)
    print("OVERLAP: TOP-K VITALITY vs TOP-K BETWEENNESS")
    print("="*65)
    n_nodes = G.number_of_nodes()
    for b in BUDGETS:
        k = max(1, int(b * n_nodes))
        top_vit = set(df.nsmallest(k, 'vitality_rank')['node'])
        top_bw  = set(df.nsmallest(k, 'betweenness_rank')['node'])
        overlap = len(top_vit & top_bw)
        pct = overlap / k * 100
        print(f"  Budget {b:.0%} (k={k:>4}): overlap = {overlap:>4} / {k} ({pct:.1f}%)")

    print("\n" + "="*65)
    print("OVERLAP: TOP-K VITALITY vs TOP-K DEGREE")
    print("="*65)
    for b in BUDGETS:
        k = max(1, int(b * n_nodes))
        top_vit = set(df.nsmallest(k, 'vitality_rank')['node'])
        top_deg = set(df.nsmallest(k, 'degree_rank')['node'])
        overlap = len(top_vit & top_deg)
        pct = overlap / k * 100
        print(f"  Budget {b:.0%} (k={k:>4}): overlap = {overlap:>4} / {k} ({pct:.1f}%)")


# ─── Vitality distribution stats ──────────────────────────────────────────────
def print_vitality_stats(df):
    print("\n" + "="*55)
    print("VITALITY SCORE DISTRIBUTION")
    print("="*55)
    v = df['vitality']
    print(f"  Total nodes: {len(v)}")
    print(f"  Positive vitality: {(v > 0).sum()} ({(v > 0).mean()*100:.1f}%)")
    print(f"  Zero vitality:     {(v == 0).sum()} ({(v == 0).mean()*100:.1f}%)")
    print(f"  Negative vitality: {(v < 0).sum()} ({(v < 0).mean()*100:.1f}%)")
    print(f"\n  Min:    {v.min():.6f}")
    print(f"  Median: {v.median():.6f}")
    print(f"  Mean:   {v.mean():.6f}")
    print(f"  Max:    {v.max():.6f}")
    print(f"\n  Interpretation: negative vitality means removing the node")
    print(f"  INCREASES modularity — these nodes disrupt community structure.")
    print(f"  The strategy selects the highest positive vitality nodes.")


# ─── Correlation analysis ─────────────────────────────────────────────────────
def print_correlations(df):
    print("\n" + "="*55)
    print("CORRELATION: VITALITY vs OTHER METRICS")
    print("="*55)
    for col, label in [('degree', 'Degree'), ('betweenness', 'Betweenness'), ('clustering', 'Clustering')]:
        r, p = __import__('scipy').stats.pearsonr(df['vitality'], df[col])
        r_s, p_s = __import__('scipy').stats.spearmanr(df['vitality'], df[col])
        print(f"  vs {label:<14}  Pearson r={r:>7.4f} (p={p:.2e})  "
              f"Spearman r={r_s:>7.4f} (p={p_s:.2e})")


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_vitality_analysis(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Modularity Vitality Analysis — Facebook Ego Network",
                 fontsize=13, fontweight='bold')

    # 1. Vitality score distribution
    ax = axes[0, 0]
    v_pos = df[df['vitality'] > 0]['vitality']
    ax.hist(v_pos, bins=50, color='#4CAF50', alpha=0.8, edgecolor='white')
    ax.set_xlabel("Modularity vitality (positive values only)", fontsize=10)
    ax.set_ylabel("Number of nodes", fontsize=10)
    ax.set_title(f"Vitality Score Distribution\n"
                 f"({len(v_pos)} nodes with positive vitality, "
                 f"{len(df) - len(v_pos)} with zero/negative)",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Vitality vs Degree
    ax = axes[0, 1]
    ax.scatter(df['degree'], df['vitality'], alpha=0.2, s=8, color='#2196F3')
    # Highlight top-40 vitality nodes
    top40 = df.nsmallest(40, 'vitality_rank')
    ax.scatter(top40['degree'], top40['vitality'], alpha=0.9, s=30,
               color='#E91E63', label='Top 40 by vitality', zorder=5)
    ax.set_xlabel("Node degree", fontsize=10)
    ax.set_ylabel("Modularity vitality", fontsize=10)
    ax.set_title("Vitality vs Degree\n(pink = top 40 vitality nodes)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Vitality vs Clustering coefficient
    ax = axes[1, 0]
    ax.scatter(df['clustering'], df['vitality'], alpha=0.2, s=8, color='#FF9800')
    ax.scatter(top40['clustering'], top40['vitality'], alpha=0.9, s=30,
               color='#E91E63', label='Top 40 by vitality', zorder=5)
    ax.set_xlabel("Clustering coefficient", fontsize=10)
    ax.set_ylabel("Modularity vitality", fontsize=10)
    ax.set_title("Vitality vs Clustering Coefficient\n(pink = top 40 vitality nodes)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Rank comparison: vitality rank vs betweenness rank (top 200)
    ax = axes[1, 1]
    top200_vit = df.nsmallest(200, 'vitality_rank')
    top200_bw  = df.nsmallest(200, 'betweenness_rank')
    shared = set(top200_vit['node']) & set(top200_bw['node'])
    shared_df = df[df['node'].isin(shared)]
    rest_vit   = df[df['node'].isin(set(top200_vit['node']) - shared)]

    ax.scatter(rest_vit['betweenness_rank'], rest_vit['vitality_rank'],
               alpha=0.4, s=10, color='#4CAF50', label='Top-200 vitality only')
    ax.scatter(shared_df['betweenness_rank'], shared_df['vitality_rank'],
               alpha=0.9, s=25, color='#E91E63', zorder=5,
               label=f'In both top-200 ({len(shared)} nodes)')
    ax.set_xlabel("Betweenness rank (lower = higher betweenness)", fontsize=10)
    ax.set_ylabel("Vitality rank (lower = higher vitality)", fontsize=10)
    ax.set_title("Top-200 Vitality vs Betweenness Rank\n"
                 "(pink = nodes selected by both strategies at 5% budget)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "vitality_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G, vitality, betweenness, clustering = load_caches()
    df = build_node_df(G, vitality, betweenness, clustering)

    print_vitality_stats(df)
    print_top_nodes(df, n=30)
    print_overlap_analysis(df, G)
    print_correlations(df)

    print("\nGenerating plots...")
    plot_vitality_analysis(df)

    # Save node table
    out_path = "results/node_analysis.csv"
    df.sort_values('vitality_rank').to_csv(out_path, index=False)
    print(f"Node analysis table saved to {out_path}")

    print("\nDone.")