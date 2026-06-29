"""
analyze_results.py
------------------
Analysis script for Facebook uniform-beta SIR simulation results.
Produces:
  1. Strategy comparison table (mean ± std)
  2. Welch t-tests (independent samples) between strategy pairs at each budget level
  3. Cohen's d effect sizes
  4. Efficiency curves plot (total infected vs budget)
  5. Std comparison plot (variance across strategies)

Usage:
    python analyze_results.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_CSV     = "results/raw/facebook_uniform.csv"
RAW_RUNS_CSV    = "results/raw/facebook_uniform_raw_runs.csv"
FIGURES_DIR     = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

STRATEGIES  = ["random", "degree", "betweenness", "community_bridge"]
BUDGETS     = [0.01, 0.05, 0.10, 0.15, 0.20]
COLORS      = {
    "random":           "#888888",
    "degree":           "#2196F3",
    "betweenness":      "#E91E63",
    "community_bridge": "#4CAF50",
}
LABELS = {
    "random":           "Random",
    "degree":           "Degree",
    "betweenness":      "Betweenness",
    "community_bridge": "Community Bridge\n(Inter-community degree)",
}


# ─── 1. Load data ──────────────────────────────────────────────────────────────
def load_data():
    agg = pd.read_csv(RESULTS_CSV)
    raw = pd.read_csv(RAW_RUNS_CSV)
    return agg, raw


# ─── 2. Print comparison table ────────────────────────────────────────────────
def print_comparison_table(agg):
    print("\n" + "="*75)
    print("STRATEGY COMPARISON — Facebook ego network, uniform β (R₀=5)")
    print("="*75)
    print(f"{'Strategy':<20} {'Budget':>7} {'Immunized':>10} {'Mean infected':>15} {'Std':>8}")
    print("-"*75)
    for _, row in agg.iterrows():
        print(f"{row['strategy']:<20} {row['budget']:>7.0%} {int(row['immunized_count']):>10} "
              f"{row['total_infected_mean']:>15,.1f} {row['total_infected_std']:>8,.1f}")

    print("\n--- RANKING BY BUDGET (total infected, lower = better) ---")
    for b in BUDGETS:
        sub = agg[agg['budget'] == b].sort_values('total_infected_mean')
        print(f"\n  Budget {b:.0%}:")
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            saved = agg[(agg['strategy'] == 'random') & (agg['budget'] == b)]['total_infected_mean'].values[0] \
                    - row['total_infected_mean']
            print(f"    {rank}. {row['strategy']:<20}  {row['total_infected_mean']:>7,.1f}  "
                  f"(saves {saved:>6,.1f} vs random)")


# ─── 3. Statistical significance tests ────────────────────────────────────────
def cohen_d(a, b):
    """Pooled Cohen's d effect size."""
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0


def run_significance_tests(raw):
    print("\n" + "="*75)
    print("WELCH t-TESTS, independent samples (betweenness vs each other strategy)")
    print("p < 0.001 = ***, p < 0.01 = **, p < 0.05 = *, ns = not significant")
    print("="*75)

    results = []
    for b in BUDGETS:
        bw_runs = raw[(raw['strategy'] == 'betweenness') & (raw['budget'] == b)]['total_infected'].values
        print(f"\n  Budget {b:.0%}:")
        for s in ["random", "degree", "community_bridge"]:
            s_runs = raw[(raw['strategy'] == s) & (raw['budget'] == b)]['total_infected'].values
            if len(bw_runs) == 0 or len(s_runs) == 0:
                print(f"    betweenness vs {s:<20}  NO DATA")
                continue
            t_stat, p_val = stats.ttest_ind(bw_runs, s_runs, equal_var=False)  # Welch
            d = cohen_d(bw_runs, s_runs)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    betweenness vs {s:<20}  t={t_stat:>8.2f}  p={p_val:.2e}  {sig}  d={d:.2f}")
            results.append({
                'budget': b, 'strategy_a': 'betweenness', 'strategy_b': s,
                't_stat': t_stat, 'p_value': p_val, 'cohen_d': d, 'significance': sig
            })

    return pd.DataFrame(results)


# ─── 4. Betweenness bimodal analysis ──────────────────────────────────────────
def analyze_betweenness_bimodal(raw):
    print("\n" + "="*75)
    print("BETWEENNESS BIMODAL DISTRIBUTION ANALYSIS")
    print("="*75)

    THRESHOLD = 2000  # infections below this = epidemic contained

    for b in BUDGETS:
        runs = raw[(raw['strategy'] == 'betweenness') & (raw['budget'] == b)]['total_infected'].values
        if len(runs) == 0:
            continue
        contained  = (runs < THRESHOLD).sum()
        spread     = (runs >= THRESHOLD).sum()
        print(f"\n  Budget {b:.0%}  (n={len(runs)}):")
        print(f"    Epidemic contained (<{THRESHOLD}):  {contained:>4} runs ({contained/len(runs)*100:.1f}%)")
        print(f"    Epidemic spread    (≥{THRESHOLD}):  {spread:>4} runs ({spread/len(runs)*100:.1f}%)")
        print(f"    Min: {runs.min():<6}  Median: {np.median(runs):<7.0f}  Max: {runs.max()}")
        print(f"    Mean: {runs.mean():.1f}  Std: {runs.std():.1f}")


# ─── 5. Efficiency curves plot ────────────────────────────────────────────────
def plot_efficiency_curves(agg):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Strategy Comparison — Facebook Ego Network (Uniform β, R₀=5)",
                 fontsize=13, fontweight='bold', y=1.01)

    budget_pct = [b * 100 for b in BUDGETS]

    # Left: total infected mean
    ax = axes[0]
    for s in STRATEGIES:
        sub = agg[agg['strategy'] == s].sort_values('budget')
        means = sub['total_infected_mean'].values
        stds  = sub['total_infected_std'].values
        ax.plot(budget_pct, means, 'o-', color=COLORS[s], label=LABELS[s], linewidth=2, markersize=6)
        ax.fill_between(budget_pct, means - stds, means + stds, alpha=0.12, color=COLORS[s])
    ax.set_xlabel("Budget (% of nodes immunized)", fontsize=11)
    ax.set_ylabel("Mean total infected", fontsize=11)
    ax.set_title("Total Infected vs Budget\n(shaded = ±1 std)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xticks(budget_pct)
    ax.set_xticklabels([f"{b:.0f}%" for b in budget_pct])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4200)

    # Right: std comparison
    ax = axes[1]
    for s in STRATEGIES:
        sub = agg[agg['strategy'] == s].sort_values('budget')
        ax.plot(budget_pct, sub['total_infected_std'].values, 'o-',
                color=COLORS[s], label=LABELS[s], linewidth=2, markersize=6)
    ax.set_xlabel("Budget (% of nodes immunized)", fontsize=11)
    ax.set_ylabel("Standard deviation of total infected", fontsize=11)
    ax.set_title("Outcome Variance vs Budget\n(higher = more sensitive to seed placement)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xticks(budget_pct)
    ax.set_xticklabels([f"{b:.0f}%" for b in budget_pct])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "efficiency_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


# ─── 6. Betweenness distribution plots ────────────────────────────────────────
def plot_betweenness_distribution(raw):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    fig.suptitle("Betweenness Strategy — Distribution of Outcomes per Run\n"
                 "Facebook Ego Network, Uniform β, R₀=5",
                 fontsize=12, fontweight='bold')

    for i, b in enumerate(BUDGETS):
        ax = axes[i]
        runs = raw[(raw['strategy'] == 'betweenness') & (raw['budget'] == b)]['total_infected'].values
        if len(runs) == 0:
            ax.set_title(f"Budget {b:.0%}\nNO DATA")
            continue
        ax.hist(runs, bins=30, color=COLORS['betweenness'], alpha=0.75, edgecolor='white')
        ax.axvline(runs.mean(), color='black', linestyle='--', linewidth=1.5, label=f"Mean={runs.mean():.0f}")
        ax.axvline(np.median(runs), color='red', linestyle=':', linewidth=1.5, label=f"Median={np.median(runs):.0f}")
        ax.set_title(f"Budget {b:.0%}\n(n={len(runs)})", fontsize=10)
        ax.set_xlabel("Total infected", fontsize=9)
        if i == 0:
            ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "betweenness_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─── 7. v1 vs v2 community bridge comparison ──────────────────────────────────
def print_v1_v2_comparison(agg):
    v1 = {
        0.01: 3952.82, 0.05: 3442.77, 0.10: 2971.25,
        0.15: 1965.38, 0.20: 1781.62
    }
    print("\n" + "="*65)
    print("COMMUNITY BRIDGE: v1 (fraction-based) vs v2 (modularity vitality)")
    print("="*65)
    print(f"{'Budget':>7}  {'v1 mean':>12}  {'v2 mean':>12}  {'Δ infected':>12}")
    print("-"*65)
    for b in BUDGETS:
        v2_mean = agg[(agg['strategy'] == 'community_bridge') & (agg['budget'] == b)]['total_infected_mean'].values
        if len(v2_mean) == 0:
            continue
        v2_mean = v2_mean[0]
        delta = v2_mean - v1[b]
        direction = "WORSE" if delta > 0 else "better"
        print(f"{b:>7.0%}  {v1[b]:>12,.2f}  {v2_mean:>12,.2f}  "
              f"{delta:>+12,.2f}  ({direction})")


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    agg, raw = load_data()
    print(f"Loaded {len(agg)} aggregated rows, {len(raw)} raw run rows.")

    print_comparison_table(agg)
    print_v1_v2_comparison(agg)
    analyze_betweenness_bimodal(raw)

    sig_df = run_significance_tests(raw)
    if not sig_df.empty:
        sig_path = "results/significance_tests.csv"
        sig_df.to_csv(sig_path, index=False)
        print(f"\nSignificance test results saved to {sig_path}")

    print("\nGenerating plots...")
    plot_efficiency_curves(agg)
    plot_betweenness_distribution(raw)

    print("\nDone.")