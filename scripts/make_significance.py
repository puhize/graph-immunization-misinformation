"""
Generate results/significance_tests_fixed.csv — the statistical comparison of the
corrected community_bridge strategy against each baseline (betweenness, degree,
random), for both beta modes, from the per-run Monte Carlo outputs.

This is the script that produces the headline significance table cited in the
write-up (previously the file existed without a committed generator; see
CODE_REVIEW.md item R3).

Method
  * Welch's t-test (equal_var=False) on the 500 per-run total_infected values
    of community_bridge vs each baseline, at every budget level.
  * Cohen's d with pooled standard deviation (ddof=1) as the effect size.
  * Sign convention: community_bridge minus baseline, so a positive value means
    the bridge infected MORE (i.e. the baseline contained better) and a negative
    value means the bridge contained better.

Input : results/raw/facebook_{uniform,personalized}_raw_runs.csv
Output: results/significance_tests_fixed.csv

Run from the project root:  python make_significance.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

RAW_DIR = "results/raw"
OUT_PATH = "results/significance_tests_fixed.csv"

BETA_MODES = ["uniform", "personalized"]
BUDGETS = [0.01, 0.05, 0.10, 0.15, 0.20]
BASELINES = ["betweenness", "degree", "random"]  # strategy_b, in output order
BRIDGE = "community_bridge"


def cohen_d(a, b):
    """Pooled Cohen's d effect size (a minus b)."""
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def runs(df, strategy, budget):
    return df[(df["strategy"] == strategy) & (df["budget"] == budget)]["total_infected"].values


def main():
    rows = []
    for mode in BETA_MODES:
        raw_path = os.path.join(RAW_DIR, f"facebook_{mode}_raw_runs.csv")
        if not os.path.exists(raw_path):
            print(f"skip (missing): {raw_path}")
            continue
        df = pd.read_csv(raw_path)
        for budget in BUDGETS:
            bridge = runs(df, BRIDGE, budget)
            if len(bridge) == 0:
                continue
            for b_name in BASELINES:
                base = runs(df, b_name, budget)
                if len(base) == 0:
                    continue
                t_stat, p_val = stats.ttest_ind(bridge, base, equal_var=False)  # Welch
                rows.append({
                    "beta_mode": mode,
                    "budget": budget,
                    "strategy_a": BRIDGE,
                    "strategy_b": b_name,
                    "mean_bridge": round(float(np.mean(bridge)), 1),
                    "mean_b": round(float(np.mean(base)), 1),
                    "t_stat": round(float(t_stat), 3),
                    "p_value": float(p_val),
                    "cohen_d": round(float(cohen_d(bridge, base)), 3),
                    "significance": stars(p_val),
                })

    out = pd.DataFrame(rows, columns=["beta_mode", "budget", "strategy_a", "strategy_b",
                                      "mean_bridge", "mean_b", "t_stat", "p_value",
                                      "cohen_d", "significance"])
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()
