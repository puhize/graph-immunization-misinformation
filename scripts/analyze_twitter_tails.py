"""
Twitter extinction-tail + rank-significance analysis.
Run AFTER results/raw/twitter_uniform_raw_runs.csv is fully present (500 runs/cell).
Produces: per-budget extinction counts, medians, Welch t-test, Mann-Whitney U,
and a histogram figure of the high-budget cells (results/fig_extinction_twitter.png).

    python analyze_twitter_tails.py
"""
import csv, statistics as st
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RAW = "results/raw/twitter_uniform_raw_runs.csv"
N_TOTAL = 81306
EXT = 5000  # runs below this (vs typical 30k-48k) = extinction/near-extinction

def load(strategy, budget):
    return [int(r["total_infected"]) for r in csv.DictReader(open(RAW))
            if r["strategy"] == strategy and r["budget"] == budget]

budgets = ["0.01", "0.05", "0.1", "0.15", "0.2"]
print(f"{'budget':>6} {'betw_med':>9} {'bridge_med':>10} {'betw_ext':>9} {'bridge_ext':>11} "
      f"{'Welch_p':>10} {'MWU_p':>10} {'winner':>8}")
for b in budgets:
    be = load("betweenness_approx", b)
    br = load("community_bridge", b)
    if not be or not br:
        print(f"{b:>6}  (missing rows — file not fully synced)"); continue
    _, wp = stats.ttest_ind(be, br, equal_var=False)
    _, mp = stats.mannwhitneyu(be, br, alternative="two-sided")
    winner = "bridge" if st.median(br) < st.median(be) else "betw"
    print(f"{b:>6} {st.median(be):>9.0f} {st.median(br):>10.0f} "
          f"{sum(x<EXT for x in be):>9} {sum(x<EXT for x in br):>11} "
          f"{wp:>10.1e} {mp:>10.1e} {winner:>8}")

# histogram of the high-budget cells (the extinction tail)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, b in zip(axes, ["0.15", "0.2"]):
    for s, c, lbl in [("betweenness_approx", "#1F3864", "Ndërmjetësia"),
                      ("community_bridge", "#C0392B", "Ura e komunitetit")]:
        v = load(s, b)
        if v:
            ax.hist(v, bins=45, alpha=0.55, color=c, label=lbl)
    ax.set_title(f"Twitter, buxheti {int(float(b)*100)}%")
    ax.set_xlabel("Numri total i të infektuarve"); ax.set_ylabel("Numri i simulimeve")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/fig_extinction_twitter.png", dpi=150, bbox_inches="tight")
print("\nsaved results/fig_extinction_twitter.png")
