"""
Swap the stale modularity-vitality community_bridge rows for the corrected
inter-community-degree numbers, in facebook_{uniform,personalized}.csv and
their _raw_runs files. Safe and idempotent: it ONLY replaces community_bridge
rows (pulled from community_bridge_fixed_*.csv) and never touches random,
degree, betweenness, or stratified. Backs up each file to *.bak once.

Run from the project root:  python update_csvs.py
"""
import os, shutil
import pandas as pd

ORDER = {"random": 0, "degree": 1, "betweenness": 2, "community_bridge": 3, "stratified": 4}

for mode in ["uniform", "personalized"]:
    for suffix in ["", "_raw_runs"]:
        main  = f"results/raw/facebook_{mode}{suffix}.csv"
        fixed = f"results/raw/community_bridge_fixed_{mode}{suffix}.csv"
        if not (os.path.exists(main) and os.path.exists(fixed)):
            print(f"skip (missing): {main}"); continue
        if not os.path.exists(main + ".bak"):
            shutil.copy(main, main + ".bak")
        df = pd.read_csv(main)
        kept = sorted(set(df["strategy"]) - {"community_bridge"})
        df = df[df["strategy"] != "community_bridge"]
        df = pd.concat([df, pd.read_csv(fixed)], ignore_index=True)
        df["_o"] = df["strategy"].map(lambda s: ORDER.get(s, 99))
        cols = ["_o", "budget"] + (["run_id"] if "run_id" in df.columns else [])
        df = df.sort_values(cols).drop(columns="_o")
        df.to_csv(main, index=False)
        print(f"updated {main}: strategies preserved = {kept} + corrected community_bridge")
print("done. Backups saved as *.bak")
