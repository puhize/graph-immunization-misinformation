"""
Repository tidy-up (run locally from the project root): python cleanup.py
Moves bug-hunt / verification scratch artifacts into scratch/ and renames the
typo'd analysis script. Idempotent and safe — only touches known scratch files,
never the canonical results CSVs, caches, or source modules.
"""
import os, shutil

SCRATCH = "scratch"
os.makedirs(SCRATCH, exist_ok=True)

# scratch artifacts left over from the verification/bug-hunt phase
move = [
    "results/bq2.csv", "results/bq3.csv", "results/bq_fast.csv", "results/bridge_quick.csv",
    "results/node_analysis.csv",
    "results/raw/strat_eff_raw.csv", "results/raw/strat_efficiency.csv",
    "results/cache/vitality_n4039_m88234.json",            # obsolete (old bridge method)
    "quick_verify.py", "fast_verify.py", "strat_eff.py",   # scratch scripts
]
# plus any *.bak backups in results/raw
for d in ["results/raw"]:
    if os.path.isdir(d):
        move += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".bak")]

for src in move:
    if os.path.exists(src):
        dst = os.path.join(SCRATCH, os.path.basename(src))
        shutil.move(src, dst)
        print("moved", src, "->", dst)

# Q3: fix the filename typo
if os.path.exists("analyze_reults.py") and not os.path.exists("analyze_results.py"):
    shutil.move("analyze_reults.py", "analyze_results.py")
    print("renamed analyze_reults.py -> analyze_results.py")

print("cleanup done. Review scratch/ and delete it when you're satisfied.")
