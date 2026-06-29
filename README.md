# Graph-based immunization strategies against misinformation

Code for the bachelor thesis *"Krahasimi i strategjive të algoritmeve të bazuara në graf për
kontrollimin e shpërndarjes së dezinformimit"* (University of Prishtina, 2026).

Misinformation spread is modeled as a discrete-time stochastic **SIR** process over real social
graphs. Fact-checkers have a limited budget and can "immunize" (pre-correct) only a fraction of
nodes; the question is **which nodes to immunize first**. Four strategies are compared — *random*,
*degree*, *betweenness centrality*, and a *community-bridge* strategy (top inter-community-degree
nodes) — across five budget levels, with 500 Monte Carlo runs per configuration, on two SNAP
networks (Facebook, Twitter), plus a community-level **fairness** analysis using the Gini coefficient.

## Key findings

- **The optimal strategy depends on the budget.** Betweenness wins at low budgets; the
  community-bridge strategy overtakes it at high budgets. On Facebook the crossover sits between the
  **10% and 15%** budgets; on the larger, more fragmented Twitter network it shifts to **15–20%**.
- **Robust.** The crossover holds under a lower reproduction number (R₀ = 5 → 2), a
  clustering-linked (heterogeneous) susceptibility model, and an alternative community-detection
  algorithm (Leiden: the immunized node set overlaps 88–93% with Louvain).
- **Efficiency vs fairness.** The most effective structural strategies are the least equitable: at
  the 20% budget on Facebook, the community bridge reaches the lowest total infections (1,469) but
  the highest Gini (0.660 vs 0.062 for random) — it contains the rumor by isolating, and thus
  sacrificing, some communities.

## Repository structure

```
config.py              experiment parameters (β, γ, R₀, budgets, runs, strategies)
src/                   core engine
  graph_loader.py        load graph, clustering, personalized/echo β
  sir_model.py           SIR engines (reference, fast "push", vectorized sparse)
  strategies.py          random / degree / betweenness(+approx) / community_bridge
  simulation.py          Facebook Monte Carlo driver  ->  results/raw/facebook_*.csv
  metrics.py             community fairness (Gini)
  run_twitter.py, precompute_twitter_betweenness.py   large-network helpers
scripts/               run + analysis (run from the repo root)
  run_network.py         large-network (Twitter) four-strategy comparison
  run_sweep.py           robustness sweeps (R₀, susceptibility model)
  make_significance.py   Welch t-test + Cohen's d
  analyze_results.py     efficiency tables + figures
  analyze_twitter_tails.py   medians / Mann–Whitney / extinction tails (Twitter)
  validate_approx_betweenness.py   k=500 approximation vs exact
  leiden_robustness.py   Louvain-vs-Leiden bridge-set overlap
  analyze_vitality.py, rerun_bridge.py, update_csvs.py, cleanup.py
tests/                 unit tests
data/                  facebook_combined.txt (committed); twitter_combined.txt (download — see below)
results/               raw/ CSVs, figures (*.png), cache/ (precomputed centralities)
```

## Requirements

Python 3.12. Install dependencies:

```bash
pip install -r requirements.txt
# optional, only for scripts/leiden_robustness.py:
pip install igraph leidenalg
```

## Data

- **Facebook** ego network — committed at `data/facebook_combined.txt`
  (SNAP: https://snap.stanford.edu/data/ego-Facebook.html).
- **Twitter** ego network — too large to commit; download `twitter_combined.txt` from
  https://snap.stanford.edu/data/ego-Twitter.html and place it in `data/`.
  The directed file (1,768,149 edges) is loaded **undirected** by the code, giving 1,342,310 edges.

## Reproducing the results

Run everything **from the repository root** (scripts resolve `src/`, `config`, `data/`, and
`results/` relative to the current directory). Determinism comes from fixed seeds
(Louvain `random_state=42`, approximate betweenness `seed=42`); precomputed centralities are cached
in `results/cache/`.

| Command (from repo root) | Produces |
|---|---|
| `python src/simulation.py --beta uniform` | `results/raw/facebook_uniform*.csv` |
| `python src/metrics.py` | Facebook community Gini (`results/raw/fairness_uniform.csv`) |
| `python scripts/make_significance.py` | `results/significance_tests_fixed.csv` |
| `python scripts/run_sweep.py --beta echo --r0 2` | robustness sweeps |
| `python scripts/run_network.py --tag twitter --runs 500 --fairness` | `results/raw/twitter_*.csv` |
| `python scripts/analyze_twitter_tails.py` | Twitter medians / Mann–Whitney |
| `python scripts/validate_approx_betweenness.py` | k=500 approximation check |
| `python scripts/leiden_robustness.py` | `results/leiden_overlap.csv` |
| `python scripts/analyze_results.py` | efficiency tables + figures in `results/` |

Run the tests with:

```bash
python tests/test_sir.py
python tests/test_strategies.py
python tests/test_twitter_graph.py
```

## Citation

If you use this code, please cite the thesis (see `CITATION.cff`):

> Rexha, P. (2026). *Krahasimi i strategjive të algoritmeve të bazuara në graf për kontrollimin e
> shpërndarjes së dezinformimit.* Bachelor thesis, University of Prishtina.

## License

MIT © 2026 Puhiza Rexha — see `LICENSE`.
