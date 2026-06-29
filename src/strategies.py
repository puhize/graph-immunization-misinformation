# src/strategies.py

import os
import json
import random
import networkx as nx
import community as community_louvain

# Resolve paths relative to the project root (one level up from src/)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _top_k_nodes(scored_nodes, k):
    """Return set of top-k nodes from a list of (node, score) pairs."""
    ranked = sorted(scored_nodes, key=lambda x: x[1], reverse=True)
    return set(node for node, _ in ranked[:k])


def _budget_to_count(G, budget):
    """Convert budget fraction (0–1) to an integer node count."""
    return max(1, int(budget * G.number_of_nodes()))


# ---------------------------------------------------------------------------
# Strategy 1: Random
# ---------------------------------------------------------------------------

def random_strategy(G, budget):
    """
    Select a random sample of nodes to immunize.

    Parameters:
        G: NetworkX graph
        budget: float (0–1), fraction of nodes to immunize

    Returns:
        set of nodes to immunize
    """
    k = _budget_to_count(G, budget)
    return set(random.sample(list(G.nodes()), k))


# ---------------------------------------------------------------------------
# Strategy 2: Degree-based
# ---------------------------------------------------------------------------

def degree_strategy(G, budget):
    """
    Immunize the top-k nodes by degree centrality.
    High-degree nodes are hubs — removing them fragments the network.

    Parameters:
        G: NetworkX graph
        budget: float (0–1), fraction of nodes to immunize

    Returns:
        set of nodes to immunize
    """
    k = _budget_to_count(G, budget)
    return _top_k_nodes(G.degree(), k)


# ---------------------------------------------------------------------------
# Strategy 3: Betweenness-based (with file caching)
# ---------------------------------------------------------------------------

BETWEENNESS_CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")


def _betweenness_cache_path(G):
    """Build a deterministic cache filename based on graph size."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return os.path.join(BETWEENNESS_CACHE_DIR, f"betweenness_n{n}_m{m}.json")


def _load_or_compute_betweenness(G):
    """
    Load betweenness centrality from cache if available,
    otherwise compute with Brandes' algorithm and save to disk.
    """
    path = _betweenness_cache_path(G)

    if os.path.exists(path):
        with open(path, "r") as f:
            # JSON keys are strings — convert back to int
            raw = json.load(f)
            return {int(k): v for k, v in raw.items()}

    print(f"Computing betweenness centrality (this may take a minute)...")
    bc = nx.betweenness_centrality(G)

    os.makedirs(BETWEENNESS_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in bc.items()}, f)
    print(f"Cached betweenness centrality to {path}")

    return bc


def betweenness_strategy(G, budget):
    """
    Immunize the top-k nodes by exact betweenness centrality.
    These nodes sit on the most shortest paths — key bridges for spread.
    Suitable for small/medium graphs (< ~10K nodes).

    Parameters:
        G: NetworkX graph
        budget: float (0–1), fraction of nodes to immunize

    Returns:
        set of nodes to immunize
    """
    k = _budget_to_count(G, budget)
    bc = _load_or_compute_betweenness(G)
    return _top_k_nodes(bc.items(), k)


# ---------------------------------------------------------------------------
# Strategy 3b: Approximate betweenness (for large graphs)
# ---------------------------------------------------------------------------

def _approx_betweenness_cache_path(G, sample_k):
    """Cache filename for approximate betweenness."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return os.path.join(BETWEENNESS_CACHE_DIR,
                        f"betweenness_approx_k{sample_k}_n{n}_m{m}.json")


def _load_or_compute_approx_betweenness(G, sample_k=500):
    """
    Load approximate betweenness centrality from cache if available,
    otherwise compute using k random pivot nodes and save to disk.
    Much faster than exact computation for large graphs.
    """
    path = _approx_betweenness_cache_path(G, sample_k)

    if os.path.exists(path):
        with open(path, "r") as f:
            raw = json.load(f)
            return {int(k): v for k, v in raw.items()}

    print(f"Computing approximate betweenness centrality "
          f"(k={sample_k} samples, {G.number_of_nodes()} nodes)...")
    bc = nx.betweenness_centrality(G, k=sample_k, seed=42)

    os.makedirs(BETWEENNESS_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in bc.items()}, f)
    print(f"Cached approximate betweenness to {path}")

    return bc


def betweenness_approx_strategy(G, budget, sample_k=500):
    """
    Immunize the top-k nodes by approximate betweenness centrality.
    Uses k random pivot nodes instead of all-pairs shortest paths.
    Suitable for large graphs (> 10K nodes).

    Parameters:
        G: NetworkX graph
        budget: float (0–1), fraction of nodes to immunize
        sample_k: int, number of pivot nodes for approximation

    Returns:
        set of nodes to immunize
    """
    k = _budget_to_count(G, budget)
    bc = _load_or_compute_approx_betweenness(G, sample_k)
    return _top_k_nodes(bc.items(), k)


# ---------------------------------------------------------------------------
# Strategy 4: Community bridge (inter-community degree)
# ---------------------------------------------------------------------------

def _bridge_cache_path(G):
    """Cache filename for inter-community degree (bridge) scores."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return os.path.join(BETWEENNESS_CACHE_DIR, f"bridge_n{n}_m{m}.json")


def _load_or_compute_bridge_scores(G):
    """
    Load inter-community degree (bridge) scores from cache if available,
    otherwise compute them from a fixed Louvain partition and save to disk.

    Bridge score of node i = number of i's neighbors that belong to a
    DIFFERENT Louvain community than i. A high score means the node carries
    many edges between communities, so immunizing it cuts the channels that
    misinformation uses to cross from one community to another.

    Note: we count cross-community neighbors (not their fraction). The count
    equals fraction * degree, which targets nodes that carry MANY crossing
    edges — the high-impact bridges — instead of a degree-1 node whose single
    edge happens to cross (fraction 1.0 but negligible containment value).
    """
    path = _bridge_cache_path(G)

    if os.path.exists(path):
        with open(path, "r") as f:
            raw = json.load(f)
            return {int(k): v for k, v in raw.items()}

    print("Detecting communities (Louvain) and scoring bridges...")

    # Use the SHARED Louvain partition (same as metrics.py / stratified) so the
    # community split is identical everywhere it is used.
    partition = _load_or_compute_partition(G)

    scores = {}
    for node in G.nodes():
        c = partition[node]
        scores[node] = sum(1 for nb in G.neighbors(node) if partition[nb] != c)

    # Cache to disk
    os.makedirs(BETWEENNESS_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in scores.items()}, f)
    print(f"Cached bridge scores to {path}")

    return scores


def community_bridge_strategy(G, budget):
    """
    Immunize the top-k nodes by inter-community degree (true bridges).

    Each node is scored by how many of its neighbors lie in other Louvain
    communities. The highest-scoring nodes are the connectors between
    communities — immunizing them isolates misinformation within communities.

    Parameters:
        G: NetworkX graph
        budget: float (0–1), fraction of nodes to immunize

    Returns:
        set of nodes to immunize
    """
    k = _budget_to_count(G, budget)
    scores = _load_or_compute_bridge_scores(G)
    return _top_k_nodes(scores.items(), k)


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "random": random_strategy,
    "degree": degree_strategy,
    "betweenness": betweenness_strategy,
    "betweenness_approx": betweenness_approx_strategy,
    "community_bridge": community_bridge_strategy,
}


def get_immunized_nodes(G, strategy_name, budget):
    """
    Dispatch to the appropriate strategy function.

    Parameters:
        G: NetworkX graph
        strategy_name: one of "random", "degree", "betweenness", "community_bridge"
        budget: float (0–1), fraction of nodes to immunize

    Returns:
        set of nodes to immunize
    """
    if strategy_name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                         f"Choose from {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[strategy_name](G, budget)


# ---------------------------------------------------------------------------
# Strategy 5: Fairness-aware stratified immunization (Contribution 2 follow-up)
# ---------------------------------------------------------------------------

def _partition_cache_path(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    return os.path.join(BETWEENNESS_CACHE_DIR, f"louvain_partition_n{n}_m{m}.json")


def _load_or_compute_partition(G):
    """Louvain partition (node -> community id), fixed seed 42, cached.
    Uses the SAME cache file as metrics.py so the community split is consistent."""
    path = _partition_cache_path(G)
    if os.path.exists(path):
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    partition = community_louvain.best_partition(G, random_state=42)
    os.makedirs(BETWEENNESS_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in partition.items()}, f)
    return partition


def stratified_strategy(G, budget):
    """
    Fairness-aware immunization. The budget is divided across Louvain
    communities in proportion to their size (largest-remainder apportionment so
    the total equals budget*N), and within each community the highest-degree
    members are immunized. Because every community receives coverage in
    proportion to its size, no community is left fully exposed -- this is
    designed to lower the Gini of community infection rates relative to the
    efficiency-only strategies, at some cost to total containment.
    """
    K = _budget_to_count(G, budget)
    partition = _load_or_compute_partition(G)
    deg = dict(G.degree())

    comms = {}
    for node, c in partition.items():
        comms.setdefault(c, []).append(node)

    sizes = {c: len(m) for c, m in comms.items()}
    N = sum(sizes.values())

    raw = {c: K * sizes[c] / N for c in comms}
    quota = {c: int(raw[c]) for c in comms}
    remainder = K - sum(quota.values())
    for c in sorted(comms, key=lambda c: raw[c] - quota[c], reverse=True)[:remainder]:
        quota[c] += 1

    selected = set()
    for c, members in comms.items():
        k_c = min(quota[c], len(members))
        if k_c > 0:
            selected.update(sorted(members, key=lambda n: deg[n], reverse=True)[:k_c])
    return selected


STRATEGY_MAP["stratified"] = stratified_strategy
