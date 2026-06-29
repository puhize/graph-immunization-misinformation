# src/sir_model.py

import random


def run_sir(G, beta, gamma, immunized_nodes, t_max, initial_infected_count):
    """
    Run one SIR simulation on graph G.

    Parameters:
        G: NetworkX graph
        beta: dict {node: beta_i} for personalized susceptibility,
              OR float for uniform susceptibility
        gamma: float, recovery probability per time step
        immunized_nodes: set of nodes pre-set to Recovered (cannot be infected)
        t_max: int, maximum time steps
        initial_infected_count: int, number of random nodes to infect at start

    Returns:
        dict with keys:
            - total_infected: nodes that were ever infected (excluding immunized)
            - peak_infected: max simultaneously infected at any time step
            - duration: time steps until no infected remain
            - history: list of (S_count, I_count, R_count) per time step
    """
    # Determine if beta is uniform or personalized
    if isinstance(beta, (int, float)):
        get_beta = lambda _: beta
    else:
        get_beta = lambda node: beta[node]

    # Initialize states: S=0, I=1, R=2
    S, I, R = 0, 1, 2
    state = {}

    # All nodes start susceptible
    for node in G.nodes():
        state[node] = S

    # Apply immunization (set to Recovered before simulation)
    for node in immunized_nodes:
        state[node] = R

    # Select initial infected nodes from non-immunized nodes
    available = [n for n in G.nodes() if state[n] == S]
    count = min(initial_infected_count, len(available))
    initial_infected = random.sample(available, count)
    for node in initial_infected:
        state[node] = I

    # Track history
    n_total = G.number_of_nodes()
    n_immunized = len(immunized_nodes)

    s_count = sum(1 for v in state.values() if v == S)
    i_count = sum(1 for v in state.values() if v == I)
    r_count = sum(1 for v in state.values() if v == R)

    history = [(s_count, i_count, r_count)]
    peak_infected = i_count
    duration = 0

    # Pre-compute adjacency lists for performance
    adj = {node: list(G.neighbors(node)) for node in G.nodes()}

    # Run simulation
    for t in range(1, t_max + 1):
        new_infections = []
        new_recoveries = []

        # Collect currently infected nodes
        infected_nodes = [n for n, s in state.items() if s == I]

        if not infected_nodes:
            break

        # For each susceptible node, check infection from each infected neighbor
        susceptible_nodes = [n for n, s in state.items() if s == S]
        for node in susceptible_nodes:
            beta_i = get_beta(node)
            for neighbor in adj[node]:
                if state[neighbor] == I:
                    if random.random() < beta_i:
                        new_infections.append(node)
                        break  # node is infected, no need to check more neighbors

        # Recovery: each infected node recovers with probability gamma
        for node in infected_nodes:
            if random.random() < gamma:
                new_recoveries.append(node)

        # Apply state changes
        for node in new_infections:
            state[node] = I
        for node in new_recoveries:
            state[node] = R

        # Update counts
        s_count = sum(1 for v in state.values() if v == S)
        i_count = sum(1 for v in state.values() if v == I)
        r_count = sum(1 for v in state.values() if v == R)

        history.append((s_count, i_count, r_count))
        peak_infected = max(peak_infected, i_count)
        duration = t

    # Total infected = everyone who left S state (excluding immunized)
    total_infected = n_total - s_count - n_immunized

    return {
        "total_infected": total_infected,
        "peak_infected": peak_infected,
        "duration": duration,
        "history": history,
    }


def run_sir_fast(adj, nodes, beta, gamma, immunized, t_max, init_count):
    """
    Fast push-based SIR run. Statistically equivalent to run_sir (validated:
    matched means within Monte-Carlo noise), but iterates only the infected
    frontier each step instead of scanning all nodes, so it is much faster when
    the epidemic is contained. Same synchronous update, the same
    1-(1-beta)^m per-step infection probability, and recovery evaluated on the
    pre-step infected set.

    Parameters:
        adj: dict {node: list of neighbours} (precompute once, reuse across runs)
        nodes: list of all nodes
        beta: float (uniform) or dict {node: beta_i} (personalized)
        gamma: float recovery probability per step
        immunized: set of nodes pre-set to Recovered
        t_max: int, maximum time steps
        init_count: int, number of random initial infected nodes

    Returns:
        (ever_infected, peak, duration) where
          ever_infected = set of nodes ever infected (excludes immunized);
                          total_infected == len(ever_infected)
          peak          = max simultaneously infected
          duration      = number of steps simulated
    """
    bget = (lambda _n: beta) if isinstance(beta, (int, float)) else (lambda n: beta[n])
    state = {n: 0 for n in nodes}            # 0=S 1=I 2=R
    for n in immunized:
        state[n] = 2

    avail = [n for n in nodes if state[n] == 0]
    seeds = random.sample(avail, min(init_count, len(avail)))
    infected = set(seeds)
    ever = set(seeds)
    for n in seeds:
        state[n] = 1
    peak = len(seeds)
    duration = 0

    for t in range(1, t_max + 1):
        if not infected:
            break
        new_inf = []
        for u in infected:
            for v in adj[u]:
                if state[v] == 0 and random.random() < bget(v):
                    state[v] = 1
                    new_inf.append(v)
        recov = [u for u in infected if random.random() < gamma]
        for u in recov:
            state[u] = 2
        infected.difference_update(recov)
        infected.update(new_inf)
        ever.update(new_inf)
        if len(infected) > peak:
            peak = len(infected)
        duration = t

    return ever, peak, duration


# ---------------------------------------------------------------------------
# Vectorized SIR (scipy sparse) — for large graphs (e.g. Twitter, ~81k nodes)
# ---------------------------------------------------------------------------
import numpy as _np
import networkx as _nx


def build_adjacency(G, nodes=None):
    """
    Build a CSR sparse adjacency matrix for the vectorized SIR.
    Returns (A, nodes, index_of) where A is symmetric (undirected), nodes is the
    ordered node list, and index_of maps node -> row index.
    """
    if nodes is None:
        nodes = list(G.nodes())
    index_of = {n: i for i, n in enumerate(nodes)}
    A = _nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=_np.float64, format="csr")
    return A, nodes, index_of


def run_sir_vectorized(A, beta, gamma, immunized_idx, t_max, init_count, rng=None, return_ever=False):
    """
    Vectorized stochastic SIR on a CSR adjacency matrix. Statistically equivalent
    to run_sir / run_sir_fast (same 1-(1-beta)^m per-step infection rule,
    synchronous update, recovery on the pre-step infected set), but each step is a
    single sparse matrix-vector product so it scales to ~10^5 nodes.

    Parameters:
        A: scipy CSR adjacency (n x n), symmetric
        beta: float (uniform) OR numpy array length n (personalized, index-aligned)
        gamma: float recovery probability per step
        immunized_idx: iterable of node INDICES pre-set to Recovered
        t_max: int max steps
        init_count: int number of random initial infected
        rng: numpy Generator (defaults to a fresh default_rng)

    Returns:
        (total_infected, peak, duration) — total_infected excludes immunized.
    """
    if rng is None:
        rng = _np.random.default_rng()
    n = A.shape[0]
    state = _np.zeros(n, dtype=_np.int8)          # 0=S 1=I 2=R
    imm = _np.fromiter(immunized_idx, dtype=_np.int64)
    if imm.size:
        state[imm] = 2

    avail = _np.where(state == 0)[0]
    k = min(init_count, avail.size)
    seeds = rng.choice(avail, size=k, replace=False)
    state[seeds] = 1
    ever = _np.zeros(n, dtype=bool)
    ever[seeds] = True

    beta_arr = beta if isinstance(beta, _np.ndarray) else None
    peak = int(k)
    duration = 0

    for t in range(1, t_max + 1):
        I_mask = state == 1
        if not I_mask.any():
            break
        m = A.dot(I_mask.astype(_np.float64))      # infected-neighbour count per node
        S_mask = state == 0
        if beta_arr is None:
            p = 1.0 - (1.0 - beta) ** m
        else:
            p = 1.0 - (1.0 - beta_arr) ** m
        new_inf = S_mask & (rng.random(n) < p)
        recov = I_mask & (rng.random(n) < gamma)   # recovery on pre-step infected
        state[new_inf] = 1
        state[recov] = 2
        ever |= new_inf
        ni = int((state == 1).sum())
        if ni > peak:
            peak = ni
        duration = t

    total = int(ever.sum())
    if return_ever:
        return total, peak, duration, ever
    return total, peak, duration
