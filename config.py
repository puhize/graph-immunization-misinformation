# config.py

# Graph
GRAPH_PATH = "data/facebook_combined.txt"

# SIR parameters
BETA_BASE = 0.05       # base infection rate
GAMMA = 0.01           # recovery rate
ALPHA = 0.5            # skepticism strength for personalized β

# Simulation
MONTE_CARLO_RUNS = 500  # number of runs per configuration
T_MAX = 200             # max time steps per simulation
INITIAL_INFECTED = 5    # number of nodes infected at start

# Budget levels (fraction of nodes immunized)
BUDGET_LEVELS = [0.01, 0.05, 0.10, 0.15, 0.20]

# Strategies to run
STRATEGIES = ["random", "degree", "betweenness", "community_bridge"]