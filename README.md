# Vehicle Routing Optimization with Genetic Algorithms

A genetic algorithm implementation
Vehicle Routing Problem (VRP): dispatch a fixed fleet of vehicles from a central depot to
visit a set of customer locations, minimizing total distance traveled while keeping each
vehicle's workload roughly balanced.

## The Problem

The Vehicle Routing Problem is a classic combinatorial optimization problem at the core of
logistics and supply chain planning, think delivery fleets, field service dispatch, or
last-mile shipping. Given a depot and a set of stops, the goal is to design routes for
multiple vehicles that cover every stop exactly once at minimum cost. It's a generalization
of the Traveling Salesman Problem to multiple vehicles, and like TSP, it's NP-hard: the
number of possible route combinations explodes combinatorially as stops are added, so exact
solutions become impractical at real-world scale. That makes it a good fit for
metaheuristics like genetic algorithms, which trade a guarantee of optimality for the
ability to find good solutions quickly on large problems.

## Approach

Each **individual** in the population is a permutation of location indices. That permutation
is split into `NUM_VEHICLES` interleaved routes (vehicle *i* takes every *i*-th stop in the
order), and each route starts and ends at the depot.

**Fitness** is two objectives, both minimized:
1. **Total distance** — the sum of every vehicle's route length.
2. **Balance penalty** — the standard deviation of distance across vehicles, so the GA is
   discouraged from dumping most of the work on one vehicle while another sits idle.

**Genetic operators:**
- **Crossover — Partially Matched Crossover (PMX):** standard two-point crossover would
  produce invalid permutations (duplicate or missing stops). PMX swaps a segment between two
  parents while remapping conflicts, so every offspring stays a valid permutation.
- **Mutation — shuffle indexes:** each position has a small independent probability of
  swapping with another random position, nudging the route order without wrecking it.
- **Selection — tournament selection:** small random subsets of the population compete, and
  the fittest of each subset survives — simple and effective for this problem size.

## Results

Convergence over 300 generations (best total distance found per generation) — actual chart
generated on each run, saved as `convergence.png`:

![Convergence plot](assets/convergence.png)

The best route found, with each vehicle's stops color-coded — saved as `optimal_route.png`:

![Optimal route](assets/optimal_route.png)

## Getting Started

### Install
```bash
pip install -r requirement.txt
```

### Run
```bash
python vrp_solution_code.py
```

This will:
- generate `NUM_LOCATIONS` random customer locations around a fixed depot,
- run the GA for 300 generations over a population of 300,
- print the best route found (per-vehicle stop counts and distances),
- save `convergence.png` and `optimal_route.png` to the working directory.

### Configure
Edit the constants at the top of `vrp_solution.py`:

| Constant | Meaning |
|---|---|
| `NUM_LOCATIONS` | number of customer stops (excluding the depot) |
| `NUM_VEHICLES` | size of the fleet |
| `DEPOT` | fixed depot coordinate |
| `RANDOM_SEED` | fixes both the generated locations and the GA run, for reproducibility |

## Project Structure
```
.
├── vrp_solution.py      # GA setup, fitness function, plotting, main entrypoint
├── requirements.txt
├── assets/
│   ├── convergence.png
│   └── optimal_route.png
└── README.md
```

## Possible Extensions
- Add vehicle capacity constraints (capacitated VRP).
- Add time windows per stop (VRPTW).
- Swap the weighted multi-objective fitness for a proper Pareto front (NSGA-II, which DEAP
  also supports) to explore the distance/balance trade-off explicitly instead of collapsing
  it into one dominance ordering.
- Benchmark PMX against other permutation crossovers (order crossover, cycle crossover) and
  compare convergence speed.

