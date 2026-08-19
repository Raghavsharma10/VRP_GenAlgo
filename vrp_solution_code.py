# -*- coding: utf-8 -*-
"""
Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)
solved with a Genetic Algorithm (DEAP)
=====================================================================

A single depot dispatches a fleet of capacity-limited vehicles to visit
customer locations that each have a demand (how much capacity they consume)
and a delivery time window (when a vehicle is allowed to arrive). This is
the "real world" version of the classic VRP: a delivery fleet, field
service dispatch, or last-mile courier problem all look like this.

Objectives (all minimized):
  1. total distance driven by all vehicles combined
  2. balance penalty  -- std. dev. of per-vehicle distance, so work is
     spread across the fleet instead of piling onto one vehicle
  3. capacity violation -- how much any vehicle's route exceeds its
     capacity, summed across vehicles (0 for a fully feasible solution)
  4. time-window violation -- total minutes late across all stops
     (0 for a fully feasible solution)

Representation: an individual is still just a permutation of location
indices (a "giant tour"). It's decoded into per-vehicle routes with a
greedy capacity-based split: keep loading the current vehicle until the
next stop would exceed its capacity, then start the next vehicle. This
means route length per vehicle is now a *consequence* of demand, not a
fixed pattern -- unlike a naive round-robin split.

Run with:  python vrp_solution.py
Requires:  deap, numpy, matplotlib   (pip install -r requirements.txt)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# ---------------------------------------------------------------------------
# Task 1 - Problem setup: locations, depot, vehicles
# ---------------------------------------------------------------------------
NUM_LOCATIONS = 15         # customer locations to visit (excludes the depot)
NUM_VEHICLES = 4           # vehicles available to service them
DEPOT = (50, 50)           # fixed central depot coordinate
RANDOM_SEED = 42           # fixed seed -> reproducible data AND GA run

VEHICLE_CAPACITY = 40      # max total demand a single vehicle can carry
VEHICLE_SPEED = 1.0        # distance units per minute (1.0 = distance IS time)
SERVICE_TIME = 10          # minutes spent at each stop (loading/unloading)
DEPOT_START_TIME = 0       # every vehicle departs the depot at t=0
TIME_WINDOW_SPAN = (30, 90)  # each stop's window is [start, start + span]

random.seed(RANDOM_SEED)


def generate_locations(n):
    """Generate synthetic customer data: coordinates, demand, time window.

    To use real data instead of synthetic data, replace this function with
    something that reads a CSV of (x, y, demand, tw_start, tw_end) --
    the rest of the script doesn't care where the data came from, as long
    as each entry has those five fields.
    """
    locs = []
    for _ in range(n):
        x, y = random.randint(0, 100), random.randint(0, 100)
        demand = random.randint(2, 12)
        tw_start = random.randint(0, 60)
        tw_end = tw_start + random.randint(*TIME_WINDOW_SPAN)
        locs.append({
            "x": x, "y": y,
            "demand": demand,
            "tw_start": tw_start,
            "tw_end": tw_end,
        })
    return locs


LOCATIONS = generate_locations(NUM_LOCATIONS)

# ---------------------------------------------------------------------------
# Genetic Algorithm setup
# ---------------------------------------------------------------------------
if "FitnessMin" not in dir(creator):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
if "Individual" not in dir(creator):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(NUM_LOCATIONS), NUM_LOCATIONS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def decode_routes(individual):
    """Split the permutation into capacity-feasible routes (greedy split).

    Walks the chromosome in order, assigning stops to the current vehicle
    until the next stop would exceed VEHICLE_CAPACITY, then moves on to the
    next vehicle. If there are more stops than the fleet can absorb even
    one-per-vehicle-at-a-time, the excess piles onto the last vehicle --
    this shows up as capacity violation in the fitness so the GA is
    pressured to reorder the chromosome to avoid it.

    Returns (routes, loads). routes is a list of routes, one per vehicle,
    each a list of location dicts (NOT including the depot). loads is the
    list of accumulated demand per vehicle.
    """
    routes = [[] for _ in range(NUM_VEHICLES)]
    loads = [0] * NUM_VEHICLES
    vehicle_idx = 0

    for loc_id in individual:
        loc = LOCATIONS[loc_id]
        while (loads[vehicle_idx] + loc["demand"] > VEHICLE_CAPACITY
               and vehicle_idx < NUM_VEHICLES - 1):
            vehicle_idx += 1
        routes[vehicle_idx].append(loc)
        loads[vehicle_idx] += loc["demand"]

    return routes, loads


def simulate_route(route):
    """Drive a single route from the depot, tracking distance and time.

    Returns (distance, arrival_times, lateness). arrival_times is the
    arrival clock time at each stop (for reporting). lateness is total
    minutes past each stop's tw_end, summed across the route (0 if fully
    on-time).
    """
    distance = 0.0
    time = DEPOT_START_TIME
    pos = DEPOT
    arrival_times = []
    lateness = 0.0

    for stop in route:
        stop_pos = (stop["x"], stop["y"])
        leg = dist(pos, stop_pos)
        distance += leg
        time += leg / VEHICLE_SPEED

        if time < stop["tw_start"]:
            time = stop["tw_start"]              # wait for the window to open
        elif time > stop["tw_end"]:
            lateness += time - stop["tw_end"]     # arrived too late

        arrival_times.append(time)
        time += SERVICE_TIME
        pos = stop_pos

    distance += dist(pos, DEPOT)
    return distance, arrival_times, lateness


def evalVRP(individual):
    routes, loads = decode_routes(individual)

    distances = []
    total_lateness = 0.0
    for route in routes:
        d, _, lateness = simulate_route(route)
        distances.append(d)
        total_lateness += lateness

    total_distance = sum(distances)
    balance_penalty = float(np.std(distances))
    capacity_violation = sum(max(0, load - VEHICLE_CAPACITY) for load in loads)
    time_violation = total_lateness

    return total_distance, balance_penalty, capacity_violation, time_violation


toolbox.register("evaluate", evalVRP)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_routes(individual, title="Routes", savepath=None):
    routes, loads = decode_routes(individual)

    plt.figure(figsize=(7, 6))
    for loc in LOCATIONS:
        plt.plot(loc["x"], loc["y"], 'o', color='lightgray', zorder=1)
    plt.plot(DEPOT[0], DEPOT[1], 'rs', markersize=10, label='Depot', zorder=3)

    for i, route in enumerate(routes):
        if not route:
            continue
        pts = [DEPOT] + [(s["x"], s["y"]) for s in route] + [DEPOT]
        xs, ys = zip(*pts)
        load = loads[i]
        plt.plot(xs, ys, '-', marker='o', zorder=2,
                  label=f"Vehicle {i + 1} (load {load}/{VEHICLE_CAPACITY})")

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(fontsize=8)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {savepath}")
    plt.show()


def plot_convergence(logbook, savepath=None):
    """Plot best total-distance objective per generation."""
    gens = logbook.select("gen")
    mins = logbook.select("min")  # list of 4-tuples
    best_distance = [m[0] for m in mins]

    plt.figure()
    plt.plot(gens, best_distance)
    plt.title("GA Convergence: Best Total Distance per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Total Distance")
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {savepath}")
    plt.show()


# ---------------------------------------------------------------------------
# Running the Genetic Algorithm
# ---------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda vals: np.mean(vals, axis=0))
    stats.register("min", lambda vals: np.min(vals, axis=0))

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.7, mutpb=0.2, ngen=300,
        stats=stats, halloffame=hof, verbose=True,
    )

    best = hof[0]
    total_distance, balance_penalty, capacity_violation, time_violation = best.fitness.values
    routes, loads = decode_routes(best)

    print("\n--- Best solution found ---")
    print(f"Total distance:        {total_distance:.2f}")
    print(f"Balance penalty:       {balance_penalty:.2f}")
    print(f"Capacity violation:    {capacity_violation:.2f}  "
          f"({'FEASIBLE' if capacity_violation == 0 else 'OVER CAPACITY'})")
    print(f"Time-window violation: {time_violation:.2f} min late total  "
          f"({'ON TIME' if time_violation == 0 else 'LATE STOPS PRESENT'})")

    for i, route in enumerate(routes):
        d, arrivals, lateness = simulate_route(route)
        print(f"\n  Vehicle {i + 1}: {len(route)} stops, "
              f"load {loads[i]}/{VEHICLE_CAPACITY}, distance {d:.2f}")
        for stop, arrival in zip(route, arrivals):
            status = "LATE" if arrival > stop["tw_end"] else "on time"
            print(f"    -> arrive {arrival:6.1f}  "
                  f"window [{stop['tw_start']:>3},{stop['tw_end']:>3}]  "
                  f"demand {stop['demand']:>2}  [{status}]")

    plot_convergence(logbook, savepath="convergence.png")
    plot_routes(best, "Optimal Route (CVRPTW)", savepath="optimal_route.png")

    return pop, stats, hof, logbook


if __name__ == "__main__":
    main()