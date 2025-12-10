#!/usr/bin/python
import argparse
import glob
from pathlib import Path

# Existing solvers
from cbs import CBSSolver

# Your MA-CBS implementation
from ma_cbs import MACBS

from visualize import Animation
from single_agent_planner import get_sum_of_cost


DEFAULT_SOLVER = "CBS"  # can be overridden with --solver


# ---------- Utility functions ----------

def print_mapf_instance(my_map, starts, goals):
    print("Start locations")
    print_locations(my_map, starts)
    print("Goal locations")
    print_locations(my_map, goals)


def print_locations(my_map, locations):
    display = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i, loc in enumerate(locations):
        display[loc[0]][loc[1]] = i

    s = ""
    for r in range(len(my_map)):
        for c in range(len(my_map[0])):
            if display[r][c] >= 0:
                s += f"{display[r][c]} "
            elif my_map[r][c]:
                s += "@ "
            else:
                s += ". "
        s += "\n"
    print(s)


def import_mapf_instance(filename):
    """Reads a MAPF instance from file (standard format)."""
    fpath = Path(filename)
    if not fpath.is_file():
        raise FileNotFoundError(f"{filename} does not exist.")

    with open(filename, "r") as f:
        # Grid dimensions
        rows, columns = [int(x) for x in f.readline().split()]
        my_map = []

        # Map grid
        for _ in range(rows):
            row_str = f.readline().strip()
            row = []
            for ch in row_str:
                if ch == "@":
                    row.append(True)
                elif ch == ".":
                    row.append(False)
            my_map.append(row)

        # Agent count
        num_agents = int(f.readline().strip())

        starts, goals = [], []
        for _ in range(num_agents):
            sx, sy, gx, gy = [int(x) for x in f.readline().split()]
            starts.append((sx, sy))
            goals.append((gx, gy))

    return my_map, starts, goals


# ---------- Main entry point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs various MAPF algorithms")
    parser.add_argument("--instance", type=str, default=None,
                        help="Instance file or pattern (use quotes for wildcards)")
    parser.add_argument("--batch", action="store_true", default=False,
                        help="Batch mode (no animation)")
    parser.add_argument("--disjoint", action="store_true", default=False,
                        help="Use disjoint splitting for CBS")
    parser.add_argument("--solver", type=str, default=DEFAULT_SOLVER,
                        help="CBS, MACBS, Independent, Prioritized")
    parser.add_argument("--merge_threshold", type=int, default=2,
                        help="Merge threshold for MA-CBS")

    args = parser.parse_args()

    # Output file
    result_file = open("results.csv", "w", buffering=1)

    # Process instance files
    for file in sorted(glob.glob(args.instance)):
        print(f"\n*** Import instance: {file} ***")
        my_map, starts, goals = import_mapf_instance(file)
        print_mapf_instance(my_map, starts, goals)

        # Select solver
        solver_name = args.solver.upper()

        if solver_name == "CBS":
            print("*** Running CBS ***")
            # surpress problem errors
            try:
                solver = CBSSolver(my_map, starts, goals)
                paths = solver.find_solution(disjoint=args.disjoint)#, meta=False)
            except:
                exit()

        elif solver_name == "MACBS":
            print("*** Running Meta-Agent CBS (MA-CBS) ***")
            solver = MACBS(
                my_map,
                starts,
                goals,
                merge_threshold=args.merge_threshold
            )
            paths = solver.find_solution()

        elif solver_name == "INDEPENDENT":
            print("*** Running Independent solver ***")
            solver = IndependentSolver(my_map, starts, goals)
            paths = solver.find_solution()

        elif solver_name == "PRIORITIZED":
            print("*** Running Prioritized Planning solver ***")
            solver = PrioritizedPlanningSolver(my_map, starts, goals)
            paths = solver.find_solution()

        else:
            raise RuntimeError(f"Unknown solver: {args.solver}")

        # Compute cost
        cost = get_sum_of_cost(paths)
        print(f"Total cost: {cost}")
        result_file.write(f"{file},{solver_name},{cost}\n")

        # Optional animation
        if not args.batch:
            print("*** Simulation ***")
            animation = Animation(my_map, starts, goals, paths)
            animation.show()

    result_file.close()
    try:
        print("\n✅ All instances complete. Results saved to results.csv.")
    except:
        print("\nAll instances complete. Results saved to results.csv.")
