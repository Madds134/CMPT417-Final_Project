import heapq
import time as timer
from collections import defaultdict
from single_agent_planner import (
    a_star,
    compute_heuristics,
    get_sum_of_cost,
    get_location,
)
from JointSolver import JointSolver

# python run_experiments.py --instance instances/test_10.txt --solver MACBS # ran using this just change the instance number 47 doesnt work like the other ones

def detect_collision(path1, path2):
    """
    Return the first collision between two paths, or None.
    Collision format: {'loc': [...], 'timestep': t}
    Vertex collision: loc = [cell]
    Edge collision:   loc = [cell_from, cell_to]
    """
    timesteps = max(len(path1), len(path2))

    for t in range(timesteps):
        loc1 = get_location(path1, t)
        loc2 = get_location(path2, t)

        # Vertex collision
        if loc1 == loc2:
            return {"loc": [loc1], "timestep": t}

        # Edge collision between t-1 and t
        if t > 0:
            prev1 = get_location(path1, t - 1)
            prev2 = get_location(path2, t - 1)
            if prev1 == loc2 and prev2 == loc1:
                return {"loc": [prev1, loc1], "timestep": t}

    return None


def detect_collisions(paths):
    """
    Return list of first collisions between all agent pairs.
    Each entry: {'a1': i, 'a2': j, 'loc': [...], 'timestep': t}
    """
    collisions = []
    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            c = detect_collision(paths[i], paths[j])
            if c is not None:
                collisions.append(
                    {"a1": i, "a2": j, "loc": c["loc"], "timestep": c["timestep"]}
                )
    return collisions


def standard_splitting(collision):
    """
    Standard CBS splitting:
      Vertex collision at v, time t:
        - forbid a1 at v at time t
        - forbid a2 at v at time t
      Edge collision at (u->v), time t:
        - forbid a1 traversing (u,v) at time t
        - forbid a2 traversing (v,u) at time t
    """
    a1 = collision["a1"]
    a2 = collision["a2"]
    loc = collision["loc"]
    t = collision["timestep"]

    # Vertex collision
    if len(loc) == 1:
        v = loc[0]
        return [
            {"agent": a1, "loc": [v], "timestep": t},
            {"agent": a2, "loc": [v], "timestep": t},
        ]

    # Edge collision
    elif len(loc) == 2:
        u, v = loc[0], loc[1]
        return [
            {"agent": a1, "loc": [u, v], "timestep": t},
            {"agent": a2, "loc": [v, u], "timestep": t},
        ]

    return []


class MACBS:
    """
    Meta-Agent Conflict-Based Search (MA-CBS) with a pluggable low-level solver for meta-agents.
    """

    def __init__(self, my_map, starts, goals, merge_threshold=1, low_level_mode="joint"):
        """
        my_map          : 2D grid (True = obstacle, False = free)
        starts/goals    : list of (row, col)
        merge_threshold : # of conflicts before merging a pair of agents
        low_level_mode  : "joint" for Joint A* on merged groups (for now)
        """
        assert len(starts) == len(goals)
        self.my_map = my_map
        self.starts = list(starts)
        self.goals = list(goals)
        self.num_agents = len(starts)
        self.merge_threshold = merge_threshold
        self.low_level_mode = low_level_mode

        # Solver statistics
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.open_list = []
        self.start_time = None

        # Low-level heuristics for single-agent A*
        self.heuristics = [
            compute_heuristics(my_map, g) for g in self.goals
        ]

        self.hl_closed = set()

    # Meta-group utilities
    def initial_meta_groups(self):
        """Each agent starts in its own meta-group."""
        return [frozenset([i]) for i in range(self.num_agents)]

    def find_group_index(self, meta_groups, agent):
        """Return index of meta-group that contains the agent."""
        for idx, g in enumerate(meta_groups):
            if agent in g:
                return idx
        raise ValueError(f"Agent {agent} not in any meta-group")

    def merge_groups(self, meta_groups, a1, a2):
        """
        Merge two meta-groups containing a1 and a2.
        Return a new list of meta-groups.
        """
        i = self.find_group_index(meta_groups, a1)
        j = self.find_group_index(meta_groups, a2)
        if i == j:
            return meta_groups  # already same group

        g1 = meta_groups[i]
        g2 = meta_groups[j]
        merged = g1.union(g2)

        new_groups = []
        for idx, g in enumerate(meta_groups):
            if idx == i or idx == j:
                continue
            new_groups.append(g)
        new_groups.append(merged)
        return new_groups

    # Low-level planning for a meta-group
    def plan_single_group(self, group, constraints):
        """
        Plan for a group assuming we solve each agent independently with A*.
        Return dict {agent_id: path}, or None if infeasible.
        """
        group_paths = {}
        for a in group:
            h_vals = self.heuristics[a]
            path = a_star(
                self.my_map,
                self.starts[a],
                self.goals[a],
                h_vals,
                a,
                constraints,
            )
            if path is None:
                return None
            group_paths[a] = path
        return group_paths

    def plan_joint_group(self, group, constraints):
        """
        Plan for a group using Joint A*.
        """
        agents = sorted(list(group))
        starts = [self.starts[a] for a in agents]
        goals = [self.goals[a] for a in agents]

        js = JointSolver(self.my_map, starts, goals, agents, constraints)
        joint_paths = js.find_solution()
        if joint_paths is None:
            return None

        group_paths = {}
        for i, a in enumerate(agents):
            group_paths[a] = joint_paths[i]
        return group_paths

    def replan_all_groups(self, constraints, meta_groups):
        """
        Given constraints and a set of meta-groups, replan for all agents.
        Returns a list of paths (index = agent id) or None if infeasible.
        """
        new_paths = [None] * self.num_agents

        for group in meta_groups:
            if len(group) == 1:
                res = self.plan_single_group(group, constraints)
            else:
                res = self.plan_joint_group(group, constraints)

            if res is None:
                return None

            for a, path in res.items():
                new_paths[a] = path

        # Ensure all agents got a path
        if any(p is None for p in new_paths):
            return None

        return new_paths

    # Open list helpers
    def push_node(self, node):
        heapq.heappush(
            self.open_list,
            (node["cost"], len(node["collisions"]), self.num_of_generated, node),
        )
        self.num_of_generated += 1

    def pop_node(self):
        _, _, _, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    # MA-CBS main search
    def find_solution(self):
        """
        Run MA-CBS and return paths for all agents.
        """
        self.start_time = timer.time()

        # Root node
        root_constraints = []
        root_meta_groups = self.initial_meta_groups()
        root_paths = self.replan_all_groups(root_constraints, root_meta_groups)
        if root_paths is None:
            raise RuntimeError("Root infeasible")

        root_collisions = detect_collisions(root_paths)
        root = {
            "constraints": root_constraints,
            "meta_groups": root_meta_groups,
            "paths": root_paths,
            "collisions": root_collisions,
            "cost": get_sum_of_cost(root_paths),
            "conflict_counts": defaultdict(int),
        }

        self.open_list = []
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.push_node(root)

        while self.open_list:
            node = self.pop_node()

            # Goal test: no collisions
            if len(node["collisions"]) == 0:
                self.print_results(node)
                return node["paths"]

            # Take the first collision
            collision = node["collisions"][0]
            a1 = collision["a1"]
            a2 = collision["a2"]
            pair_key = frozenset({a1, a2})
            node["conflict_counts"][pair_key] += 1

            # MERGE or SPLIT
            if node["conflict_counts"][pair_key] >= self.merge_threshold:
                # ---- MERGE path ----
                new_groups = self.merge_groups(node["meta_groups"], a1, a2)
                child_constraints = list(node["constraints"])
                child_paths = self.replan_all_groups(child_constraints, new_groups)
                if child_paths is None:
                    continue

                child = {
                    "constraints": child_constraints,
                    "meta_groups": new_groups,
                    "paths": child_paths,
                    "collisions": detect_collisions(child_paths),
                    "cost": get_sum_of_cost(child_paths),
                    # keep as defaultdict(int) so new pairs start at 0
                    "conflict_counts": defaultdict(int, node["conflict_counts"]),
                }
                self.push_node(child)

            else:
                # SPLIT (Standard CBS)
                new_constraints = standard_splitting(collision)

                for c in new_constraints:
                    child_constraints = list(node["constraints"]) + [c]
                    child_meta_groups = list(node["meta_groups"])
                    child_paths = self.replan_all_groups(
                        child_constraints, child_meta_groups
                    )
                    if child_paths is None:
                        continue

                    child = {
                        "constraints": child_constraints,
                        "meta_groups": child_meta_groups,
                        "paths": child_paths,
                        "collisions": detect_collisions(child_paths),
                        "cost": get_sum_of_cost(child_paths),
                        # keep as defaultdict(int) here as well
                        "conflict_counts": defaultdict(int, node["conflict_counts"]),
                    }
                    self.push_node(child)

        raise RuntimeError("No solution found")

    def print_results(self, node):
        print("\nFound a solution!\n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.4f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node["paths"])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))


if __name__ == "__main__":
    # Tiny sanity test: 3x3 empty map, 2 agents swapping corners
    my_map = [
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    starts = [(0, 0), (2, 2)]
    goals = [(2, 2), (0, 0)]

    solver = MACBS(my_map, starts, goals, merge_threshold=1)
    paths = solver.find_solution()
    print("Solution paths:")
    for i, p in enumerate(paths):
        print(f"Agent {i}:", p)
