import heapq
import time as timer
from collections import defaultdict
#python run_experiments.py --instance instances/instanceExample28.txt --solver MACBS --macbs_low_level nested
#python run_experiments.py --instance instances/instanceExample28.txt --solver MACBS --macbs_low_level nested

from single_agent_planner import (
    a_star,
    compute_heuristics,
    get_sum_of_cost,
    get_location,
)
from JointSolver import JointSolver
from NestedCBS import NestedCBSSolver

# Collision detection helpers
def detect_collision(path1, path2):
    """
    Return the first collision between two paths, or None.
    Collision format: {'loc': [...], 'timestep': t}
      - Vertex collision: loc = [cell]
      - Edge collision:   loc = [cell_from, cell_to]
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
    Return list of first collisions between ALL pairs of paths.
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
    Returned constraints use *agent indices in the current context*.
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

# MA-CBS
class MACBS:
    """
    Meta-Agent Conflict-Based Search (MA-CBS).

    - Starts with each agent in its own meta-group.
    - When a pair of agents collide too many times (>= merge_threshold),
      their groups are merged and replanned jointly (either Joint A* or
      a nested CBS).
    """

    def __init__(
        self,
        my_map,
        starts,
        goals,
        merge_threshold=2,
        low_level_mode="joint",   # "joint" or "nested"
    ):
        """
        my_map          : 2D grid (True = obstacle, False = free)
        starts/goals    : list of (row, col)
        merge_threshold : # of conflicts before merging a pair of agents
        low_level_mode  : "joint" or "nested"
        """
        assert len(starts) == len(goals)
        self.my_map = my_map
        self.starts = list(starts)
        self.goals = list(goals)
        self.num_agents = len(starts)
        self.merge_threshold = merge_threshold
        self.low_level_mode = low_level_mode

        # high-level statistics
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.open_list = []
        self.start_time = None

        # low-level statisics
        self.ll_total_expanded = 0      # Sum of the low level expanded states
        self.ll_max_peak_open = 0       # max size of open list 
        self.ll_call_count = 0          # how many meta agents low level calls

        # Low-level heuristics for single-agent A*
        # heuristics[i] is a dict for agent i's goal
        self.heuristics = [
            compute_heuristics(my_map, g) for g in self.goals
        ]

        # For counting conflicts between pairs of agents
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
            # Defensive: wrap heuristic dict so missing keys just get 0
            h_vals = defaultdict(int, self.heuristics[a])
            path = a_star(
                self.my_map,
                self.starts[a],
                self.goals[a],
                h_vals,
                a,              # global agent id
                constraints,
            )
            if path is None:
                return None
            group_paths[a] = path
        return group_paths

    def plan_joint_group(self, group, constraints):
        """
        Plan for a group using Joint A* (JointSolver).
        Returns dict {agent_id: path} or None.
        """
        agents = sorted(list(group))
        starts = [self.starts[a] for a in agents]
        goals = [self.goals[a] for a in agents]

        js = JointSolver(self.my_map, starts, goals, agents, constraints)
        joint_paths = js.find_solution()
        if joint_paths is None:
            return None

        self.ll_total_expanded += js.num_expanded
        self.ll_max_peak_open = max(self.ll_max_peak_open, js.peak_open)
        self.ll_call_count += 1

        group_paths = {}
        for i, a in enumerate(agents):
            group_paths[a] = joint_paths[i]
        return group_paths

    def plan_nested_group(self, group, constraints):
        """
        Plan for a group using nested CBS (NestedCBSSolver).
        Returns dict {agent_id: path} or None.
        """
        agents = sorted(list(group))
        starts = [self.starts[a] for a in agents]
        goals = [self.goals[a] for a in agents]

        solver = NestedCBSSolver(
            self.my_map,
            starts,
            goals,
            agents,       # global IDs
            constraints,
        )
        group_paths = solver.find_solution()
        if group_paths is None:
            return None
        
        self.ll_total_expanded += solver.num_expanded
        self.ll_max_peak_open = max(self.ll_max_peak_open, solver.peak_open)
        self.ll_call_count += 1

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
                if self.low_level_mode == "joint":
                    res = self.plan_joint_group(group, constraints)
                elif self.low_level_mode == "nested":
                    res = self.plan_nested_group(group, constraints)
                else:
                    raise ValueError(f"Unknown low-level mode: {self.low_level_mode}")

            if res is None:
                return None

            for a, path in res.items():
                new_paths[a] = path

        # Ensure all agents got a path
        if any(p is None for p in new_paths):
            return None

        return new_paths
    
    def replan_meta_group(self, constraints, meta_groups, parent_paths, agent):
        """
        Replan only the meta-group that contains 'agent', keeping all other groups paths
        """
        new_paths = list(parent_paths)
        g_idx = self.find_group_index(meta_groups, agent)
        group = meta_groups[g_idx]

        if len(group) == 1:
            res = self.plan_single_group(group, constraints)
        else:
            if self.low_level_mode == "joint":
                res = self.plan_joint_group(group, constraints)
            elif self.low_level_mode == "nested":
                res = self.plan_nested_group(group, constraints)
            else:
                raise ValueError(f"unknown low-level mode: {self.low_level_mode}")
        
        if res is None:
            return None

        for a, path in res.items():
            new_paths[a] = path

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

        # Main loop, lines 5-23
        while self.open_list:
            node = self.pop_node()

            # Goal test: no collisions, lines 6-9
            if len(node["collisions"]) == 0:
                self.print_results(node)
                return node["paths"]

            # Take the first collision, line 10
            collision = node["collisions"][0]
            a1 = collision["a1"]
            a2 = collision["a2"]
            pair_key = frozenset({a1, a2})
            node["conflict_counts"][pair_key] += 1

            # MERGE or SPLIT
            if node["conflict_counts"][pair_key] >= self.merge_threshold:
                # MERGE path lines 1-16
                new_groups = self.merge_groups(node["meta_groups"], a1, a2)
                child_constraints = list(node["constraints"])
                child_meta_groups = new_groups
                parent_paths = node["paths"]

                # Replan the merged meta group
                merged_paths = self.replan_meta_group(child_constraints, child_meta_groups, parent_paths, a1)
                if merged_paths is None:
                    continue

                child_paths = merged_paths
                #child_paths = self.replan_all_groups(child_constraints, new_groups)
                # if child_paths is None:
                #     continue

                child = {
                    "constraints": child_constraints,
                    "meta_groups": new_groups,
                    "paths": child_paths,
                    "collisions": detect_collisions(child_paths),
                    "cost": get_sum_of_cost(child_paths),
                    "conflict_counts": defaultdict(int, node["conflict_counts"]),
                }
                self.push_node(child)

            else:
                # SPLIT path (Standard CBS), lines 17-23
                new_constraints = standard_splitting(collision)

                for c in new_constraints:
                    affected_agent = c["agent"]
                    child_constraints = list(node["constraints"])
                    # c["agent"] is *global* agent index in MA-CBS
                    child_constraints.append(
                        {
                            "agent": c["agent"],
                            "loc": c["loc"],
                            "timestep": c["timestep"],
                        }
                    )
                    child_meta_groups = list(node["meta_groups"])
                    parent_paths = node["paths"]
                    child_paths = self.replan_meta_group(child_constraints, child_meta_groups, parent_paths, affected_agent)
                    # child_paths = self.replan_all_groups(
                    #     child_constraints, child_meta_groups
                    # )
                    if child_paths is None:
                        continue

                    child = {
                        "constraints": child_constraints,
                        "meta_groups": child_meta_groups,
                        "paths": child_paths,
                        "collisions": detect_collisions(child_paths),
                        "cost": get_sum_of_cost(child_paths),
                        "conflict_counts": defaultdict(int, node["conflict_counts"]),
                    }
                    self.push_node(child)

        raise RuntimeError("No solution found")

    def print_results(self, node):
        print("\nFound a solution with MA-CBS!\n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.4f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node["paths"])))
        print("Expanded HL nodes:  {}".format(self.num_of_expanded))
        print("Generated HL nodes: {}".format(self.num_of_generated))

        if self.ll_call_count > 0:
            avg_ll_expanded = self.ll_total_expanded / float(self.ll_call_count)
        else:
            avg_ll_expanded = 0.0
        
        print("Low-level calls: {}".format(self.ll_call_count))
        print("Low-level total expansions:  {}".format(self.ll_total_expanded))
        print("Low-level avg expansions/call:   {:.2f}".format(avg_ll_expanded))
        print("Low-level peak open size:    {}".format(self.ll_max_peak_open))

if __name__ == "__main__":
    # Tiny sanity test: 3x3 empty map, 2 agents swapping corners
    my_map = [
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    starts = [(0, 0), (2, 2)]
    goals = [(2, 2), (0, 0)]

    solver = MACBS(my_map, starts, goals, merge_threshold=1, low_level_mode="joint")
    paths = solver.find_solution()
    print("Solution paths:")
    for i, p in enumerate(paths):
        print(f"Agent {i}:", p)
