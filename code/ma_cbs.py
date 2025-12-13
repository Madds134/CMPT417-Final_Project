import heapq
import time as timer
from collections import defaultdict

# we used this command to check each instance works.
# python run_experiments.py --instance instances/instanceExample28.txt --solver MACBS --macbs_low_level nested 

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
    # Use max length so agents implicitly wait at goal after path end
    timesteps = max(len(path1), len(path2))

    for t in range(timesteps):
        loc1 = get_location(path1, t)
        loc2 = get_location(path2, t)

        # Vertex collision: same cell at same timestep
        if loc1 == loc2:
            return {"loc": [loc1], "timestep": t}

        # Edge collision: swap between t-1 and t
        if t > 0:
            prev1 = get_location(path1, t - 1)
            prev2 = get_location(path2, t - 1)
            if prev1 == loc2 and prev2 == loc1:
                return {"loc": [prev1, loc1], "timestep": t}

    return None


def detect_collisions(paths):
    """
    Return list of first collisions between all unordered pairs of paths.
    Only the earliest collision per pair is recorded.
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
    Standard CBS splitting.
    Generates two constraints, one per agent involved in the collision.
    """
    a1 = collision["a1"]
    a2 = collision["a2"]
    loc = collision["loc"]
    t = collision["timestep"]

    # Vertex collision: forbid occupying the vertex at time t
    if len(loc) == 1:
        v = loc[0]
        return [
            {"agent": a1, "loc": [v], "timestep": t},
            {"agent": a2, "loc": [v], "timestep": t},
        ]

    # Edge collision: forbid traversing the conflicting edge at time t
    elif len(loc) == 2:
        u, v = loc
        return [
            {"agent": a1, "loc": [u, v], "timestep": t},
            {"agent": a2, "loc": [v, u], "timestep": t},
        ]
    return []


# MA-CBS
class MACBS:
    """
    Meta-Agent Conflict-Based Search (MA-CBS).

    High level resolves conflicts between agents.
    Low level plans either independently, jointly, or via nested CBS
    depending on meta-group size and merge history.
    """

    def __init__(
        self,
        my_map,
        starts,
        goals,
        merge_threshold=2,
        low_level_mode="joint",   # "joint" or "nested"
    ):
        assert len(starts) == len(goals)

        self.my_map = my_map
        self.starts = list(starts)
        self.goals = list(goals)
        self.num_agents = len(starts)
        self.merge_threshold = merge_threshold
        self.low_level_mode = low_level_mode

        # High-level CBS statistics
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.open_list = []
        self.start_time = None

        # Aggregated low-level statistics
        self.ll_total_expanded = 0
        self.ll_max_peak_open = 0
        self.ll_call_count = 0

        # Heuristics for single-agent A* (used when group size == 1)
        self.heuristics = [
            compute_heuristics(my_map, g) for g in self.goals
        ]

        # Reserved for possible duplicate detection / analysis
        self.hl_closed = set()

    # Meta-group utilities
    def initial_meta_groups(self):
        # Start with each agent as its own meta-group
        return [frozenset([i]) for i in range(self.num_agents)]

    def find_group_index(self, meta_groups, agent):
        # Find which meta-group an agent belongs to
        for idx, g in enumerate(meta_groups):
            if agent in g:
                return idx
        raise ValueError(f"Agent {agent} not in any meta-group")

    def merge_groups(self, meta_groups, a1, a2):
        """
        Merge the two meta-groups containing agents a1 and a2.
        """
        i = self.find_group_index(meta_groups, a1)
        j = self.find_group_index(meta_groups, a2)
        if i == j:
            return meta_groups

        merged = meta_groups[i].union(meta_groups[j])

        new_groups = []
        for idx, g in enumerate(meta_groups):
            if idx != i and idx != j:
                new_groups.append(g)
        new_groups.append(merged)
        return new_groups

    # Low-level planning
    def plan_single_group(self, group, constraints):
        """
        Independent A* planning for a single-agent meta-group.
        """
        group_paths = {}
        for a in group:
            h_vals = defaultdict(int, self.heuristics[a])
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
        Joint A* planning over the Cartesian product of agent states.
        """
        agents = sorted(group)
        starts = [self.starts[a] for a in agents]
        goals = [self.goals[a] for a in agents]

        js = JointSolver(self.my_map, starts, goals, agents, constraints)
        joint_paths = js.find_solution()
        if joint_paths is None:
            return None

        # Track joint solver effort
        self.ll_total_expanded += js.num_expanded
        self.ll_max_peak_open = max(self.ll_max_peak_open, js.peak_open)
        self.ll_call_count += 1

        return {a: joint_paths[i] for i, a in enumerate(agents)}

    def plan_nested_group(self, group, constraints):
        """
        Nested CBS planning: CBS inside MA-CBS for this meta-group.
        """
        agents = sorted(group)
        starts = [self.starts[a] for a in agents]
        goals = [self.goals[a] for a in agents]

        solver = NestedCBSSolver(self.my_map, starts, goals, agents, constraints)
        group_paths = solver.find_solution()
        if group_paths is None:
            return None

        self.ll_total_expanded += solver.num_expanded
        self.ll_max_peak_open = max(self.ll_max_peak_open, solver.peak_open)
        self.ll_call_count += 1

        return group_paths

    def replan_all_groups(self, constraints, meta_groups):
        """
        Replan paths for all meta-groups from scratch.
        Used at the root and as a fallback.
        """
        new_paths = [None] * self.num_agents

        for group in meta_groups:
            if len(group) == 1:
                res = self.plan_single_group(group, constraints)
            else:
                res = (
                    self.plan_joint_group(group, constraints)
                    if self.low_level_mode == "joint"
                    else self.plan_nested_group(group, constraints)
                )

            if res is None:
                return None

            for a, path in res.items():
                new_paths[a] = path

        if any(p is None for p in new_paths):
            return None

        return new_paths

    def replan_meta_group(self, constraints, meta_groups, parent_paths, agent):
        """
        Replan only the meta-group containing 'agent'.
        Other agents reuse their parent paths unchanged.
        """
        new_paths = list(parent_paths)
        group = meta_groups[self.find_group_index(meta_groups, agent)]

        if len(group) == 1:
            res = self.plan_single_group(group, constraints)
        else:
            res = (
                self.plan_joint_group(group, constraints)
                if self.low_level_mode == "joint"
                else self.plan_nested_group(group, constraints)
            )

        if res is None:
            return None

        for a, path in res.items():
            new_paths[a] = path

        return new_paths

    # Open list helpers
    def push_node(self, node):
        # Order by cost, then number of collisions, then insertion order
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
        self.start_time = timer.time()

        # Root node: no constraints, singleton meta-groups
        root_constraints = []
        root_meta_groups = self.initial_meta_groups()
        root_paths = self.replan_all_groups(root_constraints, root_meta_groups)
        if root_paths is None:
            raise RuntimeError("Root infeasible")

        root = {
            "constraints": root_constraints,
            "meta_groups": root_meta_groups,
            "paths": root_paths,
            "collisions": detect_collisions(root_paths),
            "cost": get_sum_of_cost(root_paths),
            "conflict_counts": defaultdict(int),
        }

        self.open_list = []
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.push_node(root)

        while self.open_list:
            node = self.pop_node()

            # Goal: no remaining collisions
            if len(node["collisions"]) == 0:
                self.print_results(node)
                return node["paths"]

            collision = node["collisions"][0]
            a1, a2 = collision["a1"], collision["a2"]
            pair_key = frozenset({a1, a2})
            node["conflict_counts"][pair_key] += 1

            # MERGE branch
            if node["conflict_counts"][pair_key] >= self.merge_threshold:
                new_groups = self.merge_groups(node["meta_groups"], a1, a2)
                child_constraints = list(node["constraints"])
                parent_paths = node["paths"]

                child_paths = self.replan_meta_group(
                    child_constraints, new_groups, parent_paths, a1
                )
                if child_paths is None:
                    continue

                child = {
                    "constraints": child_constraints,
                    "meta_groups": new_groups,
                    "paths": child_paths,
                    "collisions": detect_collisions(child_paths),
                    "cost": get_sum_of_cost(child_paths),
                    "conflict_counts": defaultdict(int, node["conflict_counts"]),
                }
                self.push_node(child)

            # SPLIT branch
            else:
                new_constraints = standard_splitting(collision)

                for c in new_constraints:
                    affected_agent = c["agent"]
                    child_constraints = list(node["constraints"])
                    child_constraints.append(c)

                    parent_paths = node["paths"]
                    child_paths = self.replan_meta_group(
                        child_constraints,
                        node["meta_groups"],
                        parent_paths,
                        affected_agent,
                    )
                    if child_paths is None:
                        continue

                    child = {
                        "constraints": child_constraints,
                        "meta_groups": node["meta_groups"],
                        "paths": child_paths,
                        "collisions": detect_collisions(child_paths),
                        "cost": get_sum_of_cost(child_paths),
                        "conflict_counts": defaultdict(int, node["conflict_counts"]),
                    }
                    self.push_node(child)

        raise RuntimeError("No solution found")

    def print_results(self, node):
        CPU_time = timer.time() - self.start_time
        print("\nFound a solution with MA-CBS!\n")
        print("CPU time (s): |{:.6f}".format(CPU_time))
        print("Sum of costs: |{}".format(get_sum_of_cost(node["paths"])))
        print("Expanded HL nodes: |{}".format(self.num_of_expanded))
        print("Generated HL nodes: |{}".format(self.num_of_generated))

        avg_ll_expanded = (
            self.ll_total_expanded / float(self.ll_call_count)
            if self.ll_call_count > 0 else 0.0
        )

        print("Low-level calls: |{}".format(self.ll_call_count))
        print("Low-level total expansions: |{}".format(self.ll_total_expanded))
        print("Low-level avg expansions/call: |{:.2f}".format(avg_ll_expanded))
        print("Low-level peak open size: |{}".format(self.ll_max_peak_open))


if __name__ == "__main__":
    # Tiny sanity check: 2 agents swapping corners on empty 3x3 grid
    my_map = [
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    starts = [(0, 0), (2, 2)]
    goals = [(2, 2), (0, 0)]

    solver = MACBS(my_map, starts, goals, merge_threshold=1, low_level_mode="joint")
    paths = solver.find_solution()
    for i, p in enumerate(paths):
        print(f"Agent {i}:", p)
