import heapq
from collections import defaultdict

from single_agent_planner import (
    a_star,
    compute_heuristics,
    get_sum_of_cost,
    get_location,
)

# Collision detection helpers (local indices)


def detect_collision(path1, path2):
    # Check up to the longest path length (agents wait at goal if shorter)
    timesteps = max(len(path1), len(path2))

    for t in range(timesteps):
        loc1 = get_location(path1, t)
        loc2 = get_location(path2, t)

        # Vertex collision: same location at same timestep
        if loc1 == loc2:
            return {"loc": [loc1], "timestep": t}

        # Edge collision: agents swap locations between t-1 and t
        if t > 0:
            prev1 = get_location(path1, t - 1)
            prev2 = get_location(path2, t - 1)
            if prev1 == loc2 and prev2 == loc1:
                return {"loc": [prev1, loc1], "timestep": t}

    return None


def detect_collisions(paths):
    collisions = []
    n = len(paths)

    # Check all unordered agent pairs
    for i in range(n):
        for j in range(i + 1, n):
            c = detect_collision(paths[i], paths[j])
            if c is not None:
                collisions.append(
                    {"a1": i, "a2": j, "loc": c["loc"], "timestep": c["timestep"]}
                )
    return collisions


def standard_splitting(collision):
    a1 = collision["a1"]
    a2 = collision["a2"]
    loc = collision["loc"]
    t = collision["timestep"]

    # Vertex collision, forbiding both agents from occupying the vertex at t
    if len(loc) == 1:
        v = loc[0]
        return [
            {"agent": a1, "loc": [v], "timestep": t},
            {"agent": a2, "loc": [v], "timestep": t},
        ]

    # Edge collision, forbiding each agent from traversing the conflicting edge
    elif len(loc) == 2:
        u, v = loc
        return [
            {"agent": a1, "loc": [u, v], "timestep": t},
            {"agent": a2, "loc": [v, u], "timestep": t},
        ]
    return []


# Nested CBS (used as low-level solver inside MA-CBS)
class NestedCBSSolver:
    """
    Standard CBS over a subset of agents (meta-agent),
    respecting parent constraints from MA-CBS.

    Local agents are indexed 0..k-1.
    self.agents maps local index -> global agent ID.
    """

    def __init__(self, my_map, starts, goals, agents, parent_constraints):
        """
        my_map             : grid
        starts, goals      : starts/goals for this group (local order)
        agents             : global agent IDs
        parent_constraints : constraints inherited from MA-CBS
        """
        assert len(starts) == len(goals) == len(agents)

        self.my_map = my_map
        self.local_starts = list(starts)
        self.local_goals = list(goals)
        self.agents = list(agents)
        self.num_local = len(agents)

        # Precompute heuristics for each local agent goal
        self.heuristics = [
            compute_heuristics(my_map, g) for g in self.local_goals
        ]

        # Constraints passed from the parent MA-CBS node
        self.base_constraints = list(parent_constraints)

        self.open_list = []
        self.num_generated = 0
        self.num_expanded = 0
        self.peak_open = 0

    #  low-level planning for one local agent 
    def plan_agent(self, local_idx, constraints):
        """
        Run A* for a single local agent.
        Constraints are global; a_star filters by agent ID.
        """
        global_id = self.agents[local_idx]
        h_vals = defaultdict(int, self.heuristics[local_idx])

        path = a_star(
            self.my_map,
            self.local_starts[local_idx],
            self.local_goals[local_idx],
            h_vals,
            global_id,
            constraints,
        )
        return path

    def replan_all(self, constraints):
        # Replan all local agents under the same constraint set
        paths = []
        for i in range(self.num_local):
            p = self.plan_agent(i, constraints)
            if p is None:
                return None
            paths.append(p)
        return paths

    #  open list helpers 
    def push_node(self, node):
        # Order by cost, then number of collisions, then FIFO
        heapq.heappush(
            self.open_list,
            (node["cost"], len(node["collisions"]), self.num_generated, node),
        )
        self.num_generated += 1
        self.peak_open = max(self.peak_open, len(self.open_list))

    def pop_node(self):
        _, _, _, node = heapq.heappop(self.open_list)
        self.num_expanded += 1
        return node

    #  main CBS over this group 
    def find_solution(self):
        """
        Run CBS on this meta-agent group.
        Returns a dict {global_agent_id: path} or None.
        """
        # Root node uses only parent constraints
        root_constraints = list(self.base_constraints)
        root_paths = self.replan_all(root_constraints)
        if root_paths is None:
            return None

        root = {
            "constraints": root_constraints,
            "paths": root_paths,
            "collisions": detect_collisions(root_paths),
            "cost": get_sum_of_cost(root_paths),
        }

        self.open_list = []
        self.num_generated = 0
        self.num_expanded = 0
        self.push_node(root)

        while self.open_list:
            node = self.pop_node()

            # Collision-free solution
            if len(node["collisions"]) == 0:
                result = {}
                for i, path in enumerate(node["paths"]):
                    result[self.agents[i]] = path
                return result

            collision = node["collisions"][0]

            # Standard CBS split
            new_constraints = standard_splitting(collision)

            for c in new_constraints:
                local_idx = c["agent"]
                global_id = self.agents[local_idx]

                child_constraints = list(node["constraints"])
                child_constraints.append(
                    {
                        "agent": global_id,
                        "loc": c["loc"],
                        "timestep": c["timestep"],
                    }
                )

                # Replan entire group under new constraint
                child_paths = self.replan_all(child_constraints)
                if child_paths is None:
                    continue

                child = {
                    "constraints": child_constraints,
                    "paths": child_paths,
                    "collisions": detect_collisions(child_paths),
                    "cost": get_sum_of_cost(child_paths),
                }
                self.push_node(child)

        return None


if __name__ == "__main__":
    # Tiny sanity check: 2 agents swapping corners on empty 3x3 grid
    my_map = [
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    starts = [(0, 0), (2, 2)]
    goals = [(2, 2), (0, 0)]
    agents = [0, 1]
    parent_constraints = []

    solver = NestedCBSSolver(my_map, starts, goals, agents, parent_constraints)
    sol = solver.find_solution()
    print("Nested CBS solution:", sol)
