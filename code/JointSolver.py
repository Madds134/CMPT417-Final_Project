import heapq
import itertools
import time as timer
from single_agent_planner import (
    compute_heuristics,
    get_sum_of_cost,
    move,
    build_constraint_table,
    is_constrained,
)


class JointSolver(object):
    """
    Joint A* planner that treats a subset of agents as a single meta-agent and
    searches the combined configuration space.
    Cost function: sum of costs. Each timestep costs 1 for every agent
    that is not yet at its goal.
    """

    def __init__(self, my_map, starts, goals, agents, constraints):
        """
        my_map      : 2D list (True = obstacle, False = free)
        starts      : list of start locations for THIS meta-agent [(r,c), ...]
        goals       : list of goal locations  for THIS meta-agent [(r,c), ...]
        agents      : list of GLOBAL agent IDs (same order as starts/goals)
        constraints : full CBS constraint list (dicts with 'agent', 'loc', 'timestep', ...)
        """
        assert len(starts) == len(goals) == len(agents)

        self.my_map = my_map
        self.starts = tuple(starts)
        self.goals = tuple(goals)
        self.agents = list(agents)           # global IDs
        self.num_of_agents = len(agents)

        self.CPU_time = 0.0
        self.num_expanded = 0
        self.peak_open = 0

        # Pre-compute heuristics for every local agent index
        self.heuristics = [
            compute_heuristics(my_map, g) for g in self.goals
        ]

        # Per-agent constraint tables
        # Use global agent id when building each table.
        self.constraint_tables = []
        for global_id in self.agents:
            self.constraint_tables.append(build_constraint_table(constraints, global_id))

        # Earliest allowed goal timestep per agent (negative constraints on goal)
        self.earliest_goal_t = []
        for i, table in enumerate(self.constraint_tables):
            goal_loc = self.goals[i]
            eg = 0
            for t, entry in table.items():
                if goal_loc in entry['n_vertex']:
                    eg = max(eg, t + 1)
            self.earliest_goal_t.append(eg)
        # Joint solver wait
        self.min_goal_t = max(self.earliest_goal_t) if self.earliest_goal_t else 0

    def get_joint_heuristic(self, joint_state):
        """
        Sum of individual heuristics for the current configuration.
        joint_state: ((r1,c1), (r2,c2), ...)
        """
        h_val = 0
        for i in range(self.num_of_agents):
            loc = joint_state[i]
            h_val += self.heuristics[i].get(loc, 0)
        return h_val

    def is_valid_cell(self, loc):
        r, c = loc
        return (
            0 <= r < len(self.my_map)
            and 0 <= c < len(self.my_map[0])
            and not self.my_map[r][c]
        )

    def is_valid_joint_move(self, curr_joint, next_joint):
        """
        Checks for collisions in the joint step:
        - vertex collision (same cell)
        - edge collision (swap)
        """
        # Vertex collisions
        if len(set(next_joint)) != len(next_joint):
            return False

        # Edge collisions
        for i in range(self.num_of_agents):
            for j in range(i + 1, self.num_of_agents):
                if next_joint[i] == curr_joint[j] and next_joint[j] == curr_joint[i]:
                    return False

        return True

    # Neighbor generation with constraints
    def get_neighbors(self, curr_joint, t):
        """
        Generates all valid joint moves from curr_joint at time t -> t+1.

        For each agent i:
          - consider moves N/E/S/W + wait
          - filter by map & obstacles
          - filter by that agent's constraint table (vertex/edge)
        """
        possible_moves = []

        for i in range(self.num_of_agents):
            agent_moves = []
            curr_loc = curr_joint[i]
            table = self.constraint_tables[i]

            # Directions: 0=up, 1=right, 2=down, 3=left
            for direction in range(4):
                next_loc = move(curr_loc, direction)
                # Check map bounds and obstacles
                if not self.is_valid_cell(next_loc):
                    continue
                # Check per-agent constraints at t+1
                if is_constrained(curr_loc, next_loc, t + 1, table):
                    continue
                agent_moves.append(next_loc)

            # Explicitly wait
            if not is_constrained(curr_loc, curr_loc, t + 1, table):
                agent_moves.append(curr_loc)

            if not agent_moves:
                # This agent cannot move at all at t+1 under constraints:
                # no joint successor possible from curr_joint
                return []

            possible_moves.append(agent_moves)

        # Cartesian Product to get joint moves
        valid_neighbors = []
        for next_joint in itertools.product(*possible_moves):
            if self.is_valid_joint_move(curr_joint, next_joint):
                valid_neighbors.append(next_joint)

        return valid_neighbors

    def step_cost(self, joint_state):
        """
        Cost contributed by this timestep: 1 for each agent not at goal.
        """
        cost = 0
        for i in range(self.num_of_agents):
            if joint_state[i] != self.goals[i]:
                cost += 1
        return cost

    # A* search in (joint_state, time) space
    def find_solution(self):
        """
        Run joint-space A* to compute collision-free, constraint-respecting
        paths for this meta-agent.

        Returns:
            result: list of paths, one per local agent index (aligned with
                    self.agents / starts / goals), or None if infeasible.
        """
        start_time = timer.time()

        # Format start and goal as tuples of locations
        start_state = self.starts
        goal_state = self.goals
        start_t = 0

        # Check start state against constraints at t=0
        for i, loc in enumerate(start_state):
            table = self.constraint_tables[i]
            if is_constrained(loc, loc, 0, table):
                # No feasible path if start is illegal
                return None

        # Priority queue of (f-cost, g-cost, joint-state, time, path-history)
        initial_h = self.get_joint_heuristic(start_state)
        initial_g = 0
        initial_f = initial_g + initial_h

        open_list = [(initial_f, initial_g, start_state, start_t, [start_state])]
        self.peak_open = max(self.peak_open, len(open_list))

        closed_list = set()
        closed_list.add((start_state, start_t))

        self.num_expanded = 0
        final_joint_path = None

        while open_list:
            self.peak_open = max(self.peak_open, len(open_list))
            _, g, curr_joint, t, path = heapq.heappop(open_list)
            self.num_expanded += 1

            # Goal test: all at goals, past earliest allowed goal time,
            # and no constraint violation at this timestep.
            if curr_joint == goal_state and t >= self.min_goal_t:
                goal_ok = True
                for i, loc in enumerate(curr_joint):
                    table = self.constraint_tables[i]
                    if is_constrained(loc, loc, t, table):
                        goal_ok = False
                        break
                if goal_ok:
                    final_joint_path = path
                    break

            # Expand neighbors at time t -> t+1
            neighbors = self.get_neighbors(curr_joint, t)

            for next_joint in neighbors:
                next_t = t + 1
                state_key = (next_joint, next_t)
                if state_key in closed_list:
                    continue

                closed_list.add(state_key)

                # Cost update: sum of costs
                step_c = self.step_cost(next_joint)
                new_g = g + step_c
                new_h = self.get_joint_heuristic(next_joint)
                new_f = new_g + new_h

                new_path = list(path)
                new_path.append(next_joint)
                heapq.heappush(open_list, (new_f, new_g, next_joint, next_t, new_path))

        self.CPU_time = timer.time() - start_time

        if final_joint_path is None:
            # Infeasible under constraints
            return None

        # Transform joint path to individual paths for return format
        result = []
        for i in range(self.num_of_agents):
            agent_path = []
            for joint_state in final_joint_path:
                agent_path.append(joint_state[i])
            result.append(agent_path)

        # Optional: print stats (you may mute this in experiments)
        print("\nJointSolver solution:")
        print("  Agents:", self.agents)
        print("  CPU time (s):     {:6f}".format(self.CPU_time))
        print("  Sum of costs:     {}".format(get_sum_of_cost(result)))
        print("  Expanded states:  {}".format(self.num_expanded))
        print("  Peak open size:   {}".format(self.peak_open))

        return result


# Local test with no constraints
if __name__ == "__main__":
    my_map = [
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    starts = [(0, 0), (2, 2)]
    goals = [(2, 2), (0, 0)]
    agents = [0, 1]
    constraints = []

    solver = JointSolver(my_map, starts, goals, agents, constraints)
    paths = solver.find_solution()
    if paths is None:
        print("No solution.")
    else:
        for i, p in enumerate(paths):
            print(f"Local agent {i} (global {agents[i]}) path:", p)
