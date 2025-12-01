import heapq
import itertools
import time as timer
from single_agent_planner import compute_heuristics, get_sum_of_cost, move


class JointSolver(object):
    """
    Joint A* planner that treats all agents as a single meta agent and searches
    the combined configuration space.

    Cost function: sum of costs. Each timestep costs 1 for every agent
    that is not yet at its goal.
    """

    def __init__(self, my_map, starts, goals):
        """
        my_map : 2D list (True = obstacle, False = free)
        starts : list of start locations [(r0,c0), (r1,c1), ...]
        goals  : list of goal locations  [(r0,c0), (r1,c1), ...]
        """
        assert len(starts) == len(goals)
        self.my_map = my_map
        self.starts = tuple(starts)
        self.goals = tuple(goals)
        self.num_of_agents = len(goals)
        self.CPU_time = 0.0
        self.num_expanded = 0
        self.peak_open = 0

        # Pre-compute heuristics for every agent
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

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

    def get_neighbors(self, curr_joint):
        """
        Generates all valid joint moves from curr_joint.
        Each agent can move N/E/S/W or wait in place.
        """
        possible_moves = []

        # Get valid individual moves for each of the agents
        for i in range(self.num_of_agents):
            agent_moves = []
            curr_loc = curr_joint[i]

            # Directions: 0=up, 1=right, 2=down, 3=left
            for direction in range(4):
                next_loc = move(curr_loc, direction)
                # Check map bounds and obstacles
                if self.is_valid_cell(next_loc):
                    agent_moves.append(next_loc)

            # Explicitly add wait
            agent_moves.append(curr_loc)

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


    def find_solution(self):
        start_time = timer.time()

        # Format start and goal as tuples of locations
        start_state = self.starts
        goal_state = self.goals

        # Priority queue of (f-cost, g-cost, joint-state, path-history)
        # Path-history is a list of joint states
        initial_h = self.get_joint_heuristic(start_state)
        initial_g = 0
        initial_f = initial_g + initial_h

        open_list = [(initial_f, initial_g, start_state, [start_state])]
        self.peak_open = max(self.peak_open, len(open_list))

        closed_list = set()
        closed_list.add(start_state)

        self.num_expanded = 0

        final_joint_path = None

        while open_list:
            self.peak_open = max(self.peak_open, len(open_list))
            _, g, curr_joint, path = heapq.heappop(open_list)
            self.num_expanded += 1

            # Goal test
            if curr_joint == goal_state:
                final_joint_path = path
                break

            neighbors = self.get_neighbors(curr_joint)

            for next_joint in neighbors:
                if next_joint in closed_list:
                    continue

                closed_list.add(next_joint)

                # Cost update: sum of costs
                step_cost = self.step_cost(next_joint)
                new_g = g + step_cost
                new_h = self.get_joint_heuristic(next_joint)
                new_f = new_g + new_h

                new_path = list(path)
                new_path.append(next_joint)
                heapq.heappush(open_list, (new_f, new_g, next_joint, new_path))

        self.CPU_time = timer.time() - start_time

        if final_joint_path is None:
            raise BaseException('No solutions')

        # Transform joint path to individual paths for return format
        result = []
        for i in range(self.num_of_agents):
            agent_path = []
            for joint_state in final_joint_path:
                agent_path.append(joint_state[i])
            result.append(agent_path)

        print("\nFound a solution!")
        print("CPU time (s):     {:6f}".format(self.CPU_time))
        print("Sum of costs:     {}".format(get_sum_of_cost(result)))

        return result

# Test
if __name__ == "__main__":
    my_map = [
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    starts = [(0, 0), (2, 2)]
    goals = [(2, 2), (0, 0)]
    solver = JointSolver(my_map, starts, goals)
    paths = solver.find_solution()
    for i, p in enumerate(paths):
        print(f"Agent {i} path:", p)
