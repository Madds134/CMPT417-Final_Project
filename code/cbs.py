import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    first_goal_time = min(len(path1), len(path2))-1
    last_goal_time = max(len(path1), len(path2))-1
    if len(path1) > len(path2):
        longest_path = path1
        shortest_path = path2
    else:
        longest_path = path2
        shortest_path = path1
    for t in range(0, first_goal_time):
        if path1[t] == path2[t]:
            collision = [path1[t]]
            return (collision, t)
    for t in range(first_goal_time, last_goal_time):
        if longest_path[t] == shortest_path[first_goal_time]:
            collision = [longest_path[t]]
            return (collision, t)
    tmp_path1 = path1.copy()
    tmp_path2 = path2.copy()
    time = 1
    while len(tmp_path1) > 1 and len(tmp_path2) > 1:
        # print((tmp_path1, tmp_path2))
        if tmp_path2[1] == tmp_path1[0] and tmp_path1[1] == tmp_path2[0]:
            collision = [path2[time], path1[time]]
            return (collision, time)
        time = time + 1
        tmp_path1.pop(0)
        tmp_path2.pop(0)
    return None

def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    collisions = []
    num_of_agents = len(paths)
    for agent1 in range(0, num_of_agents):
        for agent2 in range(0, num_of_agents):
            if agent1 == agent2:
                continue
                break
            collision = detect_collision(paths[agent1], paths[agent2])
            if collision != None:
                collisions.append({'a1':agent1, 'a2':agent2, 'loc':collision[0], 'timestep':collision[1]})
    return collisions
    pass


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    agent1 = collision['a1']
    agent2 = collision['a2']
    time = collision['timestep']
    # print("col: ", collision)
    # print("collision loc:", type(collision['loc'][0]))
    # if type(collision['loc'][0]) == type(1):
    #     return None
    if len(collision['loc']) == 1:
        location = collision['loc']
        return [{'agent':agent1, 'loc': location, 'timestep':time, 'positive':False},{'agent':agent2, 'loc': location, 'timestep':time, 'positive':False}]
    else:
        location_agent1 = collision['loc']
        location_agent2 = [collision['loc'][1], collision['loc'][0]]
        return [{'agent':agent1, 'loc': (location_agent1), 'timestep':time, 'positive':False},{'agent':agent2, 'loc': location_agent2, 'timestep':time, 'positive':False}]
    pass


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly
    agent1 = collision['a1']
    agent2 = collision['a2']
    time = collision['timestep']
    # print("col: ", collision)
    # print("collision loc:", type(collision['loc'][0]))
    # if type(collision['loc'][0]) == type(1):
    #     return None
    if len(collision['loc']) == 1:
        location = collision['loc']
        if random.randint(0,1) == 0:
            return [{'agent':agent1, 'loc': location, 'timestep':time, 'positive':True},{'agent':agent2, 'loc': location, 'timestep':time, 'positive':False}]
        else:
            return [{'agent':agent1, 'loc': location, 'timestep':time, 'positive':False},{'agent':agent2, 'loc': location, 'timestep':time, 'positive':True}]
    else:
        location_agent1 = collision['loc']
        location_agent2 = [collision['loc'][1], collision['loc'][0]]
        if random.randint(0,1) == 0:
            return [{'agent':agent1, 'loc': (location_agent1), 'timestep':time, 'positive':True},{'agent':agent2, 'loc': location_agent2, 'timestep':time, 'positive':False}]
        else:
            return [{'agent':agent1, 'loc': (location_agent1), 'timestep':time, 'positive':False},{'agent':agent2, 'loc': location_agent2, 'timestep':time, 'positive':True}]
    pass


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        print(root['collisions'])

        # Task 3.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
        while len(self.open_list) > 0:
            # debug exit
            # print(len(self.open_list))
            # if len(self.open_list) > 10:
            #     return None
            P = self.pop_node()
            if len(P['collisions']) == 0:
                return P['paths']
            collision = P['collisions'][0]
            print("collision:", collision)
            if disjoint:
                constraints = disjoint_splitting(collision)
            else:
                constraints = standard_splitting(collision)
            for constraint in constraints:
                Q = {'cost': 0,
                     'constraints': [],
                     'paths': [],
                     'collisions': []}
                tmp_constraints = P['constraints'].copy()
                print("next constraint:",constraint)
                tmp_constraints.append(constraint)
                Q['constraints'] = tmp_constraints
                Q['paths'] = P['paths'].copy()
                ai = constraint['agent']
                print("constraints:", Q['constraints'])
                path = a_star(self.my_map, self.starts[ai], self.goals[ai], self.heuristics[ai], ai, Q['constraints'])
                # print("ai =", ai)
                if path is None:
                    continue
                    # print("No path found")
                print("new path =", path)
                if len(path) != 0:
                    # something
                    Q['paths'][ai] = path
                    Q['collisions'] = detect_collisions(Q['paths'])
                    Q['cost'] = get_sum_of_cost(Q['paths'])
                    self.push_node(Q)
                else:
                    continue
        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))

