import numpy as np
import time
import bisect
from dataclasses import dataclass
from typing import Self
import heapq
from collections import defaultdict
import random


class CPLP:

    def __init__(self, graph, agents, agent_start, V_vvc, V_vec, E_vec, E_eec, dist):

        # Params
        self.min_SIPP_horizon = 1
        self.CCBS_time_limit = 0.025

        # Graph
        self.graph = graph
        self.dist = dist

        # Agents
        self.agents = agents
        self.agent_start = agent_start
        self.plans = {agent: [] for agent in agents}
        self.agent_last_point = {agent: (agent_start[agent], 0) for agent in agents}
        self.agent_to_task = {agent: None for agent in agents}
        self.task_to_agent = {}

        # CTC and unsafe intervals
        # self.V_vvc = V_vvc
        self.V_vec = V_vec
        self.E_vec = E_vec
        self.E_eec = E_eec
        self.unsafe_intervals = dict()
        for v in self.graph.nodes:
            self.unsafe_intervals[v] = []
        for e in self.graph.edges:
            self.unsafe_intervals[e] = []

        # tasks
        self.task_set = set()
        self.task_priorities = dict()
        self.hpt = None
        self.hpa = None
        self.hpt_end_time = 0
        self.vertex_utilisation = dict()  # stores time intervals when agents were at vertices (for task completion)

        # planner log
        self.log = {'Plan': dict(), 'Task completion': dict()}

    def plan(self, curr_time, new_tasks):
        """Plans agent actions based on current time and new tasks.

        This method updates agent plans by:
        1. Adding wait actions for idle agents
        2. Updating the task set with new tasks
        3. Checking for completed tasks
        4. Assigning agents to tasks based on priority
        5. Planning paths for agents using CCBS for high-priority tasks and SIPP for others

        Args:
            curr_time: Current simulation time.
            new_tasks: List of new tasks to be added to the task set.

        Returns:
            The next time when re-planning should occur, or None if no tasks remain.

        Note:
            This method logs planning information and task completions for analysis.
        """
        self.log['Plan'][curr_time] = {'CCBS': None, 'SIPP': dict()}

        # Add wait-actions for idle agents
        for agent in self.agents:
            if self.agent_last_point[agent][1] < curr_time:

                # if last action was a wait action, then extend it
                if self.plans[agent] and len(self.plans[agent][-1]) == 3:
                    wait_action = (self.agent_last_point[agent][0], self.plans[agent][-1][1], curr_time)
                    self.plans[agent].pop()
                else:
                    wait_action = (self.agent_last_point[agent][0], self.agent_last_point[agent][1], curr_time)

                self.update_with_plans({agent: [wait_action]})

        # Add new task to system
        self.task_set.update(new_tasks)
        self.update_task_priorities(curr_time)

        # Remove completed tasks
        self.update_completed_tasks(curr_time)

        # Return if there are no tasks in the system
        if not self.task_set:
            return None

        # Do hpa/hpt path planning (CCBS)
        found_new_path = None
        if self.hpt_end_time <= curr_time:

            # search for a path from an agent to a task 
            # looks for the 1st nearest agent to each task in order of priority, then the 2nd nearest agent, etc. 
            agent_task_arrival_times = dict()
            tasks_by_priority = sorted(self.task_priorities.keys(), key=lambda item: self.task_priorities[item])
            found_path = False
            for i_agent in range(0, 5):
                for task in tasks_by_priority:

                    # calculate lower-bound arrival time for each agent to task (if not done already)
                    if task not in agent_task_arrival_times:
                        agent_task_arrival_times[task] = sorted(list(self.agents), key=lambda
                            agent: self.agent_last_point[agent][1] + self.dist[self.agent_last_point[agent][0]][task[0]])

                    # agent to check
                    agent = agent_task_arrival_times[task][i_agent]

                    # try to find a path for agent to task
                    self.agent_to_task[agent] = task
                    self.task_to_agent[task] = agent
                    
                    # search for plan
                    plans = self.CCBS_modified(agent)

                    if plans is not None:
                        self.update_with_plans(plans)
                        self.hpa = agent
                        self.hpt = task
                        self.hpt_end_time = plans[self.hpa][-1][1] + self.edge_weight(plans[self.hpa][-1][0])
                        found_path = True
                        found_new_path = True
                        self.log['Plan'][curr_time]['CCBS'] = plans
                        self.log['Plan'][curr_time]['HPA'] = self.hpa
                        break

                    else:
                        self.agent_to_task[agent] = None
                        del self.task_to_agent[task]
                        found_new_path = False if found_new_path is None else found_new_path

                if found_path is True:
                    break

            if self.hpt_end_time <=  curr_time:
                print(f' ------------ Failed to find HPT/HPA path. Tasks {self.task_set}')

        # Update assignments
        self.update_assigned_agent_tasks()

        # Do short planning for remaining agents (SIPP)
        remaining_tasks = self.task_set - {self.hpt} - set(self.task_to_agent)
        planned_completed_tasks = set() if self.hpt is None else {self.hpt}
        def plan_short_paths():

            agents_by_task_priority = sorted([(agent, task) for task, agent in self.task_to_agent.items()
                                              if agent != self.hpa and self.agent_last_point[agent][0] != task[0]],
                                             key=lambda item: self.task_priorities[item[1]], reverse=True)
            for agent,_ in agents_by_task_priority:
                while self.agent_last_point[agent][1] <= curr_time + self.min_SIPP_horizon:
                    # and self.agent_last_point[agent][0] != self.agent_to_task[agent][0]:

                    # If agent has reached task, assign a new one
                    if self.agent_last_point[agent][0] == self.agent_to_task[agent][0]:
                        planned_completed_tasks.add(self.agent_to_task[agent])

                        if remaining_tasks:
                            new_task = remaining_tasks.pop()  # random assignment TODO assign nearest
                            self.agent_to_task[agent] = new_task
                            self.task_to_agent[new_task] = agent
                            print('Assigned new task to agent')

                        else:
                            break

                    # get a short path from the agent's current point to a node that is closer to its task and not
                    # occupied by another agent at any time after
                    plan = self.SIPP(
                        start_time=self.agent_last_point[agent][1],
                        start_vertex=self.agent_last_point[agent][0],
                        end_vertex=self.agent_to_task[agent][0],
                        unsafe_intervals=self.unsafe_intervals,
                        quick_return=True
                    )

                    if plan is not None:
                        plan = plan[:-1] # remove last infinite wait action
                        self.update_with_plans({agent: plan})
                        self.log['Plan'][curr_time]['SIPP'].setdefault(agent, []).extend(plan)
                    else:
                        break
        #plan_short_paths()

        # Do random planning if CCBS failed
        def random_short_paths():

            for agent in self.agents:

                random_vertex = random.choice(list(self.graph.nodes))
                while self.agent_last_point[agent][1] <= curr_time + self.min_SIPP_horizon\
                        and self.agent_last_point[agent][0] != random_vertex:

                    # get a short path from the agent's current point to a node that is closer to its task and not
                    # occupied by another agent at any time after
                    plan = self.SIPP(
                        start_time=self.agent_last_point[agent][1],
                        start_vertex=self.agent_last_point[agent][0],
                        end_vertex=random_vertex,
                        unsafe_intervals=self.unsafe_intervals,
                        quick_return=True
                    )

                    if plan is not None:
                        plan = plan[:-1]  # remove last infinite wait action
                        self.update_with_plans({agent: plan})
                        self.log['Plan'][curr_time]['SIPP'].setdefault(agent, []).extend(plan)
                    else:
                        break
        if found_new_path is False:
            random_short_paths()
            print(f' ------------ Randomise paths:')
        else:
            plan_short_paths()

        # Get next re-planning time
        last_planned_times = [self.agent_last_point[agent][1] for agent in self.agents]
        if not self.task_set - planned_completed_tasks:

            # all tasks planned to be completed
            return max(last_planned_times)

        else:
            times_in_future = [time for time in last_planned_times if time > curr_time]

            if times_in_future:

                # agents are still planned to move
                return max(curr_time + self.min_SIPP_horizon, min(times_in_future))

            else:

                # No agents are planned to move anymore, return end of horizon
                return curr_time + self.min_SIPP_horizon

    def CCBS_modified(self, agent):
        """Plans collision-free paths for agents using a modified Continuous-time Conflict-Based Search algorithm.

        This modified CCBS algorithm handles idle agents and removes the "disappear at target"
        assumption present in standard CCBS. It ensures all agents' paths end at vertices where
        no other agent is scheduled to move after.

        Args:
            agent: The primary agent to plan a path for, typically the high-priority agent.

        Returns:
            A dictionary mapping agent IDs to their collision-free plans, where each plan
            is a list of actions (either move or wait actions).

        Raises:
            Exception: If no valid plan is found or if execution times out (10 seconds).

        Note:
            The algorithm builds a constraint tree (CT) where each node contains plans for
            all agents. When a collision is detected, new CT nodes are created with additional
            constraints to resolve the conflict. The algorithm prioritizes the primary agent
            when checking for collisions.
        """

        def detect_collision(node):
            """Detects and prioritizes collisions between agent actions in a plan node.

            This function implements a sophisticated collision detection strategy:
            1. Finite-duration collisions are immediately returned when found
            2. Infinite-wait collisions are stored and only returned if no finite collisions exist
            3. Among multiple infinite-wait collisions, the chronologically earliest is preferred

            The collision detection examines both:
            - Vertex collisions: When an edge traversal conflicts with an agent waiting at a vertex
            - Edge collisions: When two edge traversals conflict with each other

            Args:
                node: A CTNode containing agent plans to check for collisions.

            Returns:
                A tuple describing the collision if one is found, or None if no collision exists.
                The collision format depends on the type:
                - For vertex collision: ((agent1, vertex, t0, t1), (agent2, edge, time))
                  where agent1 is waiting at vertex during (t0, t1), and agent2 traverses edge at time
                - For edge collision: ((agent1, edge1, time1), (agent2, edge2, time2))
                  where agent1 traverses edge1 at time1, and agent2 traverses edge2 at time2

            Note:
                This prioritized approach significantly improves CCBS performance by addressing
                simpler finite-duration conflicts first and handling infinite-wait conflicts in
                chronological order when necessary. This reduces the likelihood of timeout by
                guiding the search toward more efficiently resolvable conflict patterns.
            """

            def intervals_overlap(t1_start, t1_end, t2_start, t2_end, epsilon=1e-6):
                return (t1_end - t2_start > epsilon) and (t2_end - t1_start > epsilon)

            preliminary_collision = None

            # Initialise action lookup table
            action_lookup = defaultdict(list)

            # Check each agent's plan in priority order
            for a in self.agents:
                plan = node.plans[a]

                for action in plan:

                    if len(action) == 2:  # move action
                        e, t = action

                        # collision with agents waiting at vertices
                        for v, (tau0, tau1) in self.V_vec[e].items():
                            unsafe_interval = (t + tau0, t + tau1)

                            for other_a, other_t0, other_t1 in action_lookup[v]:
                                if other_a != a and intervals_overlap(unsafe_interval[0], unsafe_interval[1], other_t0, other_t1):

                                    if other_t1 == np.inf:
                                        if (preliminary_collision is None
                                                or min(other_t0, t) < min(preliminary_collision[0][2], preliminary_collision[1][2])):
                                            preliminary_collision = ((other_a, v, other_t0, other_t1), (a, e, t))
                                    else:
                                        return ((other_a, v, other_t0, other_t1), (a, e, t))

                        # collision with agents traversing edges
                        for e_, (tau0, tau1) in self.E_eec[e].items():
                            unsafe_interval = (t + tau0, t + tau1)

                            for other_a, other_t in action_lookup[e_]:
                                if other_a != a and intervals_overlap(unsafe_interval[0], unsafe_interval[1], other_t, other_t):
                                    return ((a, e, t), (other_a, e_, other_t))

                        # add action to lookup
                        action_lookup[e].append((a, t))

                    elif len(action) == 3:  # wait action
                        v, t0, t1 = action

                        # collision with agents traversing edges
                        for e, (tau0, tau1) in self.E_vec[v].items():
                            unsafe_interval = (t0 + tau0, t1 + tau1)

                            for other_a, other_t in action_lookup[e]:
                                if other_a != a and intervals_overlap(unsafe_interval[0], unsafe_interval[1], other_t, other_t):

                                    if t1 == np.inf:
                                        if (preliminary_collision is None
                                                or min(t0, other_t) < min(preliminary_collision[0][2], preliminary_collision[1][2])):
                                            preliminary_collision = ((a, v, t0, t1), (other_a, e, other_t))
                                    else:
                                        return ((a, v, t0, t1), (other_a, e, other_t))

                        # add action to lookup
                        action_lookup[v].append((a, t0, t1))

            return preliminary_collision

        def collect_plans(node):
            """Extracts and cleans agent plans from a planning node.

            Args:
                node: A planning node containing agent plans.

            Returns:
                A dictionary mapping agent IDs to their plans, with the last infinite
                wait action removed from each plan and empty plans filtered out.
            """
            plans = node.plans

            # remove the last infinite wait action and return the plans
            for a, plan in plans.items():
                plan.pop()

            # remove empty plans
            plans = {a: plan for a, plan in plans.items() if plan}

            return plans

        def modify_unsafe_intervals_with_constraints(constraints):
            """Updates unsafe intervals based on planning constraints.

            Args:
                constraints: List of tuples (location, interval) where each constraint
                             represents a time interval during which a location is unsafe.

            Returns:
                A dictionary of updated unsafe intervals incorporating the new constraints.
            """

            modified_unsafe_intervals = self.unsafe_intervals.copy()
            for x, interval in constraints:
                modified_unsafe_intervals[x] = self.insert_interval(modified_unsafe_intervals[x], interval)
            return modified_unsafe_intervals

        def spawn_node_from_constraints(parent, new_constraints):
            """Creates a new planning node by applying constraints to a parent node.

            Generates a new search node by copying the parent's plans and constraints,
            adding the new constraints for the specified agent, and re-planning that
            agent's path to satisfy all constraints.

            Args:
                parent: Parent planning node containing [cost, plans, constraints].
                new_constraints: List of constraints (agent, location, interval) to add, all for the same agent.

            Returns:
                A new planning node [cost, plans, constraints] if a valid path exists with the new constraints,
                or None if the constraints are inconsistent.

            Note:
                Overlapping constraints for the same location are merged into a single constraint with the union of
                their time intervals.
            """

            # copy from parent
            constraints = parent.constraints.copy()
            plans = parent.plans.copy()

            # ensure constraint set is present for this agent
            a = next(iter(new_constraints))[0]
            constraints[a] = set() if a not in constraints else constraints[a].copy()

            # add each constraint, merge with existing constraints for the same edge/vertex if the intervals overlap
            # TODO this can be more efficient by first sorting the constraints into the same edge/vertex, then merging, then adding to the set.
            for c in new_constraints:
                replacement_constraint = None
                for existing_constraint in list(constraints[a]):
                    if existing_constraint[0] == c[1]:
                        if existing_constraint[1][0] <= c[2][1] and existing_constraint[1][1] >= c[2][0]:
                            replacement_constraint = (c[1], (min(existing_constraint[1][0], c[2][0]), max(existing_constraint[1][1], c[2][1])))
                            constraints[a].remove(existing_constraint)
                if replacement_constraint is not None:
                    constraints[a].add(replacement_constraint)
                else:
                    constraints[a].add((c[1], c[2]))

            # re-plan path of constrained agent
            modified_unsafe_intervals = modify_unsafe_intervals_with_constraints(constraints[c[0]])
            plans[a] = self.SIPP(
                start_time=self.agent_last_point[a][1],
                start_vertex=self.agent_last_point[a][0],
                end_vertex=self.agent_to_task[a][0] if a == agent else None,
                unsafe_intervals=modified_unsafe_intervals
            )
            if plans[a] is None:  # the constraints are inconsistent
                return None
            cost = plans[agent][-1][1]

            return CTNode(cost, plans, constraints)

        def spawn_node_from_new_path(parent, a, new_path):
            """Creates a new planning node by replacing an agent's path.

            Generates a new search node by copying the parent's plans and constraints,
            then replacing the specified agent's path with a new one. The function
            handles the transition between the old and new paths by adjusting the
            agent's last wait action.

            Args:
                parent: Parent planning node containing [cost, plans, constraints].
                a: The agent whose path is being replaced.
                new_path: The new path for the agent, starting after the agent's
                         current position.

            Returns:
                A new planning node [cost, plans, constraints] with the updated path.

            Note:
                Assumes the new path starts after the beginning of the agent's last
                action, which must be a wait action.
            """

            # copy from parent
            constraints = parent.constraints.copy()
            plans = {a_: plan.copy() for a_, plan in parent.plans.items()}

            # trim last wait action, or remove it if the new path starts when the last wait starts
            last_wait = plans[a][-1]
            if round(last_wait[1], 6) < round(new_path[0][1], 6):
                plans[a][-1] = (last_wait[0], last_wait[1], new_path[0][1])
            else:
                plans[a].pop()

            plans[a] += new_path
            cost = plans[agent][-1][1]

            return CTNode(cost, plans, constraints)

        class CTNode:

            def __init__(self, cost, plans, constraints, collision_type=None):
                self.cost = cost
                self.plans = plans
                self.constraints = constraints
                self.collision_type = collision_type # EE: edge-edge, VE: vertex-edge collision, VE-inf: vertex-edge collision with infinite wait

            def __lt__(self, other):

                if self.cost != other.cost:
                    return self.cost < other.cost
                elif self.collision_type != other.collision_type:
                    if self.collision_type == 'EE':
                        return True
                    elif self.collision_type == 'VE' and other.collision_type == 'VE-inf':
                        return True
                    else:
                        return False
                else:
                    return True  # arbitrary tie-breaker

        # Construct the ct-tree root
        plans = dict()
        for a in self.agents:
            if a == agent:
                plan = self.SIPP(
                    start_time=self.agent_last_point[a][1],
                    start_vertex=self.agent_last_point[a][0],
                    end_vertex=self.agent_to_task[a][0],
                    unsafe_intervals=self.unsafe_intervals)
            else:
                plan = [(self.agent_last_point[a][0], self.agent_last_point[a][1], np.inf)]  # wait at node
            plans[a] = plan
        cost = plans[agent][-1][1]
        ct_root = CTNode(cost, plans, dict(), None)  # constraints

        ct_nodes = [ct_root]
        heapq.heapify(ct_nodes)

        timeout = 10
        comp_time_start = time.time()
        while ct_nodes:
            """
            if time.time() - comp_time_start > timeout:
                self.log['Error'] = 'Timeout in CCBS'
                raise Exception('Timeout in CCBS')
            """
            if time.time() - comp_time_start > self.CCBS_time_limit:
                return None

            node = heapq.heappop(ct_nodes)
            collision = detect_collision(node)

            if collision is None:
                return collect_plans(node)
            else:

                # edge-edge collision
                if len(collision[0]) == 3 and len(collision[1]) == 3:

                    # if first agent gets priority
                    constraints_1 = set()
                    tau0, tau1 = self.E_eec[collision[0][1]][collision[1][1]]
                    unsafe_interval = (collision[0][2] + tau0, collision[0][2] + tau1)
                    constraints_1.add((collision[1][0], collision[1][1], unsafe_interval))

                    # if second agent gets priority
                    constraints_2 = set()
                    tau0, tau1 = self.E_eec[collision[1][1]][collision[0][1]]
                    unsafe_interval = (collision[1][2] + tau0, collision[1][2] + tau1)
                    constraints_2.add((collision[0][0], collision[0][1], unsafe_interval))

                    new_node1 = spawn_node_from_constraints(node, constraints_1)
                    new_node2 = spawn_node_from_constraints(node, constraints_2)

                    if new_node1 is not None:
                        new_node1.collision_type = 'EE'
                        heapq.heappush(ct_nodes, new_node1)
                    if new_node2 is not None:
                        new_node2.collision_type = 'EE'

                        heapq.heappush(ct_nodes, new_node2)

                # vertex-edge collision
                elif len(collision[0]) == 4 and len(collision[1]) == 3:

                    if collision[0][3] == np.inf:

                        idle_agent = collision[0][0]
                        if idle_agent in node.constraints:
                            modified_unsafe_intervals = modify_unsafe_intervals_with_constraints(node.constraints[idle_agent])
                        else:
                            modified_unsafe_intervals = self.unsafe_intervals
                        new_path = self.SIPP(node.plans[idle_agent][-1][1],
                                             node.plans[idle_agent][-1][0],
                                             None,
                                             modified_unsafe_intervals)
                        if new_path is not None:
                            new_node1 = spawn_node_from_new_path(node, idle_agent, new_path)
                            new_node1.collision_type = 'VE-inf'
                            heapq.heappush(ct_nodes, new_node1)

                    else:

                        # get collision interval
                        tau0, tau1 = self.V_vec[collision[1][1]][collision[0][1]]
                        vertex_unsafe_interval = (collision[1][2] + tau0, collision[1][2] + tau1)
                        collision_interval = (
                            max(vertex_unsafe_interval[0], collision[0][2]),
                            min(vertex_unsafe_interval[1], collision[0][3]))

                        # if the moving agent get priority
                        constraints_1 = set()
                        constraints_1.add((collision[0][0], collision[0][1], collision_interval))

                        # if the waiting agent gets priority
                        constraints_2 = set()
                        tau0, tau1 = self.E_vec[collision[0][1]][collision[1][1]]
                        unsafe_interval = (collision_interval[0] + tau0, collision_interval[1] + tau1)
                        constraints_2.add((collision[1][0], collision[1][1], unsafe_interval))

                        # add the new nodes to the queue
                        new_node1 = spawn_node_from_constraints(node, constraints_1)
                        new_node2 = spawn_node_from_constraints(node, constraints_2)

                        if new_node1 is not None:
                            new_node1.collision_type = 'VE'
                            heapq.heappush(ct_nodes, new_node1)
                        if new_node2 is not None:
                            new_node2.collision_type = 'VE'

                            heapq.heappush(ct_nodes, new_node2)

        raise Exception('No plan found, this should never happen')

    def SIPP(self, start_time, start_vertex, end_vertex, unsafe_intervals, quick_return=False):
        """Plans a path from start to end vertex while avoiding unsafe time intervals.

        Implements Safe Interval Path Planning (SIPP) algorithm that performs A* search in the
        (vertex, interval) space. Each node represents a vertex and a safe time interval when
        the agent can be at that vertex.

        Args:
            start_time: The time when the agent starts at the start vertex.
            start_vertex: The vertex where the agent starts.
            end_vertex: The target vertex, or None if any vertex is acceptable.
            unsafe_intervals: Dictionary mapping vertices/edges to lists of time intervals when they cannot be used.
            quick_return: If True, finds a path to any safe vertex quickly without guaranteeing optimality.

        Returns:
            A list of actions for the agent to follow, where each action is either:
            - A move action: (edge, start_time)
            - A wait action: (vertex, start_time, end_time)
            The final action is always an infinite wait at the destination.
            Returns None if no valid path exists.

        Note:
            The algorithm ensures the agent can remain at the end vertex indefinitely.
            When quick_return=True, the agent avoids vertices where other agents are
            scheduled to be after the current time.
        """

        @dataclass
        class Node:
            v: str
            I: tuple[float, float]
            end_vertex_reached: bool
            f: float
            arrival_time: float
            parent_edge: tuple[str, str] | None
            parent_node: Self | None

            def __lt__(self, other):
                return self.f < other.f

        # root and node list
        root_intervals = self.intersected_safe_intervals(unsafe_intervals[start_vertex], (start_time, start_time))
        root_interval = (start_time, start_time) if not root_intervals else root_intervals[0]
        root_f = start_time + self.dist[start_vertex][end_vertex] if end_vertex is not None else start_time
        root = Node(start_vertex,
                    root_interval,
                    end_vertex is None or quick_return,
                    root_f,
                    start_time,
                    None,
                    None)
        nodes = {f'{root.v}{root.I}{root.end_vertex_reached}': root}
        queue = [root]
        heapq.heapify(queue)

        while queue:
            node = heapq.heappop(queue)

            # Skip if this is an outdated version of a node
            node_label = f'{node.v}{node.I}{node.end_vertex_reached}'
            if node_label in nodes and node is not nodes[node_label]:
                continue

            # Check if node is a goal node
            if (node.I[1] == np.inf
                    and node.end_vertex_reached
                    and ((end_vertex is not None and not quick_return) or node.parent_node is not None)):
                # add final wait action

                actions = [(node.v, node.arrival_time, np.inf)]

                # add movement actions and implicit wait actions
                traversal_time = None
                while node is not root:

                    # move action
                    traversal_time = round(node.arrival_time - self.edge_weight(node.parent_edge), 6)
                    actions.append((node.parent_edge, traversal_time))

                    # Wait action
                    previous_node_arrival_time = node.parent_node.arrival_time
                    if round(previous_node_arrival_time, 6) < round(traversal_time, 6):
                        actions.append((node.parent_node.v, previous_node_arrival_time, traversal_time))

                    node = node.parent_node

                actions.reverse()
                return actions

            # Add/update successor nodes
            successors = self.get_successors(node.v, (node.arrival_time, node.I[1]), unsafe_intervals)
            # if quick return, then agent cannot intersect any node that another agent was last scheduled at, at a time
            # after it was there.
            if quick_return:
                successors_to_remove = set()
                for s in list(successors):

                    # check if edge intersects with any nodes where other agents have last been scheduled to
                    for v, (tau0, tau1) in self.V_vec[s[0]].items():
                        unsafe_interval_at_v = (s[2] + tau0, s[2] + tau1)
                        if any(lp[0] == v and lp[0] != start_vertex
                               and round(lp[1], 6) <= unsafe_interval_at_v[1]
                               for lp in self.agent_last_point.values()):
                            successors_to_remove.add(s)
                successors -= successors_to_remove

            for s in successors:

                edge = s[0]
                next_node_interval = s[1]
                departure_time = s[2]
                next_node_goal_reached = node.end_vertex_reached or edge[1] == end_vertex
                next_node_label = f'{edge[1]}{next_node_interval}{next_node_goal_reached}'
                next_node_arrival_time = round(departure_time + self.edge_weight(edge), 6)
                if end_vertex is None:
                    next_node_f = next_node_arrival_time
                elif node.end_vertex_reached and not quick_return:
                    next_node_f = node.f
                else:
                    next_node_f = round(next_node_arrival_time + self.dist[edge[1]][end_vertex], 6)

                # Update next node if already visited, else create it
                if next_node_label in nodes:

                    # If the new value is lower, update it. However, if the start vertex must be left, then we must be
                    # able to update the root node.
                    if (next_node_f < nodes[next_node_label].f
                            or (end_vertex is None and nodes[next_node_label] == root and nodes[next_node_label].parent_node is None)):
                        nodes[next_node_label].f = next_node_f
                        nodes[next_node_label].arrival_time = next_node_arrival_time
                        nodes[next_node_label].parent_edge = edge
                        nodes[next_node_label].parent_node = node
                        heapq.heappush(queue, nodes[next_node_label])

                else:
                    next_node = Node(edge[1], next_node_interval, next_node_goal_reached, next_node_f, next_node_arrival_time, edge, node)
                    nodes[next_node_label] = next_node
                    heapq.heappush(queue, next_node)

        return None  # No path found

    def update_completed_tasks(self, curr_time):
        # get completed tasks
        completed_tasks = dict()
        for task in self.task_set:
            task_v, task_t = task

            # check vertex_utilisation for if an agent was present at the task vertex at or after the task time
            if task_v not in self.vertex_utilisation:
                continue

            # index of the first interval that ends after the task time
            i = bisect.bisect_left(self.vertex_utilisation[task_v], task_t, key=lambda interval: interval[1])
            if i < len(self.vertex_utilisation[task_v]):

                completion_time = max(task_t, self.vertex_utilisation[task_v][i][0])

                # update completion after it has happened.
                """
                if curr_time < completion_time:
                    continue
                """
                if task in completed_tasks:
                    completed_tasks[task] = min(completed_tasks[task], completion_time)
                else:
                    completed_tasks[task] = completion_time

        # remove completed tasks
        for task, completion_time in completed_tasks.items():

            self.task_set.remove(task)
            del self.task_priorities[task]

            # update log
            if curr_time in self.log['Task completion']:
                self.log['Task completion'][curr_time].append((task[0], task[1], completion_time))
            else:
                self.log['Task completion'][curr_time] = [(task[0], task[1], completion_time)]

            # update task assignment and hpt/hpa
            agent = self.task_to_agent.pop(task, None)
            if agent is not None:
                self.agent_to_task[agent] = None
            if self.hpt == task:
                self.hpt = None
                self.hpa = None
                self.hpt_end_time = completion_time

    def update_with_plans(self, plans):
        """Updates agent plans and unsafe intervals with new action plans.

        Integrates new agent plans into the existing plans, updating the agent's last known
        position and the unsafe intervals for collision avoidance. Handles the transition
        between existing plans and new plans by merging wait actions when appropriate.

        Args:
            plans: Dictionary mapping agent IDs to lists of action tuples, where each action
                   is either a move action (edge, time) or a wait action (vertex, start_time, end_time).

        Note:
            For each action added:
            - Move actions update the agent's position to the destination vertex
            - Wait actions extend the agent's stay at the current vertex
            - Unsafe intervals are updated to reflect where agents will be at what times
        """

        for a, plan in plans.items():

            # if end of existing plan was before start of current plan, add wait action
            if self.plans[a] and len(self.plans[a][-1]) == 3 and len(plan[0]) == 3:
                new_wait = (plan[0][0], self.plans[a][-1][1], plan[0][2])
                plan[0] = new_wait
                self.plans[a].pop()

            for action in plan:

                # move action
                if len(action) == 2:
                    e, t = action
                    arrival_time = t + self.edge_weight(e)
                    self.plans[a].append(action)
                    self.agent_last_point[a] = (e[1], arrival_time)
                    self.update_unsafe_intervals_with_move(e, t)

                    # update vertex utilisation
                    if e[1] in self.vertex_utilisation:
                        self.vertex_utilisation[e[1]] = self.insert_interval(self.vertex_utilisation[e[1]], (arrival_time, arrival_time))
                    else:
                        self.vertex_utilisation[e[1]] = [(arrival_time, arrival_time)]

                # wait action
                elif len(action) == 3:
                    v, t0, t1 = action
                    self.plans[a].append(action)
                    self.agent_last_point[a] = (v, t1)
                    self.update_unsafe_intervals_with_wait(v, t0, t1)

                    # update vertex utilisation
                    if v in self.vertex_utilisation:
                        self.vertex_utilisation[v] = self.insert_interval(self.vertex_utilisation[v], (t0, t1))
                    else:
                        self.vertex_utilisation[v] = [(t0, t1)]

    def update_task_priorities(self, curr_time):
        """Updates the priority ranking of tasks based on their waiting time.

        Calculates task priorities by measuring the time elapsed since each task's
        release time, then sorts tasks in descending order of priority (longer
        waiting time = higher priority).

        Args:
            curr_time: The current simulation time used to calculate waiting times.

        Note:
            This implements a simple aging mechanism where tasks that have waited
            longer receive higher priority, preventing starvation of older tasks.
        """
        self.task_priorities = {task: curr_time - task[1] for task in self.task_set}

    def update_assigned_agent_tasks(self):
        """Assigns tasks to agents based on estimated arrival times.

        First assigns the high-priority task (hpt) to the high-priority agent (hpa),
        then assigns remaining tasks to other agents based on which agent can reach
        each task earliest. Tasks are considered in order of their priority ranking.

        Note:
            This greedy assignment approach minimizes individual task completion times
            rather than optimizing for global efficiency. The high-priority task-agent
            pair is preserved regardless of potential efficiency gains from reassignment.
        """

        assigned_tasks = {agent: None for agent in self.agents}

        if self.hpa is not None:
            assigned_tasks[self.hpa] = self.hpt
        assignable_agents = self.agents - {self.hpa}
        assignable_tasks = self.task_set - {self.hpt}

        ordered_tasks = sorted(list(assignable_tasks), key=lambda x: self.task_priorities[x], reverse=True)
        for task in ordered_tasks:
            if task not in assignable_tasks:
                continue
            if not assignable_agents:
                break
            arrival_times = {agent: self.agent_last_point[agent][1] + self.dist[self.agent_last_point[agent][0]][task[0]] for agent in assignable_agents}
            agent = min(arrival_times, key=lambda x: arrival_times[x])
            assigned_tasks[agent] = task
            assignable_agents.remove(agent)

        self.agent_to_task = assigned_tasks
        self.task_to_agent = {task: agent for agent, task in assigned_tasks.items() if task is not None}

    def randomize_assigned_agent_tasks(self):

        assigned_tasks = {agent: None for agent in self.agents}
        assignable_agents = self.agents.copy()
        assignable_tasks = self.task_set.copy()

        ordered_tasks = sorted(list(assignable_tasks), key=lambda x: self.task_priorities[x], reverse=True)
        for task in ordered_tasks:
            if task not in assignable_tasks:
                continue
            if not assignable_agents:
                break

            assigned_agent = random.choice(list(assignable_agents))
            assigned_tasks[assigned_agent] = task
            assignable_agents.remove(assigned_agent)

        self.agent_to_task = assigned_tasks
        self.task_to_agent = {task: agent for agent, task in assigned_tasks.items() if task is not None}

    @staticmethod
    def insert_interval(intervals, new_interval):
        """ Inserts a new interval into a list of intervals. Any overlaps are merged.

        Args:
            intervals: list of tuples, intervals
            new_interval: tuple, interval to insert

        Returns:
            merged_intervals: list of tuples, merged intervals
        """
        start, end = new_interval

        # Find the insertion index using binary search
        i = bisect.bisect_left(intervals, (start,))

        # Merge overlapping intervals
        if i > 0 and intervals[i - 1][1] >= start:
            i -= 1  # Move back to the overlapping interval

        merged_intervals = intervals[:i]  # Keep intervals before insertion point

        while i < len(intervals) and intervals[i][0] <= end:
            start = min(start, intervals[i][0])
            end = max(end, intervals[i][1])
            i += 1

        merged_intervals.append((start, end))

        # Add remaining intervals
        merged_intervals.extend(intervals[i:])

        return merged_intervals

    def update_unsafe_intervals_with_wait(self, v, t0, t1):
        """Updates unsafe intervals when an agent waits at a vertex.

        When an agent waits at vertex v from time t0 to t1, this function updates
        the unsafe intervals for:
        1. Edges that could collide with the waiting agent
        2. Vertices that could collide with the waiting agent

        Args:
            v: The vertex where the agent is waiting.
            t0: The start time of the wait action.
            t1: The end time of the wait action.

        Note:
            Uses the E_vec mapping to find edges affected by the waiting agent,
            and the V_vvc mapping to find vertices affected by the waiting agent.
            The unsafe intervals are updated by inserting the appropriate time
            intervals during which collisions could occur.
        """

        # update affected edges
        for e, (tau0, tau1) in self.E_vec[v].items():
            self.unsafe_intervals[e] = self.insert_interval(self.unsafe_intervals[e],(t0 + tau0, t1 + tau1))

        # update affected vertices
        # for v_ in self.V_vvc[v]:
        #    self.insert_interval(self.unsafe_intervals[v_],(t0, t1))

    def update_unsafe_intervals_with_move(self, e, t):
        """Updates unsafe intervals when an agent moves along an edge.

        When an agent traverses edge e starting at time t, this function updates
        the unsafe intervals for:
        1. Edges that could collide with the moving agent
        2. Vertices that could collide with the moving agent

        Args:
            e: The edge being traversed by the agent.
            t: The time when the agent starts traversing the edge.

        Note:
            Uses the E_eec mapping to find edges affected by the moving agent,
            and the V_vec mapping to find vertices affected by the moving agent.
            The unsafe intervals are updated by inserting the appropriate time
            intervals during which collisions could occur.
        """

        # update affected edges
        for e_, (tau0, tau1) in self.E_eec[e].items():
            self.unsafe_intervals[e_] = self.insert_interval(self.unsafe_intervals[e_], (t + tau0, t + tau1))

        # update affected vertices
        for v, (tau0, tau1) in self.V_vec[e].items():
            self.unsafe_intervals[v] = self.insert_interval(self.unsafe_intervals[v], (t + tau0, t + tau1))

    def edge_weight(self, edge):
        """Helper function to get the weight of an edge."""
        return self.graph.edges[edge]['weight']

    @staticmethod
    def intersected_safe_intervals(unsafe_intervals, interval):
        """ Get the safe intervals, given a set of unsafe intervals and a time interval which they must intersect.

        Args:
            unsafe_intervals: list of tuples, unsafe intervals, in ascending order
            interval: tuple, time interval

        Returns:
            safe_intervals: list of tuples, safe intervals
        """

        if not unsafe_intervals:
            return [(-np.inf, np.inf)]

        safe_intervals = []

        # idx of first interval to start before interval
        i = bisect.bisect_left(unsafe_intervals, (interval[0],)) - 1
        while i < len(unsafe_intervals):

            si_start = -np.inf if i < 0 else unsafe_intervals[i][1]
            si_end = np.inf if i + 1 >= len(unsafe_intervals) else unsafe_intervals[i + 1][0]
            lower_intersection = max(si_start, interval[0])
            upper_intersection = min(si_end, interval[1])
            if round(lower_intersection, 5) <= round(upper_intersection, 5) and lower_intersection < np.inf and -np.inf < upper_intersection:
                safe_intervals.append((si_start, si_end))

            i += 1

        return safe_intervals

    def get_successors(self, v, interval, unsafe_intervals):
        """ This finds the reachable vertex-interval pairs from a given vertex-interval.

        The given vertex-interval pair implies that an agent can move from the vertex at any time within the interval.
        This function then finds every edge-interval-time, where traversing the edge at time will land the agent at the
        edge's target vertex at the start of the interval, being the target vertex's safe interval.

        Args:
            v: vertex label
            interval: tuple, safe interval at v,
            unsafe_intervals: dict, vertex/edge: list of tuples, unsafe intervals

        Returns:
            successors: set of tuples, (edge, safe_interval, time)
        """
        successors = set()
        for e in self.graph.out_edges(v):

            Je = self.intersected_safe_intervals(unsafe_intervals[e], interval)                 # all reachable intervals on e

            for tau0, tau1 in Je:
                t0_, t1_ = max(tau0, interval[0]), min(tau1, interval[1])                       # possible time to traverse e
                s0, s1 = t0_ + self.edge_weight(e), t1_ + self.edge_weight(e)                   # possible arrival time at next vertex
                Ju = self.intersected_safe_intervals(unsafe_intervals[e[1]], (s0, s1))  # all reachable intervals at next vertex

                for tau0_, tau1_ in Ju:
                    s0_, s1_ = max(tau0_, s0), min(tau1_, s1)
                    successors.add((e, (tau0_, tau1_), round(s0_ - self.edge_weight(e), 6)))

        return successors