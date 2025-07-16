import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow
import matplotlib.colors as mcolors
import networkx as nx
import datetime
import os
import numpy as np


class frameDrawer:

    def __init__(self, fig, ax, graph, planner, agent_speed, agent_radius, time_per_frame, movement_time):
        self.graph = graph
        self.planner = planner
        self.agent_start = planner.agent_start
        self.agents = list(planner.agents)
        self.agent_speed = agent_speed
        self.agent_radius = agent_radius
        self.fig = fig
        self.ax = ax
        self.time_per_frame = time_per_frame
        self.movement_time = movement_time

        self.curr_paths = {agent: [] for agent in self.agents}
        self.tasks = [task for task_list in planner.log['Task completion'].values() for task in task_list]
        self.tasks.sort(key=lambda x: x[1])
        self.active_tasks = []
        self.curr_task_index = 0

    def draw_frame(self, frame):

        def draw_graph():

            # nodes
            for v in self.graph.nodes:
                pos = nx.get_node_attributes(self.graph, 'pos')
                self.ax.add_patch(
                    Circle(pos[v], node_size, facecolor=graph_color, edgecolor=None)
                )
                # self.ax.text(pos[v][0], pos[v][1], v, fontsize=12, color=node_font_color, ha='center', va='center')

            # edges
            for u, v in self.graph.edges:
                # offset the arrow head to avoid overlap with the node
                start = (self.graph.nodes[u]['pos'][0], self.graph.nodes[u]['pos'][1])
                end = (self.graph.nodes[v]['pos'][0], self.graph.nodes[v]['pos'][1])
                length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
                direction = ((self.graph.nodes[v]['pos'][0] - self.graph.nodes[u]['pos'][0]) / length,
                             (self.graph.nodes[v]['pos'][1] - self.graph.nodes[u]['pos'][1]) / length)
                offset_length = length - node_size
                offset_end = (start[0] + offset_length * direction[0], start[1] + offset_length * direction[1])

                edge_arrow = FancyArrow(
                    start[0], start[1], offset_end[0] - start[0], offset_end[1] - start[1],
                    width=edge_width, color=graph_color,
                    length_includes_head=True,
                    head_width=edge_arrow_head_size, head_length=edge_arrow_head_size,
                )
                self.ax.add_patch(edge_arrow)

            # get bounding box of all nodes and set ax limits
            x = [self.graph.nodes[v]['pos'][0] for v in self.graph.nodes]
            y = [self.graph.nodes[v]['pos'][1] for v in self.graph.nodes]
            self.ax.set_xlim(min(x) - border_margin, max(x) + border_margin)
            self.ax.set_ylim(min(y) - border_margin, max(y) + border_margin)

            # print time
            self.ax.text(np.mean(x), max(y)+border_margin/2, f'{time:.1f}', fontsize=14, color='black', ha='center', va='center')

        def draw_agents():

            agent_colors = {
                agent: plt.cm.get_cmap(agent_color_map, len(self.agents))(i)  # Direct colormap call instead of .colors
                for i, agent in enumerate(self.agents)
            }
            #agent_colors = {agent: '#0055FF' for i, agent in enumerate(self.agents)}

            # get agent positions
            agent_positions = dict()
            for agent in self.agents:

                pos = None
                for action in self.planner.plans[agent]:

                    # move_action
                    if len(action) == 2:
                        if action[1] <= time <= action[1] + self.graph.edges[action[0]]['weight']:
                            direction = (
                                (self.graph.nodes[action[0][1]]['pos'][0] - self.graph.nodes[action[0][0]]['pos'][0]) / self.graph.edges[action[0]]['weight'],
                                (self.graph.nodes[action[0][1]]['pos'][1] - self.graph.nodes[action[0][0]]['pos'][1]) / self.graph.edges[action[0]]['weight']
                            )
                            time_on_edge = time - action[1]
                            pos = (self.graph.nodes[action[0][0]]['pos'][0] + direction[0] * time_on_edge * self.agent_speed,
                                   self.graph.nodes[action[0][0]]['pos'][1] + direction[1] * time_on_edge * self.agent_speed)
                            break

                    # wait_action
                    else:
                        if action[1] <= time <= action[2]:
                            pos = self.graph.nodes[action[0]]['pos']
                            break

                if pos is None:

                    # reached end of plan
                    if self.planner.plans[agent]:
                        last_action = self.planner.plans[agent][-1]
                        if len(last_action) == 2:
                            pos = self.graph.nodes[last_action[0][1]]['pos']
                        else:
                            pos = self.graph.nodes[last_action[0]]['pos']

                    # no plan, it never moved
                    else:
                        pos = self.graph.nodes[self.agent_start[agent]]['pos']

                agent_positions[agent] = pos

            # draw agents
            for agent in self.agents:

                circle = Circle(agent_positions[agent], self.agent_radius,
                                facecolor=mcolors.to_rgba(agent_colors[agent], alpha=agent_alpha),
                                edgecolor=None,
                                linestyle=(0, (15, 5)), linewidth=1)
                self.ax.add_patch(circle)

                """
                self.ax.text(agent_positions[agent][0] + self.agent_radius * agent_label_offset_factor[0],
                             agent_positions[agent][1] + self.agent_radius * agent_label_offset_factor[1],
                             agent, fontsize=12, color=agent_colors[agent], ha='center', va='center')
                """

            # collect new agent paths
            time_prev = time - self.time_per_frame
            new_plans = {t: plans for t, plans in self.planner.log['Plan'].items() if time_prev < t <= time}
            for t, plans in new_plans.items():

                if plans['CCBS'] is not None:

                    # store HPA path
                    hpa_path = plans['CCBS'][plans['HPA']]
                    self.curr_paths[plans['HPA']].append({'type': 'HPA',
                                                          'path': [action for action in hpa_path if len(action) == 2]})

                    # store avoidance path
                    for agent in plans['CCBS']:
                        if agent != plans['HPA']:
                            self.curr_paths[agent].append({'type': 'Avoidance',
                                                           'path': [action for action in plans['CCBS'][agent] if len(action) == 2]})

                for agent in plans['SIPP']:
                    self.curr_paths[agent].append({'type': 'SIPP',
                                                   'path': [action for action in plans['SIPP'][agent] if len(action) == 2]})

            # update agent paths to remove completed move actions
            for agent, plans in self.curr_paths.items():
                if not plans:
                    continue

                while plans and plans[0]['path'][0][1] + self.graph.edges[plans[0]['path'][0][0]]['weight'] < time:
                    plans[0]['path'].pop(0)
                    if not plans[0]['path']:
                        plans.pop(0)

            # draw agent paths
            for agent, plans in self.curr_paths.items():
                if not plans:
                    continue

                for plan in plans:
                    points_to_connect = []
                    for action in plan['path']:
                        if action[1] < time:
                            points_to_connect.append(agent_positions[agent])
                        else:
                            points_to_connect.append(self.graph.nodes[action[0][0]]['pos'])
                    points_to_connect.append(self.graph.nodes[plan['path'][-1][0][1]]['pos'])

                    x, y = zip(*points_to_connect)
                    plt.plot(x, y,
                             linewidth=3,
                             color=path_color[plan['type']])

        def draw_tasks():

            # add new tasks to the active task set
            while self.curr_task_index < len(self.tasks) and self.tasks[self.curr_task_index][1] <= time:
                self.active_tasks.append(self.tasks[self.curr_task_index])
                self.curr_task_index += 1

            # remove completed tasks
            self.active_tasks = [task for task in self.active_tasks if time < task[2]]

            # draw active tasks
            for task in self.active_tasks:
                pos = self.graph.nodes[task[0]]['pos']
                circle = Circle(pos, task_size, facecolor=task_color, edgecolor=None, alpha=1)
                self.ax.add_patch(circle)

        # params
        graph_color = 'lightgrey'
        node_size = 0.05
        node_font_color = 'black'
        edge_width = 0.02
        edge_arrow_head_size = 0
        border_margin = 6
        agent_color_map = 'magma'
        agent_alpha = 1
        agent_label_offset_factor = (0.4, -0.4)
        task_color = 'green'
        task_size = 0.25

        path_color = {'HPA': '#c334eb', 'SIPP': '#00a9b5', 'Avoidance': '#f56a00'}

        time = frame * self.time_per_frame
        # print progress at every 25% of the movement time
        if time % (self.movement_time / 10) < self.time_per_frame:
            print(f"Progress: {time/self.movement_time:.0%}")

        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.set_axisbelow(True)
        self.ax.set_aspect('equal')
        # self.ax.grid(True, linestyle='--', linewidth=0.25)

        draw_graph()
        draw_tasks()
        draw_agents()


def animate_MAPF(graph, planner, agent_speed, agent_radius, fps=25, time_scale=1):

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    movement_end_times = [p[-1][1] + graph.edges[p[-1][0]]['weight'] if p and len(p[-1])==2 else p[-1][1]
                          for p in planner.plans.values()]
    movement_time = max(movement_end_times) if movement_end_times else 0

    pause_at_end = 1
    animation_time = (pause_at_end + movement_time) / time_scale
    total_nr_frames = int(animation_time * fps) + 1
    movement_time_per_frame = movement_time / total_nr_frames

    fd = frameDrawer(fig, ax, graph, planner, agent_speed, agent_radius, movement_time_per_frame, movement_time)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = 'videos'
    os.makedirs(folder, exist_ok=True)
    file_name = f'{folder}/vid_LMAPFcT_{date}.mp4'

    print("Animating...")
    ani = animation.FuncAnimation(fig, fd.draw_frame, frames=total_nr_frames, interval=1000/fps, repeat=False)
    ani.save(file_name, writer='ffmpeg', dpi=300)
    print("Done!")


def plot_graph(graph, agent_start, agent_radius):

    # params
    graph_color = '#cecee0'
    node_size = 0.1
    node_font_color = 'black'
    edge_width = 0.1
    edge_arrow_head_size = 0.03
    border_margin = 2
    agent_color_map = 'magma'
    agent_alpha = 0.5
    agent_label_offset_factor = (0.4, -0.4)

    fig, ax = plt.subplots(figsize=(25, 25))


    # ---------- GRAPH ----------
    # nodes
    for v in graph.nodes:
        pos = nx.get_node_attributes(graph, 'pos')
        ax.add_patch(
            Circle(pos[v], node_size, facecolor=graph_color, edgecolor=None)
        )

    # edges
    for u, v in graph.edges:
        # offset the arrow head to avoid overlap with the node
        start = (graph.nodes[u]['pos'][0], graph.nodes[u]['pos'][1])
        end = (graph.nodes[v]['pos'][0], graph.nodes[v]['pos'][1])
        length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        direction = ((graph.nodes[v]['pos'][0] - graph.nodes[u]['pos'][0]) / length,
                     (graph.nodes[v]['pos'][1] - graph.nodes[u]['pos'][1]) / length)
        offset_length = length - node_size
        offset_end = (start[0] + offset_length * direction[0], start[1] + offset_length * direction[1])

        edge_arrow = FancyArrow(
            start[0], start[1], offset_end[0] - start[0], offset_end[1] - start[1],
            width=edge_width, color=graph_color,
            length_includes_head=True,
            head_width=edge_arrow_head_size, head_length=edge_arrow_head_size,
        )
        ax.add_patch(edge_arrow)

    # get bounding box of all nodes and set ax limits
    x = [graph.nodes[v]['pos'][0] for v in graph.nodes]
    y = [graph.nodes[v]['pos'][1] for v in graph.nodes]
    ax.set_xlim(min(x) - border_margin, max(x) + border_margin)
    ax.set_ylim(min(y) - border_margin, max(y) + border_margin)

    # ---------- AGENTS ----------
    agent_colors = {agent: plt.cm.get_cmap(agent_color_map, len(agent_start)).colors[i]
                    for i, agent in enumerate(agent_start)}
    agent_positions = {agent: graph.nodes[start_v]['pos'] for agent, start_v in agent_start.items()}

    # draw agents
    for agent, pos in agent_positions.items():
        circle = Circle(agent_positions[agent], agent_radius,
                        facecolor=mcolors.to_rgba(agent_colors[agent], alpha=agent_alpha),
                        edgecolor=None,
                        linestyle=(0, (15, 5)),
                        linewidth=1)
        ax.add_patch(circle)

    # show axis and grid with ticks
    ax.set_axisbelow(True)
    ax.set_aspect('equal')
    ax.axis('off')
    # Remove padding around the axis
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # save and show
    plt.savefig('graph.png', dpi=300)
    plt.show()
    # ax.grid(False, linestyle='--', linewidth=0.25)
