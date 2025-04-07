import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import json
import os
import datetime
from scipy.spatial import Voronoi
import multiprocessing
import uuid


def generate_instance(n_agents, n_vertices, n_tasks, end_time, space_size=(30, 30), gamma1=0.2, gamma2=0.02, min_agent_separation=2):
    """Generates a MAPF problem instance with a Voronoi-based graph.

    Creates a problem instance with a graph based on a Voronoi diagram,
    places agents with minimum separation, and generates random tasks.

    Args:
        n_agents (int): Number of agents to place.
        n_vertices (int): Target number of vertices in the final graph.
        n_tasks (int): Number of tasks to generate.
        end_time (float): Maximum time for task generation.
        space_size (tuple, optional): Dimensions of the space. Defaults to (30, 30).
        gamma1 (float, optional): Proportion of extra vertices to generate initially. Defaults to 0.2.
        gamma2 (float, optional): Proportion of random edges to add. Defaults to 0.02.
        min_agent_separation (float, optional): Minimum distance between agents. Defaults to 2.

    Returns:
        tuple: (networkx.DiGraph, dict of agent start positions, list of tasks)

    Raises:
        Exception: If graph creation or agent placement fails.
    """

    def create_voronoi_graph(n_vertices, space_size):
        """Creates a Voronoi diagram graph from random points.

        Args:
            n_vertices (int): Number of points to generate.
            space_size (list): 2D space dimensions [width, height].

        Returns:
            tuple: (networkx.Graph of the Voronoi diagram, numpy.ndarray of point coordinates)
        """

        # generate random points
        points = np.random.uniform(0, [space_size[0], space_size[1]], size=(n_vertices, 2))

        # create a graph, add vertices with the point positions
        G = nx.Graph()
        for i, point in enumerate(points):
            G.add_node(i, pos=tuple(point))

        # Compute Voronoi diagram
        vor = Voronoi(points)

        # Add edges based on ridge points
        for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
            # Ridge points contains indices of the points whose cells are adjacent
            # Add an edge between these points in our graph
            if ridge_vertices[0] >= 0 and ridge_vertices[1] >= 0:  # Avoid infinite ridges
                G.add_edge(ridge_points[0], ridge_points[1])

        return G, points

    def create_graph(vertices, connections):
        """Creates a directed graph from vertices and connections.

        Args:
            vertices (dict): Dictionary mapping node IDs to coordinate tuples.
            connections (list): List of node ID pairs to connect with edges.

        Returns:
            networkx.DiGraph: Directed graph with weighted bidirectional edges.
        """

        graph = nx.DiGraph()

        for node_id, coord in vertices.items():
            graph.add_node(node_id, pos=(float(coord[0]), float(coord[1])))

        for node1, node2 in connections:
            distance = float(np.linalg.norm(np.array(vertices[node1]) - np.array(vertices[node2])))
            graph.add_edge(node1, node2, weight=distance, opposite_edge=(node2, node1))
            graph.add_edge(node2, node1, weight=distance, opposite_edge=(node1, node2))

        return graph

    def generate_agent_start_positions(vertices):
        """Generates starting positions for agents with minimum separation.

        Args:
            vertices (dict): Dictionary mapping vertex IDs to coordinate tuples.

        Returns:
            dict: Mapping of agent IDs to assigned vertex IDs.

        Raises:
            Exception: If an agent cannot be placed with minimum separation.
        """

        agents_start = {f'a{i}': None for i in range(n_agents)}
        vertex_labels = set(vertices.keys())
        for a in agents_start:

            available_vertices = vertex_labels - {v for v in agents_start.values() if v is not None}
            for v in available_vertices:

                if all(np.linalg.norm(vertices[v] - vertices[u]) >= min_agent_separation
                       for u in agents_start.values() if u is not None):
                    agents_start[a] = v
                    break

            if agents_start[a] is None:
                raise Exception('Failed to place agent')

        return agents_start

    def generate_tasks(vertices):
        """Generates random tasks at vertices with random times.

        Args:
            vertices (dict): Dictionary mapping vertex IDs to coordinate tuples.

        Returns:
            list: Time-sorted list of (vertex_id, time) tuples representing tasks.
        """
        tasks = []
        for i in range(n_tasks):
            task_time = np.random.random() * end_time
            task_vertex = str(np.random.choice(list(vertices.keys())))
            tasks.append((task_vertex, task_time))

        tasks.sort(key=lambda x: x[1])
        return tasks

    n_added_points = int(np.ceil(n_vertices * gamma1))
    graph, points = create_voronoi_graph(n_vertices + n_added_points, space_size)

    # remove n_added_points from the graph, ensuring it remains connected (100 attempts)
    nr_failed_attempts = 0
    while n_added_points > 0:

        v = np.random.choice(list(graph.nodes))
        new_graph = graph.copy()
        new_graph.remove_node(v)

        if nx.is_connected(new_graph):
            graph = new_graph
            n_added_points -= 1

        else:
            nr_failed_attempts += 1
            if nr_failed_attempts > 100:
                raise Exception('Failed to create graph')

    # add a few random connections
    for _ in range(int(np.ceil(n_vertices * gamma2))):
        v1 = int(np.random.choice(list(graph)))
        v2 = int(np.random.choice(list(graph)))
        if v1 != v2:
            graph.add_edge(v1, v2)
            graph.add_edge(v2, v1)

    # rename vertices
    vertex_mapping = {j: f'v{i}' for i, j in enumerate(graph.nodes)}
    vertices = {vertex_mapping[node]: np.array(graph.nodes[node]['pos']) for node in graph.nodes}
    connections = {(vertex_mapping[edge[0]], vertex_mapping[edge[1]]) for edge in graph.edges}

    # create final graph from vertices and connections
    graph = create_graph(vertices, connections)

    # create agent start positions and tasks
    agent_start = generate_agent_start_positions(vertices)
    tasks = generate_tasks(vertices)

    return graph, agent_start, tasks


def generate_instance_set(path, n_instances, n_agents, n_vertices, n_tasks, end_time, space_size):
    """Generates and saves a set of MAPF problem instances.

    Creates multiple problem instances and saves them as JSON files in the specified path.

    Args:
        path (str): Directory path where instance files will be saved.
        n_instances (int): Number of instances to generate.
        n_agents (int): Number of agents in each instance.
        n_vertices (int): Number of vertices in each instance graph.
        n_tasks (int): Number of tasks in each instance.
        end_time (float): Maximum time for task generation.
        space_size (tuple): Dimensions of the space.

    Returns:
        None: Instances are saved to files.
    """

    instances = []
    while len(instances) < n_instances:
        try:
            print(f'Generating instance {len(instances) + 1}/{n_instances}')
            instance = generate_instance(n_agents, n_vertices, n_tasks, end_time, space_size=space_size)
            instances.append(instance)
        except:
            print('\tFailed to generate an instance')

    # save instances to files
    print(f'Saving instances to {path}')
    for instance in instances:
        file_hash = hash(datetime.datetime.now())
        file_name = f'{path}/instance_{file_hash}.json'
        save_instance_to_json(instance, file_name)


def create_benchmark_set():
    """Creates a benchmark set of MAPF instances with different parameters.

    Generates multiple sets of MAPF instances with varying numbers of agents, vertices,
    and tasks, organizing them into a structured folder hierarchy with settings metadata.

    Returns:
        None: Benchmark sets are saved to disk in a timestamped folder.
    """

    # define and create folders
    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    master_folder = f'Benchmark_Sets/BENCHMARK_{current_datetime}'
    settings_file_path = os.path.join(master_folder, "settings.json")
    sets_folder = os.path.join(master_folder, "sets")
    os.makedirs(master_folder, exist_ok=True)
    os.makedirs(sets_folder, exist_ok=True)

    # define settings for the instance sets
    settings = dict()
    n_instances = 4
    task_release_end_time = 200
    for vert_factor in [5, 10, 15]:
        for i, n_agents in enumerate([800]):
            n_vertices = n_agents * vert_factor
            task_release_rate = n_agents / 20  # 1 task per agent every x time steps
            n_tasks = int(np.ceil(task_release_rate * task_release_end_time))
            space_side_length = float(3 * np.sqrt(n_vertices))

            settings[f'set_{len(settings)}'] = {
                "n_agents": n_agents,
                "n_vertices": n_vertices,
                "n_tasks": n_tasks,
                "end_time": task_release_end_time,
                "space_size": (space_side_length, space_side_length)
            }

    # Save settings to a JSON file in the master folder
    with open(settings_file_path, "w") as settings_file:
        json.dump(settings, settings_file, indent=4)

    # generate instance set folders and files
    for label, params in settings.items():
        set_folder = os.path.join(sets_folder, label)
        os.makedirs(set_folder, exist_ok=True)

        generate_instance_set(
            path=set_folder,
            n_instances=n_instances,
            n_agents=params["n_agents"],
            n_vertices=params["n_vertices"],
            n_tasks=params["n_tasks"],
            end_time=params["end_time"],
            space_size=params["space_size"]
        )


def save_instance_to_json(instance, path):
    """Saves a MAPF instance to a JSON file.

    Converts a graph, agent start positions, and tasks into a JSON-serializable format
    and saves it to the specified path.

    Args:
        instance (tuple): A tuple containing (graph, agent_start_dict, tasks).
        path (str): File path where the JSON will be saved.

    Returns:
        str: The path where the instance was saved.
    """

    # Unpack the instance
    graph, label_dict, tasks = instance

    # Convert the graph to a JSON-serializable format
    # Explicitly include is_directed information
    graph_data = json_graph.node_link_data(graph)  # Using node_link_data instead of adjacency_data

    # Create a dictionary with all components
    data = {
        "graph": graph_data,
        "agent_start": label_dict,
        "tasks": tasks,
        "is_directed": graph.is_directed()  # Explicitly store whether the graph is directed
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write to file
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    return path


def load_instance_from_json(file_path):
    """Loads a single MAPF instance from a JSON file.

    Reads a JSON file and converts it to a MAPF instance with a graph,
    agent start positions, and tasks.

    Args:
        file_path (str): Path to the instance JSON file.

    Returns:
        tuple: (networkx.DiGraph, dict of agent start positions, list of tasks)
    """

    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract the graph data
    graph_data = data['graph']

    # Create a new DiGraph (always directed)
    G = nx.DiGraph()

    # Add nodes
    for node in graph_data['nodes']:
        node_id = node.pop('id')
        G.add_node(node_id, **node)

    # Add edges
    if 'adjacency' in graph_data:  # For adjacency_data format
        for edge in graph_data['adjacency']:
            source = edge[0]
            for target_data in edge[1:]:
                target = target_data.get('id', target_data)
                attrs = {k: v for k, v in target_data.items() if k != 'id'}
                G.add_edge(source, target, **attrs)
    elif 'links' in graph_data:  # For node_link_data format
        for link in graph_data['links']:
            source = link['source']
            target = link['target']
            attrs = {k: v for k, v in link.items() if k not in ['source', 'target']}
            G.add_edge(source, target, **attrs)

    # Extract agent_start and tasks
    agent_start = data['agent_start']
    tasks = [tuple(task) for task in data['tasks']]

    return G, agent_start, tasks


def load_instances_from_json(folder_path):
    """Loads MAPF instances from JSON files in a folder.

    Reads all JSON files in the specified folder and converts them to MAPF instances
    with graphs, agent start positions, and tasks.

    Args:
        folder_path (str): Path to the folder containing instance JSON files.

    Returns:
        dict: Dictionary mapping filenames (without extension) to instance tuples.
    """

    instances = {}

    # Iterate through all JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            instance = load_instance_from_json(file_path)
            key = os.path.splitext(filename)[0]
            instances[key] = instance

    return instances


def load_benchmark_sets(benchmark_folder):
    """Loads all benchmark sets from a benchmark folder.

    Reads all instance sets from the specified benchmark folder structure,
    organizing them by set name.

    Args:
        benchmark_folder (str): Path to the benchmark folder containing the 'sets' subfolder.

    Returns:
        dict: Dictionary mapping set names to dictionaries of instances.
    """

    sets = dict()

    # get list of set folders
    sets_folder = os.path.join(benchmark_folder, "sets")

    for set_folder in os.listdir(sets_folder):
        if set_folder.endswith('.DS_Store'):
            continue
        set_path = os.path.join(sets_folder, set_folder)
        instances = load_instances_from_json(set_path)
        sets[set_folder] = instances

    return sets


def worker_generate_instance(args):
    """Worker function to generate a single instance in a separate process.

    Args:
        args (tuple): Contains (n_agents, n_vertices, n_tasks, end_time, space_size, gamma1, gamma2, min_agent_separation)

    Returns:
        tuple: (success (bool), instance or None, error_message or None)
    """
    n_agents, n_vertices, n_tasks, end_time, space_size, gamma1, gamma2, min_agent_separation = args

    try:
        # Initialize random seed uniquely for this process
        np.random.seed()  # Reset random seed based on system time in each process
        instance = generate_instance(n_agents, n_vertices, n_tasks, end_time,
                                     space_size=space_size,
                                     gamma1=gamma1,
                                     gamma2=gamma2,
                                     min_agent_separation=min_agent_separation)
        return (True, instance, None)
    except Exception as e:
        return (False, None, str(e))


def generate_instance_set_mp(path, n_instances, n_agents, n_vertices, n_tasks, end_time, space_size=(30, 30),
                             gamma1=0.2, gamma2=0.02, min_agent_separation=2, n_processes=None):
    """Generates and saves a set of MAPF problem instances using multiprocessing.

    Creates multiple problem instances in parallel and saves them as JSON files.

    Args:
        path (str): Directory path where instance files will be saved.
        n_instances (int): Number of instances to generate.
        n_agents (int): Number of agents in each instance.
        n_vertices (int): Number of vertices in each instance graph.
        n_tasks (int): Number of tasks in each instance.
        end_time (float): Maximum time for task generation.
        space_size (tuple, optional): Dimensions of the space. Defaults to (30, 30).
        gamma1 (float, optional): Proportion of extra vertices to generate initially. Defaults to 0.2.
        gamma2 (float, optional): Proportion of random edges to add. Defaults to 0.02.
        min_agent_separation (float, optional): Minimum distance between agents. Defaults to 2.
        n_processes (int, optional): Number of processes to use. Defaults to None (uses CPU count).

    Returns:
        None: Instances are saved to files.
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Determine number of processes
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()

    print(f'Generating {n_instances} instances using {n_processes} processes')

    # Arguments for generating an instance
    instance_args = (n_agents, n_vertices, n_tasks, end_time, space_size, gamma1, gamma2, min_agent_separation)

    # Use a process pool to generate instances in parallel
    instances = []
    with multiprocessing.Pool(processes=n_processes) as pool:
        # Submit initial batch of tasks
        pending_results = []
        for _ in range(min(n_processes, n_instances)):
            pending_results.append(pool.apply_async(worker_generate_instance, (instance_args,)))

        # Process results and add new tasks as needed
        while pending_results and len(instances) < n_instances:
            # Get the first completed result
            result = pending_results.pop(0)
            success, instance, error = result.get()

            if success:
                instances.append(instance)
                print(f'Generated instance {len(instances)}/{n_instances}')
            else:
                print(f'\tFailed to generate an instance: {error}')

            # If we need more instances, submit another task
            if len(instances) + len(pending_results) < n_instances:
                pending_results.append(pool.apply_async(worker_generate_instance, (instance_args,)))

    # Save instances to files
    print(f'Saving {len(instances)} instances to {path}')
    for instance in instances:
        # Use UUID to ensure unique filenames across processes
        file_id = uuid.uuid4()
        file_name = f'{path}/instance_{file_id}.json'
        save_instance_to_json(instance, file_name)


def create_benchmark_set_mp(n_processes=None):
    """Creates a benchmark set of MAPF instances with different parameters using multiprocessing.

    Generates multiple sets of MAPF instances with varying numbers of agents, vertices,
    and tasks, organizing them into a structured folder hierarchy with settings metadata.

    Args:
        n_processes (int, optional): Number of processes per instance set. Defaults to None (uses CPU count).

    Returns:
        str: Path to the created benchmark set.
    """
    # Define and create folders
    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    master_folder = f'Benchmark_Sets/BENCHMARK_{current_datetime}'
    settings_file_path = os.path.join(master_folder, "settings.json")
    sets_folder = os.path.join(master_folder, "sets")
    os.makedirs(master_folder, exist_ok=True)
    os.makedirs(sets_folder, exist_ok=True)

    # Define settings for the instance sets
    settings = dict()
    n_instances = 3
    task_release_end_time = 200
    for vert_factor in [5, 10, 15]:
        for i, n_agents in enumerate([10, 25, 50]):
            n_vertices = n_agents * vert_factor
            task_release_rate = n_agents / 20  # 1 task per agent every x time steps
            n_tasks = int(np.ceil(task_release_rate * task_release_end_time))
            space_side_length = float(3 * np.sqrt(n_vertices))

            settings[f'set_{len(settings)}'] = {
                "n_agents": n_agents,
                "n_vertices": n_vertices,
                "n_tasks": n_tasks,
                "end_time": task_release_end_time,
                "space_size": (space_side_length, space_side_length)
            }

    # Save settings to a JSON file in the master folder
    with open(settings_file_path, "w") as settings_file:
        json.dump(settings, settings_file, indent=4)

    # Generate instance set folders and files
    for label, params in settings.items():
        set_folder = os.path.join(sets_folder, label)
        os.makedirs(set_folder, exist_ok=True)

        print(f'Generating instance set: {label}')
        generate_instance_set_mp(
            path=set_folder,
            n_instances=n_instances,
            n_agents=params["n_agents"],
            n_vertices=params["n_vertices"],
            n_tasks=params["n_tasks"],
            end_time=params["end_time"],
            space_size=params["space_size"],
            n_processes=n_processes
        )

    return master_folder


if __name__ == '__main__':

    # Benchmark parameters are set in the functions below

    # Create a benchmark set (sequential)
    # create_benchmark_set()

    # Create a benchmark set (parallel)
    create_benchmark_set_mp(n_processes=4)


