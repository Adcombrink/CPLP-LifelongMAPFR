from Planner import CPLP
from GraphPreProcessing import GraphPreProcessor
from ResultStatisticsLib import compute_and_plot
import EnvironmentLib
import VisualiserLib
import time
import datetime
import os
import json
import multiprocessing as mp
import traceback


def run_instance(instance, agent_radius, agent_speed, timelimit=None):
    """Runs a multi-agent path finding instance with online task assignment.

    This function processes a problem instance by:
    1. Preprocessing the graph to compute collision avoidance data
    2. Initializing the planner with agent information
    3. Executing an online planning loop that handles new tasks as they arrive

    Args:
        instance: A tuple containing (graph, agent_start, tasks) where:
            - graph: The environment graph
            - agent_start: Dictionary mapping agent IDs to their starting positions
            - tasks: List of tasks, each with a location and release time
        agent_radius: The physical radius of each agent
        agent_speed: The movement speed of each agent

    Returns:
        A dictionary containing statistics about the planning process:
        - 'GPP': Time spent on graph preprocessing
        - 'Planning': Total planning time
        - 'Status': Final status ('Completed' or 'Failed')
        - 'Planning computation': List of times for each planning cycle
        - 'Planning horizon': List of planning horizons
        - 'Planner log': Detailed log from the planner

    Note:
        The function handles tasks incrementally as they are released over time,
        using a combination of CCBS and SIPP algorithms for collision-free planning.
    """

    graph, agent_start, tasks = instance

    stats = dict()
    stats['GPP'] = None
    stats['Planning'] = None
    stats['Status'] = None
    stats['Planning computation'] = []
    stats['Planning horizon'] = []
    stats['Planner log'] = None

    # preprocess graph
    print('\tPreprocessing graph')
    t0 = time.time()
    gpp = GraphPreProcessor(graph, agent_radius, agent_speed)
    gpp.annotate_graph_with_ctc()
    stats['GPP'] = time.time() - t0
    print(f'\t\tDone ({stats["GPP"]:.6f} s)')

    # plan
    print('\tPlanning')
    t0_planning = time.time()

    print('\t\tInitialising planner')
    agents = set(agent_start.keys())
    planner = CPLP(graph, agents, agent_start, gpp.V_vvc, gpp.V_vec, gpp.E_vec, gpp.E_eec, gpp.dist)

    print('\t\tPerforming online planning')
    i_task = 0
    nr_agents = len(agents)
    curr_time = tasks[0][1]                     # time of first task
    Delta = max(nr_agents ** 1.25, 500) / 1000  # how far in the future new plans start
    while i_task < len(tasks) or planner.task_set:

        if timelimit is not None and time.time() - t0_planning > timelimit:
            raise RuntimeError(f'Timelimit to run instance exceeded, remaining tasks: {planner.task_set}')

        # get new tasks released at time
        new_tasks = set()
        while i_task < len(tasks) and tasks[i_task][1] <= curr_time:
            new_tasks.add(tasks[i_task])
            i_task += 1

        plan_start_time = curr_time + Delta
        try:
            t0 = time.time()
            plan_end_time = planner.plan(plan_start_time, new_tasks)
            stats['Planning computation'].append(time.time() - t0)
            stats['Planning horizon'].append(
                plan_end_time - plan_start_time if plan_end_time is not None else None)
        except:
            print('\t\tFailed')
            stats['Status'] = 'Failed'
            break


        if plan_end_time is not None:
            next_curr_time = plan_end_time - Delta
        else:

            # if planner returns "planning done" but there are still tasks left, Failed
            if planner.task_set:
                stats['Status'] = 'Failed'
                break

            # planner is done with all tasks, if there are still tasks to release, do so, else done
            if i_task < len(tasks):
                next_curr_time = tasks[i_task][1]
            else:
                stats['Status'] = 'Completed'
                break

        curr_time = next_curr_time

    stats['Planning'] = time.time() - t0_planning
    print(f'\t\tDone ({stats["Planning"]:.6f} s) Remaining tasks {len(planner.task_set)}')
    stats['Planner log'] = planner.log

    return stats, planner


def run_benchmark_set(benchmark_label):
    """Runs benchmarks on a set of MAPF instances and saves results.

    Executes all instances in a benchmark set, collecting performance statistics
    and saving them to a timestamped results folder.

    Args:
        benchmark_label (str): Label of the benchmark set to run.

    Returns:
        None: Results are saved to files in the output folder.
    """

    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Run benchmarks
    benchmark_folder = os.path.join('Benchmark_Sets', benchmark_label)
    output_folder = os.path.join('Benchmark_Results', benchmark_label + f'_{current_datetime}')
    os.makedirs(output_folder, exist_ok=True)

    # Get instance paths
    sets_folder = os.path.join(benchmark_folder, "sets")
    sets = dict()
    for set_folder in os.listdir(sets_folder):
        if set_folder.endswith('.DS_Store'):
            continue

        sets[set_folder] = dict()

        set_path = os.path.join(sets_folder, set_folder)
        for instance_label in os.listdir(set_path):
            if instance_label.endswith('.DS_Store'):
                continue
            sets[set_folder][instance_label] = os.path.join(set_path, instance_label)

    for i_set, set_label in enumerate(sets):
        print(f'Running \t{set_label} ({i_set+1}/{len(sets)}) ########################')
        set_folder = os.path.join(output_folder, 'sets', set_label)
        os.makedirs(set_folder, exist_ok=True)

        for i_instance, (instance_label, instance_path) in enumerate(sets[set_label].items()):
            print(f'\t{instance_label} ({i_instance+1}/{len(sets[set_label])}) ---------------------')

            # load and run instance
            instance = EnvironmentLib.load_instance_from_json(instance_path)
            stats, _ = run_instance(instance, agent_radius=1, agent_speed=1)

            # save instance
            with open(os.path.join(set_folder, f"{instance_label}"), "w") as stats_file:
                json.dump(stats, stats_file, indent=4)

    print('Done')


def worker_run_instance(task):
    """Worker function for running a single instance in a separate process."""

    timelimit = None  # Total runtime time limit (to catch potentially infeasible instances).
    # Any triggering instance must be checked manually

    set_folder = task['set_folder']
    instance_set_label = task['instance_set_label']
    instance_label = task['instance_label']
    instance = task['instance']
    i_set = task['i_set']
    len_benchmark_sets = task['len_benchmark_sets']
    i_instance = task['i_instance']
    len_instance_set = task['len_instance_set']

    process_id = os.getpid()
    print(
        f"Process-{process_id} running ({i_set + 1}/{len_benchmark_sets}, {i_instance + 1}/{len_instance_set}): {instance_set_label} {instance_label}")

    try:
        start_time = time.time()
        stats, _ = run_instance(instance, agent_radius=1, agent_speed=1, timelimit=timelimit)
        elapsed_time = time.time() - start_time

        # Save the results to a file
        with open(os.path.join(set_folder, f"{instance_label}.json"), "w") as stats_file:
            json.dump(stats, stats_file, indent=4)

        return True, f"Completed: {instance_set_label} {instance_label} in {elapsed_time:.2f} seconds"
    except Exception as e:
        error_trace = traceback.format_exc()
        return False, f"Failed: {instance_set_label} {instance_label}, Error: {str(e)}\n{error_trace}"


def run_benchmark_set_parallel(benchmark_label, num_processes=None):
    """Runs benchmarks on a set of MAPF instances in parallel and saves results.

    Executes instances in a benchmark set using multiprocessing to run multiple
    instances simultaneously, collecting performance statistics and saving them
    to a timestamped results folder.

    Args:
        benchmark_label (str): Label of the benchmark set to run.
        num_processes (int, optional): Number of processes to use. Defaults to CPU count.

    Returns:
        str: Path to the output folder containing results.
    """
    # If num_processes is not specified, use the number of CPU cores
    if num_processes is None:
        num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes for parallel execution")

    # Folders
    current_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    benchmark_folder = os.path.join('Benchmark_Sets', benchmark_label)
    output_folder = os.path.join('Benchmark_Results', benchmark_label + f'_{current_datetime}')
    os.makedirs(output_folder, exist_ok=True)

    # Load benchmark sets
    benchmark_sets = EnvironmentLib.load_benchmark_sets(benchmark_folder)

    # Create sets directories in advance
    for instance_set_label in benchmark_sets.keys():
        set_folder = os.path.join(output_folder, 'sets', instance_set_label)
        os.makedirs(set_folder, exist_ok=True)

    # Create all tasks
    all_tasks = []
    for i_set, (instance_set_label, instance_set) in enumerate(benchmark_sets.items()):
        set_folder = os.path.join(output_folder, 'sets', instance_set_label)

        for i_instance, (instance_label, instance) in enumerate(instance_set.items()):
            task = {
                'set_folder': set_folder,
                'instance_set_label': instance_set_label,
                'instance_label': instance_label,
                'instance': instance,
                'i_set': i_set,
                'len_benchmark_sets': len(benchmark_sets),
                'i_instance': i_instance,
                'len_instance_set': len(instance_set)
            }
            all_tasks.append(task)

    total_instances = len(all_tasks)
    print(f"Total instances to process: {total_instances}")

    # Start timing the parallel execution
    start_time = time.time()

    # Run tasks in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(worker_run_instance, all_tasks)

    # Calculate total execution time
    total_time = time.time() - start_time

    # Print a summary of results
    successes = sum(1 for success, _ in results if success)
    failures = sum(1 for success, _ in results if not success)

    print(f'\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX SUMMARY XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')
    print(f"Benchmark completed in {total_time:.2f} seconds")
    print(f"Summary: {successes} succeeded, {failures} failed out of {total_instances}")
    print(f'Results saved in {output_folder}')

    # Print any failure messages
    if failures > 0:
        print("\nFailures:")
        for success, message in results:
            if not success:
                print(f"  {message}")

    return output_folder


if __name__ == '__main__':

    # Generate a benchmark set (See EnvironmentLib file for more details)
    # EnvironmentLib.create_benchmark_set_mp(n_processes=2)

    # Run a benchmark set (sequential)
    # run_benchmark_set('BENCHMARK_20250407_085740')

    # Run a benchmark set (parallel)
    # run_benchmark_set_parallel('BENCHMARK_20250407_085740', num_processes=3)

    # Load, plot, run, and animate a single instance
    """
    g, agent_start, tasks = EnvironmentLib.load_instance_from_json(
        'Benchmark_Sets/BENCHMARK_20250407_085740/sets/set_0/instance_38ad8a67-f3ad-458c-934d-5085d3fc0a08.json')
    VisualiserLib.plot_graph(g, agent_start, 1)
    stats, planner = run_instance((g, agent_start, tasks), 1, 1)
    VisualiserLib.animate_MAPF(g, planner, 1, 1, 10, 10)
    """

    # Plot results
    """
    results_folder = 'Benchmark_Results/BENCHMARK_20250407_085740_20250407_090329'
    benchmark_folder = 'Benchmark_Sets/BENCHMARK_20250407_085740'
    output_file_path = os.path.join(results_folder, 'stats.json')
    compute_and_plot(results_folder, benchmark_folder, output_file_path)
    """