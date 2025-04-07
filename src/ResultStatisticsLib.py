import json
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def load_planning_stats(filename='planning_stats.json'):
    """
    Load planning statistics from a JSON file.

    Args:
        filename (str): Path to the JSON file containing planning stats.

    Returns:
        dict: The loaded statistics dictionary.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        with open(filename, 'r') as f:
            stats = json.load(f)
        print(f"Successfully loaded stats from {filename}")

        # Optional: Print a summary of the loaded data
        completed = sum(1 for instance in stats.values()
                        if instance.get('Status') == 'Completed')
        failed = sum(1 for instance in stats.values()
                     if instance.get('Status') == 'Failed')
        total = len(stats)

        return stats
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' contains invalid JSON.")
        raise


def create_visualizations(settings_file, results_stats_file, output_folder):
    """Creates visualization plots from multi-agent planning results."""
    os.makedirs(output_folder, exist_ok=True)

    with open(settings_file, 'r') as f:
        settings = json.load(f)
    with open(results_stats_file, 'r') as f:
        results_stats = json.load(f)

    ratio_data = process_data(settings, results_stats)
    ratios = sorted(ratio_data.keys())

    plt.style.use('seaborn-v0_8')
    colors = plt.cm.magma([0.2, 0.5, 0.8])

    plot_computation_time(ratio_data, ratios, colors, output_folder)
    plot_planning_horizon(ratio_data, ratios, colors, output_folder)
    plot_computation_over_horizon(ratio_data, ratios, colors, output_folder)
    plot_throughput_ratio(ratio_data, ratios, colors, output_folder)
    plot_completion_percentage(ratio_data, ratios, colors, output_folder)
    plot_gpp_time(settings, results_stats, output_folder)
    plot_combined_metrics(ratio_data, ratios, colors, output_folder)

    print("Visualizations have been saved to the output folder.")


def process_data(settings, results_stats):
    """Processes raw data into a format suitable for visualization."""
    ratio_data = {}

    for set_key in settings:
        if set_key not in results_stats:
            print(f"Warning: No results for set {set_key}. Skipping.")
            continue

        n_agents = settings[set_key]["n_agents"]
        n_vertices = settings[set_key]["n_vertices"]
        ratio = round(n_vertices / n_agents, 2)

        if ratio not in ratio_data:
            ratio_data[ratio] = {
                'n_agents': [],
                'n_vertices': [],
                'planning_comp_time': {'mean': [], 'std': [], 'min': [], 'max': []},
                'planning_horizon': [],
                'planning_comp_over_horizon': [],
                'throughput': [],
                'throughput_windowed_ratio': [],
                'gpp_time': [],
                'completion_percentage': []
            }

        ratio_data[ratio]['n_agents'].append(n_agents)
        ratio_data[ratio]['n_vertices'].append(n_vertices)

        comp_time = results_stats[set_key]["Planning Computation Time (per call)"]
        ratio_data[ratio]['planning_comp_time']['mean'].append(comp_time["mean"])
        ratio_data[ratio]['planning_comp_time']['std'].append(comp_time["std"])
        ratio_data[ratio]['planning_comp_time']['min'].append(comp_time["min"])
        ratio_data[ratio]['planning_comp_time']['max'].append(comp_time["max"])

        ratio_data[ratio]['planning_horizon'].append(
            results_stats[set_key]["Planning Horizon"]["mean"])
        ratio_data[ratio]['planning_comp_over_horizon'].append(
            results_stats[set_key]["Planning Computation over Horizon"]["mean"])
        ratio_data[ratio]['throughput'].append(
            results_stats[set_key]["Throughput"]["mean"])
        ratio_data[ratio]['throughput_windowed_ratio'].append(
            results_stats[set_key]["Throughput_windowed_ratio"]["mean"])
        ratio_data[ratio]['gpp_time'].append(
            results_stats[set_key]["GPP Time"]["mean"])
        ratio_data[ratio]['completion_percentage'].append(
            results_stats[set_key]["Completion Percentage"])

    return ratio_data


def plot_computation_time(ratio_data, ratios, colors, output_folder):
    """Plots planning computation time per call."""
    plt.figure(figsize=(4, 2))

    for i, ratio in enumerate(ratios):
        idx = sorted(range(len(ratio_data[ratio]['n_agents'])),
                     key=lambda k: ratio_data[ratio]['n_agents'][k])

        n_agents = [ratio_data[ratio]['n_agents'][j] for j in idx]
        planning_comp_time_ms = [ratio_data[ratio]['planning_comp_time']['mean'][j] * 1000
                                 for j in idx]

        plt.plot(
            n_agents,
            planning_comp_time_ms,
            color=colors[i],
            linewidth=2,
            label=f'$\\rho$ = {int(ratio)}',
            marker='o',
            markersize=4
        )

    plt.xlabel("Number of Agents", fontsize=10)
    plt.ylabel("Time (milliseconds)", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(
        output_folder, "Planning_Computation_Time_per_Call.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_planning_horizon(ratio_data, ratios, colors, output_folder):
    """Plots planning horizon for different agent/vertex ratios."""
    plt.figure(figsize=(5, 3))

    for i, ratio in enumerate(ratios):
        idx = sorted(range(len(ratio_data[ratio]['n_agents'])),
                     key=lambda k: ratio_data[ratio]['n_agents'][k])

        n_agents = [ratio_data[ratio]['n_agents'][j] for j in idx]
        data = [ratio_data[ratio]['planning_horizon'][j] for j in idx]

        plt.plot(
            n_agents,
            data,
            marker='o',
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=f'$\\rho$ = {int(ratio)}'
        )

    plt.xlabel("Number of Agents", fontsize=10)
    plt.ylabel("Planning Horizon", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(output_folder, "Planning_Horizon.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_computation_over_horizon(ratio_data, ratios, colors, output_folder):
    """Plots planning computation over horizon."""
    plt.figure(figsize=(5, 3))

    for i, ratio in enumerate(ratios):
        idx = sorted(range(len(ratio_data[ratio]['n_agents'])),
                     key=lambda k: ratio_data[ratio]['n_agents'][k])

        n_agents = [ratio_data[ratio]['n_agents'][j] for j in idx]
        data = [ratio_data[ratio]['planning_comp_over_horizon'][j] for j in idx]

        plt.plot(
            n_agents,
            data,
            marker='o',
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=f'$\\rho$ = {int(ratio)}'
        )

    plt.xlabel("Number of Agents", fontsize=10)
    plt.ylabel("Planning Computation over Horizon", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(
        output_folder, "Planning_Computation_over_Horizon.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_throughput_ratio(ratio_data, ratios, colors, output_folder):
    """Plots throughput windowed ratio."""
    plt.figure(figsize=(4, 2))

    for i, ratio in enumerate(ratios):
        idx = sorted(range(len(ratio_data[ratio]['n_agents'])),
                     key=lambda k: ratio_data[ratio]['n_agents'][k])

        n_agents = [ratio_data[ratio]['n_agents'][j] for j in idx]
        data = [ratio_data[ratio]['throughput_windowed_ratio'][j]*100 for j in idx]

        plt.plot(
            n_agents,
            data,
            marker='o',
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=f'$\\rho$ = {int(ratio)}'
        )

    plt.xlabel("Number of Agents", fontsize=10)
    plt.ylabel("Throughput / Release rate (%)", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(output_folder, "Throughput_windowed_ratio.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_completion_percentage(ratio_data, ratios, colors, output_folder):
    """Plots completion percentage."""
    plt.figure(figsize=(5, 3))

    for i, ratio in enumerate(ratios):
        idx = sorted(range(len(ratio_data[ratio]['n_agents'])),
                     key=lambda k: ratio_data[ratio]['n_agents'][k])

        n_agents = [ratio_data[ratio]['n_agents'][j] for j in idx]
        completion_pct = [ratio_data[ratio]['completion_percentage'][j] for j in idx]

        plt.plot(
            n_agents,
            completion_pct,
            marker='o',
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=f'V/A = {ratio}'
        )

    plt.xlabel("Number of Agents", fontsize=10)
    plt.ylabel("Completion Percentage (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(output_folder, "Completion_Percentage.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_gpp_time(settings, results_stats, output_folder):
    """Plots average GPP time over number of vertices."""
    vertex_to_gpp_times = defaultdict(list)

    for set_label in settings:
        if set_label not in results_stats:
            continue

        n_vertices = settings[set_label]["n_vertices"]
        gpp_time_mean = results_stats[set_label]['GPP Time']['mean']
        vertex_to_gpp_times[n_vertices].append(gpp_time_mean)

    unique_vertices = sorted(vertex_to_gpp_times.keys())
    avg_gpp_times = [sum(vertex_to_gpp_times[v]) / len(vertex_to_gpp_times[v]) / 60  # sec to min
                     for v in unique_vertices]

    plt.figure(figsize=(4, 1.7))
    color = plt.cm.magma(0.5)

    plt.plot(
        unique_vertices,
        avg_gpp_times,
        color=color,
        linewidth=2.5,
        marker='o',
        markersize=6
    )

    plt.xlabel("Number of Vertices", fontsize=10)
    plt.ylabel("Time (minutes)", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_folder, "GPP_Time_over_Vertices.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_combined_metrics(ratio_data, ratios, colors, output_folder):
    """Creates a single figure with 3 subplots for computation time metrics."""
    fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Plot computation time (top subplot)
    for i, ratio in enumerate(ratios):
        idx = sorted(range(len(ratio_data[ratio]['n_agents'])),
                     key=lambda k: ratio_data[ratio]['n_agents'][k])

        n_agents = [ratio_data[ratio]['n_agents'][j] for j in idx]
        planning_comp_time_ms = [ratio_data[ratio]['planning_comp_time']['mean'][j] * 1000
                                 for j in idx]
        planning_comp_time_std_ms = [ratio_data[ratio]['planning_comp_time']['std'][j] * 1000
                                     for j in idx]

        # Plot mean line
        axs[0].plot(
            n_agents,
            planning_comp_time_ms,
            color=colors[i],
            linewidth=1.5,
            label=f'$\\rho$ = {int(ratio)}',
            marker='o',
            markersize=4
        )

        # Add fill between for standard deviation
        """
        axs[0].fill_between(
            n_agents,
            [max(0, mean - std) for mean, std in zip(planning_comp_time_ms, planning_comp_time_std_ms)],
            [mean + std for mean, std in zip(planning_comp_time_ms, planning_comp_time_std_ms)],
            color=colors[i],
            alpha=0.2
        )
        """

    axs[0].set_ylabel("Time (milliseconds)", fontsize=10)
    axs[0].tick_params(axis='both', which='major', labelsize=8)
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)

    # Plot throughput ratio (bottom subplot - now half height)
    for i, ratio in enumerate(ratios):
        idx = sorted(range(len(ratio_data[ratio]['n_agents'])),
                     key=lambda k: ratio_data[ratio]['n_agents'][k])

        n_agents = [ratio_data[ratio]['n_agents'][j] for j in idx]
        data = [ratio_data[ratio]['throughput_windowed_ratio'][j] * 100 for j in idx]

        axs[1].plot(
            n_agents,
            data,
            marker='o',
            color=colors[i],
            linewidth=1.5,
            markersize=4,
            label=f'$\\rho$ = {int(ratio)}'
        )

    axs[1].set_xlabel("Number of Agents", fontsize=10)
    axs[1].set_ylabel("Throughput (%)", fontsize=10)
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    axs[1].legend(fontsize=8)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_folder, "Combined_Metrics.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def compute_and_plot(results_folder, benchmark_folder, output_file_path):

    # Helper function to calculate mean, std, min, and max
    def calculate_stats(data):
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        }

    # Initialize a dictionary to store the benchmarking summary
    benchmarking_summary = {}

    # Iterate over each set folder and calculate statistics
    for set_folder in sorted([folder for folder in os.listdir(results_folder + '/sets') if not folder.startswith('.')]):
        set_folder_path = os.path.join(results_folder, 'sets', set_folder)

        result_files = sorted(os.listdir(set_folder_path))

        # Initialize lists to collect metrics for this set
        gpp_times = []
        planning_times_per_call = []
        planning_horizons = []
        planning_comp_horizon_ratio = []
        throughputs = []
        throughputs_windowed_ratio = []
        completed_instances = 0
        total_instances = len(result_files)

        for result_file in result_files:
            result_file_path = os.path.join(set_folder_path, result_file)

            if not result_file.endswith(".json"):
                continue

            # Load the stats file
            with open(result_file_path, "r") as f:
                result = json.load(f)

            # Process stats
            gpp_times.append(result["GPP"])

            # Check if the instance was completed
            if result["Status"] == "Completed":
                completed_instances += 1

                # Collect planning computation and horizon, but filter out None horizon values
                # (when there were no tasks to plan for)
                for i in range(len(result["Planning computation"])):
                    if result["Planning horizon"][i] is not None:
                        planning_times_per_call.append(result["Planning computation"][i])
                        planning_horizons.append(result["Planning horizon"][i])
                        planning_comp_horizon_ratio.append(
                            result["Planning computation"][i] / result["Planning horizon"][i])

                # Calculate throughput (tasks completed over total time)
                task_completion_times = [
                    completion[0][2] for completion in result["Planner log"]["Task completion"].values()
                ]
                throughput = len(task_completion_times) / max(task_completion_times) if task_completion_times else 0
                throughputs.append(throughput)

                # calculate completion rate over release rate during window 100-200 sec
                tasks_released_in_window = len([task for tasklist in result["Planner log"]["Task completion"].values()
                                                for task in tasklist
                                                if 100 <= task[1] <= 200])
                tasks_completed_in_window = len([task for tasklist in result["Planner log"]["Task completion"].values()
                                                 for task in tasklist
                                                 if 100 <= task[2] <= 200])
                throughputs_windowed_ratio.append(tasks_completed_in_window / tasks_released_in_window)

        # Calculate statistics for this set
        set_summary = {
            "GPP Time": calculate_stats(gpp_times),
            "Completion Percentage": (completed_instances / total_instances) * 100,
            "Planning Computation Time (per call)": calculate_stats(planning_times_per_call),
            "Planning Horizon": calculate_stats(planning_horizons),
            "Planning Computation over Horizon": calculate_stats(planning_comp_horizon_ratio),
            "Throughput": calculate_stats(throughputs),
            "Throughput_windowed_ratio": calculate_stats(throughputs_windowed_ratio),
        }

        # Add to the benchmarking summary
        benchmarking_summary[set_folder] = set_summary

    # Save the benchmarking summary to a JSON file
    with open(output_file_path, "w") as f:
        json.dump(benchmarking_summary, f, indent=4)

    print(f"Benchmarking summary saved to {output_file_path}")

    print('\nCreating visualisations')
    settings_path = os.path.join(benchmark_folder, 'settings.json')
    visualisations_folder_path = os.path.join(results_folder, 'visualisations')
    create_visualizations(settings_path, output_file_path, visualisations_folder_path)


if __name__ == '__main__':

    # Define the folder paths
    results_folder = 'Benchmark_Results/BENCHMARK_all_results'
    benchmark_folder = 'Benchmark_Sets/BENCHMARK_all_sets'
    output_file_path = os.path.join(results_folder, 'stats.json')

    # Compute and plot statistics over a benchmark set's results
    compute_and_plot(results_folder, benchmark_folder, output_file_path)




