
<br>
<h1 align="center">Prioritised Planning for Continuous-time Lifelong Multi-agent Pathfinding</h1>
<br>

<p align="center">
  Alvin Combrink, Sabino Franceso Roselli, and Martin Fabian.
</p>
<br>
<br>
<div align="center">
  <p>
    <img src="https://github.com/user-attachments/assets/ab9d7369-8d4d-49c3-b403-50ad443a0d0b" width="32%" hspace="10">
    <img src="https://github.com/user-attachments/assets/411516e1-1c79-46c5-be17-ca8793d5c2b4" width="32%" hspace="10">
    <img src="https://github.com/user-attachments/assets/5fa743b3-3f2f-42d1-89c3-49c8322423aa" width="32%" hspace="10">
  </p>
</div>

<br> 

This is the official respository for the article **Prioritised Planning for Continuous-time Lifelong Multi-agent Pathfinding**, which describes a fast, sub-optimal planning algorithm for collision-free movements of agents with volumes in continuous-time for the lifelong multi-agent pathfinding problem. The pre-print version is available at [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-B31B1B.svg)](https://arxiv.org/abs/2503.13175), to be submitted to [CoDIT25](https://www.codit2025.org/).

<br> 

## Abstract
_Multi-agent Path Finding (MAPF) is the problem of planning collision-free movements of agents such that they get from where they are to where they need to be. Commonly, agents are located on a graph and can traverse edges. This problem has many variations and has been studied for decades. Two such variations are the continuous-time and the lifelong MAPF problems. In the continuous-time MAPF problem, edges can have non-unit lengths and agents can traverse them at any real-valued time. Additionally, agent volumes are often considered. In the lifelong MAPF problem, agents must attend to a continuous stream of incoming tasks. Much work has been devoted to designing solution methods within these two areas. However, to our knowledge, the combined problem of continuous-time lifelong MAPF has yet to be addressed._

_This work addresses continuous-time lifelong MAPF with agent volumes by presenting the fast and sub-optimal Continuous-time Prioritized Lifelong Planner (CPLP). CPLP continuously re-prioritizes tasks, assigns agents to them, and computes agent plans using a combination of two path planners; one based on CCBS and the other on SIPP. Experimental results with up to 700 agents on graphs with over 10 000 vertices demonstrate average computation times below 80 ms per call, well below a planning horizon of 1 second. In online settings where available time to compute plans is limited, CPLP ensures collision-free movement even when failing to meet these time limits. Therefore, the robustness of CPLP highlights its potential for real-world applications._

<br>

## Repository Structure

```
├── src/                           # Source code for the project
│   ├── Benchmark_Sets/            # Benchmark sets, continaining instances to test the algorithm
│   ├── Benchmark_Results/         # Benchmark results, follows the same file structure as Benchmark_Sets
│   ├── Infeasible_Instances       # Contains infeasible instances found during benchmarking.
│   ├── Plots/                     # Result plots
│   ├── requirements.txt           # Required Python libraries and dependencies
├── README.md                      # Project documentation (this file)
└── LICENSE                        # License information
```

<br>

## Getting Started

### Prerequisites

1. **Python** (version 3.13 or higher)
2. Install required dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Code



<br>

## License

This project is licensed under the MIT License. See `LICENSE` for details.
