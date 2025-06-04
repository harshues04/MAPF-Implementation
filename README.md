# Multi-Agent Path Finding (MAPF) Implementations

A collection of Python implementations for Multi-Agent Path Finding algorithms based on the paper ["Overview of Multi-Agent Path Finding (MAPF)"](https://aaai.org/ocs/index.php/SOCS/SOCS20/paper/view/18510) by Wolfgang Hönig, Jiaoyang Li, and Sven Koenig (Model AI 2020 Assignments).

These implementations solve the MAPF problem for robotic navigation in grid-based environments, demonstrating different algorithms and their applications on various grid configurations.

## Quick Start

### Prerequisites

- Python 3.6 or higher
- Required libraries: `numpy`, `matplotlib`

```bash
pip install numpy matplotlib
```

### Running the Scripts

Each script is self-contained and can be executed independently:

```bash
python join_loc.py
python larger_grid_implementation.py
python paper_implementation.py
```

## File Overview

| Script | Grid Size | Agents | Algorithms | Description |
|--------|-----------|--------|------------|-------------|
| `join_loc.py` | 3×3 | 2 | Joint Location Space, Prioritized Planning, CBS | Direct implementation of paper's Figure 2 example |
| `paper_implementation.py` | 3×3 | 2 | Space-Time A* with Prioritized Planning | Cross-shaped environment from paper |
| `larger_grid_implementation.py` | 10×10 | 5 | Meta-Agent CBS | Extended CBS with obstacles and meta-agents |

## Detailed Descriptions

### `join_loc.py`
**The Complete Algorithm Showcase**

Implements all three core MAPF algorithms on the paper's canonical 3×3 grid example:
- **Joint Location Space** - Explores all possible joint configurations (Section 3)
- **Prioritized Planning** - Sequential agent planning with collision avoidance (Section 4)  
- **Conflict-Based Search (CBS)** - Tree-based conflict resolution (Section 5)

**Features:**
- Two agents navigating from A→E and B→D
- Collision detection and validation
- Performance comparison between algorithms
- Matplotlib visualizations matching paper's Figures 3-5

### `paper_implementation.py`
**Space-Time A* Focus**

Dedicated implementation of the Prioritized Planning approach using Space-Time A*:
- 3×3 grid with cross-shaped free space (white cross on gray background)
- Demonstrates space-time path planning concepts
- Static and animated visualizations
- Direct correspondence to paper's Section 4, Figure 4

### `larger_grid_implementation.py`
**Scalability Demonstration**

Extended CBS implementation showcasing real-world applicability:
- 10×10 grid environment with static obstacles
- Up to 5 agents with coordinated planning
- Meta-Agent CBS approach for complex scenarios
- Both static environment display and animated path execution

## Algorithm Results

Each script produces optimal solutions with cost 5 for the 3×3 examples, including paths like:
- Agent 1: `[A, C, E]` and Agent 2: `[B, B, C, D]`
- Agent 1: `[A, A, C, E]` and Agent 2: `[B, C, D]`

## Output Features

All scripts provide:
- **Console Output**: Problem setup, algorithm progress, solution paths, and performance metrics
- **Static Visualizations**: Grid layout, obstacles, start/goal positions
- **Animated Visualizations**: Real-time path execution (where applicable)
- **Performance Metrics**: Execution times, total costs, makespan analysis

## Performance Notes

- **Joint Location Space**: Limited to ≤3 agents due to exponential complexity
- **CBS and Meta-Agent CBS**: Scales better for larger problems
- **Visualization**: Uses Matplotlib for paper-consistent visual representation
- **Grid Format**: 0 = free cell, 1 = obstacle

## Customization

Easily modify the implementations for your own experiments:
- Adjust grid dimensions and obstacle placement
- Change start/goal positions
- Modify agent groups (in `larger_grid_implementation.py`)
- Experiment with different algorithm parameters

## Paper Correspondence

| Paper Section | Implementation | Key Concepts |
|---------------|----------------|--------------|
| Section 2, Figure 2 | `join_loc.py`, `paper_implementation.py` | MAPF problem definition, 3×3 example |
| Section 3, Figure 3 | `join_loc.py` (Joint Location Space) | State space explosion |
| Section 4, Figure 4 | `paper_implementation.py` | Prioritized planning with Space-Time A* |
| Section 5, Figure 5 | `join_loc.py` (CBS), `larger_grid_implementation.py` | Conflict-based search |

## Example Usage

```python
# Basic execution - outputs paths and visualizations
python join_loc.py

# Expected output includes:
# - Algorithm comparison results
# - Solution paths for both agents
# - Collision analysis
# - Performance timing
# - Matplotlib visualizations
```

## License & Attribution

This project is designed for educational and research purposes. When using in academic or professional work, please provide proper attribution to:
- Original paper authors: Wolfgang Hönig, Jiaoyang Li, and Sven Koenig
- Model AI 2020 Assignments
