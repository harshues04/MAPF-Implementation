import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Optional
from copy import deepcopy
from itertools import product
import time

# Type Definitions
Position = Tuple[int, int]
Node = Tuple[int, int, int]  
Path = List[Node]
Solution = List[Path]
Constraint = Dict[str, any]

# Utility Functions
def manhattan_distance(pos1: Position, pos2: Position) -> int:
    """Compute Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_neighbors(pos: Position, grid: np.ndarray) -> List[Position]:
    """Return valid neighboring positions including wait action."""
    x, y = pos
    neighbors = []
    
    # Cardinal directions: up, down, left, right
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 
            grid[nx, ny] == 0):  # 0 = unblocked
            neighbors.append((nx, ny))
    
    # Wait action (stay in current position)
    if grid[x, y] == 0:
        neighbors.append((x, y))
    
    return neighbors

def detect_collisions(solution: Solution) -> List[Dict]:
    """Detect vertex and edge collisions in a solution."""
    collisions = []
    
    if not solution or len(solution) < 2:
        return collisions
    
    # Get maximum time across all paths
    max_time = max(len(path) for path in solution)
    
    # Check each time step
    for t in range(max_time):
        # Get positions of all agents at time t
        positions = {}
        for agent_id, path in enumerate(solution):
            if t < len(path):
                pos = (path[t][0], path[t][1])
                positions[agent_id] = pos
            else:
                # Agent has reached goal, stays there
                pos = (path[-1][0], path[-1][1])
                positions[agent_id] = pos
        
        # Check vertex collisions
        pos_to_agents = {}
        for agent_id, pos in positions.items():
            if pos not in pos_to_agents:
                pos_to_agents[pos] = []
            pos_to_agents[pos].append(agent_id)
        
        for pos, agents in pos_to_agents.items():
            if len(agents) > 1:
                collisions.append({
                    'type': 'vertex',
                    'agents': agents,
                    'position': pos,
                    'time': t
                })
        
        # Check edge collisions (if not at time 0)
        if t > 0:
            prev_positions = {}
            for agent_id, path in enumerate(solution):
                if t-1 < len(path):
                    prev_pos = (path[t-1][0], path[t-1][1])
                else:
                    prev_pos = (path[-1][0], path[-1][1])
                prev_positions[agent_id] = prev_pos
            
            # Check if agents swap positions
            for agent1 in range(len(solution)):
                for agent2 in range(agent1 + 1, len(solution)):
                    pos1_prev = prev_positions[agent1]
                    pos1_curr = positions[agent1]
                    pos2_prev = prev_positions[agent2]
                    pos2_curr = positions[agent2]
                    
                    if (pos1_prev == pos2_curr and pos2_prev == pos1_curr and 
                        pos1_prev != pos1_curr):  # Actual swap, not both waiting
                        collisions.append({
                            'type': 'edge',
                            'agents': [agent1, agent2],
                            'positions': [pos1_prev, pos1_curr],
                            'time': t
                        })
    
    return collisions

def space_time_astar(grid: np.ndarray, start: Position, goal: Position,
                     constraints: List[Constraint], agent_id: int, 
                     max_time: int = 100) -> Optional[Path]:
    """Find shortest path in space-time with constraints."""
    
    # Priority queue: (f_score, g_score, node)
    open_set = []
    heappush(open_set, (manhattan_distance(start, goal), 0, (start[0], start[1], 0)))
    
    # g_scores: node -> cost from start
    g_scores = {(start[0], start[1], 0): 0}
    
    # came_from: node -> parent node
    came_from = {}
    
    # closed_set
    closed_set = set()
    
    while open_set:
        f_score, g_score, current = heappop(open_set)
        x, y, t = current
        
        if (x, y, t) in closed_set:
            continue
            
        closed_set.add((x, y, t))
        
        # Check if reached goal
        if (x, y) == goal:
            # Reconstruct path
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append((start[0], start[1], 0))
            return list(reversed(path))
        
        if t >= max_time:
            continue
        
        # Generate neighbors
        neighbors = get_neighbors((x, y), grid)
        
        for nx, ny in neighbors:
            next_node = (nx, ny, t + 1)
            
            # Check constraints
            if violates_constraints(current, next_node, constraints, agent_id):
                continue
            
            if next_node in closed_set:
                continue
            
            tentative_g = g_score + 1
            
            if next_node not in g_scores or tentative_g < g_scores[next_node]:
                g_scores[next_node] = tentative_g
                came_from[next_node] = current
                f_score = tentative_g + manhattan_distance((nx, ny), goal)
                heappush(open_set, (f_score, tentative_g, next_node))
    
    return None

def violates_constraints(current: Node, next_node: Node, 
                        constraints: List[Constraint], agent_id: int) -> bool:
    """Check if a move violates any constraints."""
    for constraint in constraints:
        if constraint['agent'] != agent_id:
            continue
        
        if constraint['type'] == 'vertex':
            if (next_node[0] == constraint['x'] and 
                next_node[1] == constraint['y'] and 
                next_node[2] == constraint['t']):
                return True
        
        elif constraint['type'] == 'edge':
            if (current[0] == constraint['x1'] and 
                current[1] == constraint['y1'] and
                next_node[0] == constraint['x2'] and 
                next_node[1] == constraint['y2'] and 
                next_node[2] == constraint['t']):
                return True
    
    return False

# Algorithm Functions
def joint_location_space(grid: np.ndarray, agents: List[Dict]) -> Optional[Solution]:
    """Find optimal solution by searching joint location space."""
    print("Running Joint Location Space algorithm...")
    
    if len(agents) > 3:  # Limit due to exponential complexity
        print("Warning: Joint Location Space limited to 3 agents due to complexity")
        return None
    
    start_state = tuple((agent['start'][0], agent['start'][1]) for agent in agents)
    goal_state = tuple((agent['goal'][0], agent['goal'][1]) for agent in agents)
    
    # Priority queue: (f_score, g_score, state, time, path_history)
    open_set = []
    start_heuristic = sum(manhattan_distance(agents[i]['start'], agents[i]['goal']) 
                         for i in range(len(agents)))
    heappush(open_set, (start_heuristic, 0, start_state, 0, [start_state]))
    
    # Visited states with time
    visited = set()
    
    max_time = 50  # Limit search depth
    
    while open_set:
        f_score, g_score, state, time, path_history = heappop(open_set)
        
        state_time = (state, time)
        if state_time in visited:
            continue
        visited.add(state_time)
        
        # Check if reached goal
        if state == goal_state:
            # Convert path_history to individual agent paths
            solution = [[] for _ in range(len(agents))]
            for t, joint_state in enumerate(path_history):
                for agent_id, pos in enumerate(joint_state):
                    solution[agent_id].append((pos[0], pos[1], t))
            return solution
        
        if time >= max_time:
            continue
        
        # Generate all possible joint moves
        all_moves = []
        for i, agent in enumerate(agents):
            current_pos = state[i]
            agent_moves = get_neighbors(current_pos, grid)
            all_moves.append(agent_moves)
        
        # Generate cartesian product of all moves
        for joint_move in product(*all_moves):
            # Check for collisions in this joint move
            if has_collision(joint_move, state):
                continue
            
            new_state = tuple(joint_move)
            new_path_history = path_history + [new_state]
            
            # Calculate costs
            new_g = g_score + count_non_wait_moves(state, new_state)
            new_h = sum(manhattan_distance(new_state[i], (agents[i]['goal'][0], agents[i]['goal'][1])) 
                       for i in range(len(agents)))
            new_f = new_g + new_h
            
            heappush(open_set, (new_f, new_g, new_state, time + 1, new_path_history))
    
    return None

def has_collision(joint_move: Tuple[Position, ...], prev_state: Tuple[Position, ...]) -> bool:
    """Check if a joint move has vertex or edge collisions."""
    # Vertex collision: same position
    if len(set(joint_move)) != len(joint_move):
        return True
    
    # Edge collision: agents swap positions
    for i in range(len(joint_move)):
        for j in range(i + 1, len(joint_move)):
            if (joint_move[i] == prev_state[j] and 
                joint_move[j] == prev_state[i] and
                joint_move[i] != prev_state[i]):  # Actual swap
                return True
    
    return False

def count_non_wait_moves(prev_state: Tuple[Position, ...], 
                        new_state: Tuple[Position, ...]) -> int:
    """Count number of agents that actually moved (non-wait actions)."""
    return sum(1 for i in range(len(prev_state)) if prev_state[i] != new_state[i])

def prioritized_planning(grid: np.ndarray, agents: List[Dict]) -> Optional[Solution]:
    """Plan paths sequentially with priority ordering."""
    print("Running Prioritized Planning algorithm...")
    
    solution = []
    all_constraints = []
    
    for agent in agents:
        print(f"  Planning for Agent {agent['id']}...")
        
        # Find path for current agent avoiding all previous paths
        path = space_time_astar(grid, agent['start'], agent['goal'], 
                               all_constraints, agent['id'])
        
        if path is None:
            print(f"  Failed to find path for Agent {agent['id']}")
            return None
        
        solution.append(path)
        
        # Add constraints from this path for future agents
        path_constraints = generate_path_constraints(path, agent['id'])
        all_constraints.extend(path_constraints)
    
    return solution

def generate_path_constraints(path: Path, agent_id: int) -> List[Constraint]:
    """Generate vertex and edge constraints from a path."""
    constraints = []
    
    # Vertex constraints
    for x, y, t in path:
        constraints.append({
            'type': 'vertex',
            'agent': 'other',  # Will be checked against other agents
            'x': x, 'y': y, 't': t
        })
    
    # Edge constraints
    for i in range(len(path) - 1):
        x1, y1, t1 = path[i]
        x2, y2, t2 = path[i + 1]
        constraints.append({
            'type': 'edge',
            'agent': 'other',
            'x1': x2, 'y1': y2,  # Reverse edge
            'x2': x1, 'y2': y1,
            't': t2
        })
    
    return constraints

class ConstraintTreeNode:
    """Node in the CBS constraint tree."""
    def __init__(self, constraints: List[Constraint] = None, 
                 solution: Solution = None, cost: int = 0):
        self.constraints = constraints or []
        self.solution = solution or []
        self.cost = cost
    
    def __lt__(self, other):
        return self.cost < other.cost

def conflict_based_search(grid: np.ndarray, agents: List[Dict]) -> Optional[Solution]:
    """Find optimal solution using Conflict-Based Search."""
    print("Running Conflict-Based Search algorithm...")
    
    # Initialize root node with unconstrained shortest paths
    root_constraints = []
    root_solution = []
    
    for agent in agents:
        path = space_time_astar(grid, agent['start'], agent['goal'], 
                               root_constraints, agent['id'])
        if path is None:
            print("  Failed to find initial path for agent", agent['id'])
            return None
        root_solution.append(path)
    
    root_cost = sum(len(path) for path in root_solution)
    root_node = ConstraintTreeNode(root_constraints, root_solution, root_cost)
    
    # Priority queue for constraint tree nodes
    open_set = [root_node]
    
    iteration = 0
    max_iterations = 1000
    
    while open_set and iteration < max_iterations:
        iteration += 1
        current_node = heappop(open_set)
        
        # Check for collisions
        collisions = detect_collisions(current_node.solution)
        
        if not collisions:
            print(f"  CBS found solution in {iteration} iterations")
            return current_node.solution
        
        # Pick first collision to resolve
        collision = collisions[0]
        
        if collision['type'] == 'vertex':
            # Create two child nodes with vertex constraints
            agents_in_collision = collision['agents']
            pos = collision['position']
            time = collision['time']
            
            for agent_id in agents_in_collision:
                child_constraints = deepcopy(current_node.constraints)
                child_constraints.append({
                    'type': 'vertex',
                    'agent': agent_id,
                    'x': pos[0], 'y': pos[1], 't': time
                })
                
                # Replan path for constrained agent
                new_solution = deepcopy(current_node.solution)
                agent_info = agents[agent_id]
                new_path = space_time_astar(grid, agent_info['start'], 
                                          agent_info['goal'], child_constraints, agent_id)
                
                if new_path is not None:
                    new_solution[agent_id] = new_path
                    new_cost = sum(len(path) for path in new_solution)
                    child_node = ConstraintTreeNode(child_constraints, new_solution, new_cost)
                    heappush(open_set, child_node)
        
        elif collision['type'] == 'edge':
            # Create two child nodes with edge constraints
            agents_in_collision = collision['agents']
            positions = collision['positions']
            time = collision['time']
            
            for i, agent_id in enumerate(agents_in_collision):
                child_constraints = deepcopy(current_node.constraints)
                pos1, pos2 = positions
                child_constraints.append({
                    'type': 'edge',
                    'agent': agent_id,
                    'x1': pos1[0], 'y1': pos1[1],
                    'x2': pos2[0], 'y2': pos2[1],
                    't': time
                })
                
                # Replan path for constrained agent
                new_solution = deepcopy(current_node.solution)
                agent_info = agents[agent_id]
                new_path = space_time_astar(grid, agent_info['start'], 
                                          agent_info['goal'], child_constraints, agent_id)
                
                if new_path is not None:
                    new_solution[agent_id] = new_path
                    new_cost = sum(len(path) for path in new_solution)
                    child_node = ConstraintTreeNode(child_constraints, new_solution, new_cost)
                    heappush(open_set, child_node)
    
    print(f"  CBS failed to find solution after {iteration} iterations")
    return None

# Visualization
def visualize_paths(grid: np.ndarray, solution: Solution, title: str):
    """Visualize the grid and agent paths."""
    if solution is None:
        print(f"No solution to visualize for {title}")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Display grid
    plt.imshow(grid, cmap='gray_r', alpha=0.3)
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for agent_id, path in enumerate(solution):
        if not path:
            continue
        
        # Extract x, y coordinates
        x_coords = [node[0] for node in path]
        y_coords = [node[1] for node in path]
        
        color = colors[agent_id % len(colors)]
        
        # Plot path
        plt.plot(y_coords, x_coords, color=color, linewidth=2, 
                marker='o', markersize=6, label=f'Agent {agent_id + 1}')
        
        # Mark start and goal
        plt.plot(y_coords[0], x_coords[0], color=color, marker='s', 
                markersize=10, markerfacecolor='lightgreen')
        plt.plot(y_coords[-1], x_coords[-1], color=color, marker='*', 
                markersize=12, markerfacecolor='lightcoral')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('Y coordinate')
    plt.ylabel('X coordinate')
    plt.show()

def print_solution(solution: Solution, algorithm_name: str):
    """Print solution details."""
    if solution is None:
        print(f"{algorithm_name}: No solution found")
        return
    
    print(f"\n{algorithm_name} Solution:")
    total_cost = 0
    
    for agent_id, path in enumerate(solution):
        print(f"  Agent {agent_id + 1}: {path}")
        total_cost += len(path)
    
    print(f"  Total cost: {total_cost}")
    
    # Check for collisions
    collisions = detect_collisions(solution)
    if collisions:
        print(f"  WARNING: {len(collisions)} collisions detected!")
        for collision in collisions:
            print(f"    {collision}")
    else:
        print("  No collisions detected ✓")

# Main Program
def main():
    """Main program to test all MAPF algorithms."""
    print("Multi-Agent Path Finding (MAPF) Implementation")
    print("=" * 50)
    
    # Test case from the paper: 3x3 grid
    grid = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    
    agents = [
        {"id": 1, "start": (0, 0), "goal": (2, 2)},  # Agent 1: A to E
        {"id": 2, "start": (0, 1), "goal": (2, 1)}   # Agent 2: B to D
    ]
    
    print(f"Grid shape: {grid.shape}")
    print(f"Number of agents: {len(agents)}")
    print(f"Agents: {agents}")
    print()
    
    algorithms = [
        ("Joint Location Space", joint_location_space),
        ("Prioritized Planning", prioritized_planning),
        ("Conflict-Based Search", conflict_based_search)
    ]
    
    results = {}
    
    for name, algorithm in algorithms:
        print(f"Testing {name}...")
        start_time = time.time()
        
        try:
            solution = algorithm(grid, agents)
            end_time = time.time()
            
            results[name] = {
                'solution': solution,
                'time': end_time - start_time
            }
            
            print_solution(solution, name)
            print(f"  Execution time: {end_time - start_time:.4f} seconds")
            
            # Visualize solution
            visualize_paths(grid, solution, name)
            
        except Exception as e:
            print(f"  Error in {name}: {e}")
            results[name] = {'solution': None, 'time': 0, 'error': str(e)}
        
        print("-" * 40)
    
    # Summary
    print("\nSUMMARY:")
    print("=" * 50)
    for name, result in results.items():
        if result['solution'] is not None:
            cost = sum(len(path) for path in result['solution'])
            collisions = len(detect_collisions(result['solution']))
            print(f"{name}:")
            print(f"  Cost: {cost}, Collisions: {collisions}, Time: {result['time']:.4f}s")
        else:
            print(f"{name}: Failed")

if __name__ == "__main__":
    main()