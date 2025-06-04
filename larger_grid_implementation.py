import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import heapq
from collections import defaultdict
import copy

class Constraint:
    def __init__(self, agent, location, time_step, constraint_type='vertex'):
        self.agent = agent
        self.location = location
        self.time_step = time_step
        self.type = constraint_type  # 'vertex' or 'edge'
    
    def __str__(self):
        return f"Constraint(agent={self.agent}, location={self.location}, time={self.time_step}, type={self.type})"

class CBSNode:
    def __init__(self):
        self.constraints = []
        self.solution = {}
        self.cost = 0
        
    def __lt__(self, other):
        return self.cost < other.cost

class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = len(grid), len(grid[0])
    
    def heuristic(self, pos, goal):
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions including staying in place"""
        neighbors = []
        row, col = pos
        
        # Stay in place
        neighbors.append((row, col))
        
        # Move in 4 directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.rows and 0 <= new_col < self.cols and 
                self.grid[new_row][new_col] == 0):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def is_constrained(self, agent, pos, time_step, constraints):
        """Check if the position at time_step is constrained for the agent"""
        for constraint in constraints:
            if constraint.agent == agent and constraint.time_step == time_step:
                if constraint.type == 'vertex' and constraint.location == pos:
                    return True
                elif constraint.type == 'edge' and len(constraint.location) == 2:
                    # Edge constraint: check if moving from location[0] to location[1]
                    if constraint.location[1] == pos:
                        return True
        return False
    
    def find_path(self, start, goal, agent, constraints, max_time=100):
        """Find path for a single agent with given constraints"""
        # Priority queue: (f_cost, g_cost, time, position, path)
        open_list = [(self.heuristic(start, goal), 0, 0, start, [start])]
        closed_set = set()
        
        while open_list:
            f_cost, g_cost, time, current_pos, path = heapq.heappop(open_list)
            
            # Goal reached
            if current_pos == goal:
                return path
            
            state = (time, current_pos)
            if state in closed_set:
                continue
            closed_set.add(state)
            
            if time >= max_time:
                continue
            
            # Explore neighbors
            for next_pos in self.get_neighbors(current_pos):
                new_time = time + 1
                new_state = (new_time, next_pos)
                
                if new_state in closed_set:
                    continue
                
                # Check if this move violates any constraints
                if self.is_constrained(agent, next_pos, new_time, constraints):
                    continue
                
                # Check edge constraints
                edge_constrained = False
                for constraint in constraints:
                    if (constraint.agent == agent and 
                        constraint.type == 'edge' and 
                        constraint.time_step == new_time and
                        len(constraint.location) == 2 and
                        constraint.location[0] == current_pos and 
                        constraint.location[1] == next_pos):
                        edge_constrained = True
                        break
                
                if edge_constrained:
                    continue
                
                new_g_cost = g_cost + 1
                new_f_cost = new_g_cost + self.heuristic(next_pos, goal)
                new_path = path + [next_pos]
                
                heapq.heappush(open_list, 
                             (new_f_cost, new_g_cost, new_time, next_pos, new_path))
        
        return None  # No path found

class MetaAgentCBS:
    def __init__(self, grid, starts, goals, meta_agent_groups=None):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.rows, self.cols = len(grid), len(grid[0])
        self.num_agents = len(starts)
        self.astar = AStar(grid)
        
        # Define meta-agent groups (if not provided, each agent is its own meta-agent)
        if meta_agent_groups is None:
            self.meta_agent_groups = [[i] for i in range(self.num_agents)]
        else:
            self.meta_agent_groups = meta_agent_groups
        
        print(f"Meta-agent groups: {self.meta_agent_groups}")
    
    def detect_conflict(self, solution):
        """Detect the first conflict in the solution"""
        # Get maximum path length
        max_len = max(len(path) for path in solution.values())
        
        # Extend all paths to same length (agents stay at goal)
        extended_paths = {}
        for agent, path in solution.items():
            extended_path = path + [path[-1]] * (max_len - len(path))
            extended_paths[agent] = extended_path
        
        # Check for vertex conflicts
        for t in range(max_len):
            positions = {}
            for agent, path in extended_paths.items():
                pos = path[t]
                if pos in positions:
                    return {
                        'type': 'vertex',
                        'agents': [positions[pos], agent],
                        'location': pos,
                        'time': t
                    }
                positions[pos] = agent
        
        # Check for edge conflicts (swapping)
        for t in range(max_len - 1):
            for agent1 in extended_paths:
                for agent2 in extended_paths:
                    if agent1 >= agent2:
                        continue
                    
                    pos1_t = extended_paths[agent1][t]
                    pos1_t1 = extended_paths[agent1][t + 1]
                    pos2_t = extended_paths[agent2][t]
                    pos2_t1 = extended_paths[agent2][t + 1]
                    
                    # Edge conflict: agents swap positions
                    if pos1_t == pos2_t1 and pos1_t1 == pos2_t:
                        return {
                            'type': 'edge',
                            'agents': [agent1, agent2],
                            'location': [pos1_t, pos1_t1],
                            'time': t + 1
                        }
        
        return None
    
    def get_meta_agent_for_agent(self, agent):
        """Get which meta-agent group contains the given agent"""
        for i, group in enumerate(self.meta_agent_groups):
            if agent in group:
                return i
        return None
    
    def solve_meta_agent(self, meta_agent_id, constraints):
        """Solve for a single meta-agent (group of agents) with constraints"""
        agents_in_group = self.meta_agent_groups[meta_agent_id]
        
        if len(agents_in_group) == 1:
            # Single agent - use A*
            agent = agents_in_group[0]
            path = self.astar.find_path(self.starts[agent], self.goals[agent], 
                                      agent, constraints)
            if path:
                return {agent: path}
            else:
                return None
        else:
            # Multiple agents - use Space-Time A* for the group
            # This is a simplified version - in practice, you might want more sophisticated planning
            group_solution = {}
            group_constraints = constraints.copy()
            
            # Plan for each agent in the group sequentially
            for agent in agents_in_group:
                path = self.astar.find_path(self.starts[agent], self.goals[agent], 
                                          agent, group_constraints)
                if path is None:
                    return None
                group_solution[agent] = path
                
                # Add constraints for other agents in the group based on this path
                for t, pos in enumerate(path):
                    for other_agent in agents_in_group:
                        if other_agent != agent and other_agent not in group_solution:
                            constraint = Constraint(other_agent, pos, t, 'vertex')
                            group_constraints.append(constraint)
            
            return group_solution
    
    def solve(self):
        """Solve MAPF using Meta-Agent CBS"""
        print("Starting Meta-Agent CBS...")
        
        # Root node
        root = CBSNode()
        
        # Find initial solution for each meta-agent
        for meta_agent_id in range(len(self.meta_agent_groups)):
            meta_solution = self.solve_meta_agent(meta_agent_id, root.constraints)
            if meta_solution is None:
                print(f"No initial solution found for meta-agent {meta_agent_id}")
                return None
            root.solution.update(meta_solution)
        
        # Calculate initial cost
        root.cost = sum(len(path) - 1 for path in root.solution.values())
        
        # Priority queue for CBS tree
        open_list = [root]
        
        iteration = 0
        while open_list:
            iteration += 1
            if iteration > 1000:  # Safety limit
                print("CBS iteration limit reached")
                break
                
            # Get node with lowest cost
            current_node = heapq.heappop(open_list)
            
            print(f"CBS iteration {iteration}, cost: {current_node.cost}")
            
            # Check for conflicts
            conflict = self.detect_conflict(current_node.solution)
            
            if conflict is None:
                # No conflicts - solution found!
                print(f"Solution found after {iteration} iterations")
                return current_node.solution
            
            print(f"Conflict detected: {conflict}")
            
            # Create child nodes with new constraints
            conflicting_agents = conflict['agents']
            
            for agent in conflicting_agents:
                # Create new node
                child_node = CBSNode()
                child_node.constraints = current_node.constraints.copy()
                child_node.solution = current_node.solution.copy()
                
                # Add constraint for this agent
                if conflict['type'] == 'vertex':
                    constraint = Constraint(agent, conflict['location'], 
                                          conflict['time'], 'vertex')
                else:  # edge conflict
                    constraint = Constraint(agent, conflict['location'], 
                                          conflict['time'], 'edge')
                
                child_node.constraints.append(constraint)
                
                # Replan for the meta-agent containing this agent
                meta_agent_id = self.get_meta_agent_for_agent(agent)
                meta_solution = self.solve_meta_agent(meta_agent_id, child_node.constraints)
                
                if meta_solution is not None:
                    # Update solution with new paths for this meta-agent
                    for meta_agent in self.meta_agent_groups[meta_agent_id]:
                        child_node.solution[meta_agent] = meta_solution[meta_agent]
                    
                    # Calculate new cost
                    child_node.cost = sum(len(path) - 1 for path in child_node.solution.values())
                    
                    # Add to open list
                    heapq.heappush(open_list, child_node)
        
        print("No solution found")
        return None

class MAPFVisualizer:
    def __init__(self, grid, starts, goals, paths=None):
        self.grid = np.array(grid)
        self.starts = starts
        self.goals = goals
        self.paths = paths
        self.rows, self.cols = self.grid.shape
        
        # Set up the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 8))
        self.fig.suptitle('Multi-Agent Path Finding: Meta-Agent CBS', fontsize=16, fontweight='bold')
        
        # Colors for agents
        self.agent_colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow', 'lime']
        
        self.setup_static_plot()
        if paths:
            self.setup_animation()
    
    def setup_static_plot(self):
        """Setup the static environment plot"""
        self.ax1.set_title('10x10 Grid Environment Layout', fontweight='bold')
        self.ax1.set_xlim(-0.5, self.cols - 0.5)
        self.ax1.set_ylim(-0.5, self.rows - 0.5)
        self.ax1.set_aspect('equal')
        
        # Draw grid
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 0:  # Free space - white
                    rect = patches.Rectangle((j - 0.5, self.rows - i - 1.5), 1, 1, 
                                           linewidth=1, edgecolor='lightgray', facecolor='white')
                    self.ax1.add_patch(rect)
                else:  # Obstacle - dark gray
                    rect = patches.Rectangle((j - 0.5, self.rows - i - 1.5), 1, 1, 
                                           linewidth=1, edgecolor='black', facecolor='darkgray')
                    self.ax1.add_patch(rect)
        
        # Draw starts and goals
        for agent_id, (start, goal) in enumerate(zip(self.starts, self.goals)):
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            
            # Start position - filled circle
            start_circle = patches.Circle((start[1], self.rows - start[0] - 1), 0.25, 
                                        facecolor=color, alpha=0.9, edgecolor='black', linewidth=2)
            self.ax1.add_patch(start_circle)
            self.ax1.text(start[1], self.rows - start[0] - 1, f'{agent_id+1}', 
                         ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            
            # Goal position - square
            goal_square = patches.Rectangle((goal[1] - 0.25, self.rows - goal[0] - 1.25), 0.5, 0.5,
                                         facecolor=color, alpha=0.4, edgecolor=color, linewidth=2)
            self.ax1.add_patch(goal_square)
        
        self.ax1.set_xticks(range(self.cols))
        self.ax1.set_yticks(range(self.rows))
        self.ax1.grid(True, alpha=0.3)
    
    def setup_animation(self):
        """Setup the animation plot"""
        self.ax2.set_title('Path Execution Animation', fontweight='bold')
        self.ax2.set_xlim(-0.5, self.cols - 0.5)
        self.ax2.set_ylim(-0.5, self.rows - 0.5)
        self.ax2.set_aspect('equal')
        
        # Draw grid
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 0:  # Free space
                    rect = patches.Rectangle((j - 0.5, self.rows - i - 1.5), 1, 1, 
                                           linewidth=1, edgecolor='lightgray', facecolor='white')
                    self.ax2.add_patch(rect)
                else:  # Obstacle
                    rect = patches.Rectangle((j - 0.5, self.rows - i - 1.5), 1, 1, 
                                           linewidth=1, edgecolor='black', facecolor='darkgray')
                    self.ax2.add_patch(rect)
        
        # Draw goal positions
        for agent_id, goal in enumerate(self.goals):
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            goal_square = patches.Rectangle((goal[1] - 0.25, self.rows - goal[0] - 1.25), 0.5, 0.5,
                                         facecolor=color, alpha=0.4, edgecolor=color, linewidth=2)
            self.ax2.add_patch(goal_square)
        
        self.ax2.set_xticks(range(self.cols))
        self.ax2.set_yticks(range(self.rows))
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize agent circles for animation
        self.agent_circles = []
        for agent_id in range(len(self.starts)):
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            circle = patches.Circle((0, 0), 0.25, facecolor=color, alpha=0.9, 
                                  edgecolor='black', linewidth=2)
            self.ax2.add_patch(circle)
            self.agent_circles.append(circle)
        
        # Calculate max path length for animation
        self.max_time = max(len(path) for path in self.paths.values())
        
        # Add time step text
        self.time_text = self.ax2.text(0.02, 0.98, '', transform=self.ax2.transAxes, 
                                      fontsize=12, fontweight='bold', 
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def animate(self, frame):
        """Animation function"""
        time_step = frame
        self.time_text.set_text(f'Time Step: {time_step}')
        
        for agent_id, circle in enumerate(self.agent_circles):
            if agent_id in self.paths:
                path = self.paths[agent_id]
                if time_step < len(path):
                    pos = path[time_step]
                    circle.center = (pos[1], self.rows - pos[0] - 1)
                    circle.set_alpha(0.9)
                else:
                    # Agent has reached goal, keep at final position
                    final_pos = path[-1]
                    circle.center = (final_pos[1], self.rows - final_pos[0] - 1)
                    circle.set_alpha(0.5)
        
        return self.agent_circles + [self.time_text]
    
    def show_with_animation(self, interval=1000):
        """Show plot with animation"""
        if self.paths:
            self.ani = FuncAnimation(self.fig, self.animate, frames=self.max_time + 2, 
                                   interval=interval, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()

# Create 10x10 grid with some obstacles
def create_10x10_grid():
    grid = [[0 for _ in range(10)] for _ in range(10)]
    
    # Add some obstacles to make it interesting
    obstacles = [
        (2, 2), (2, 3), (2, 4),
        (4, 6), (4, 7), (5, 6), (5, 7),
        (7, 1), (7, 2), (8, 1), (8, 2),
        (1, 8), (2, 8), (3, 8)
    ]
    
    for obs in obstacles:
        if 0 <= obs[0] < 10 and 0 <= obs[1] < 10:
            grid[obs[0]][obs[1]] = 1
    
    return grid

# Create the 10x10 grid
grid = create_10x10_grid()

# Define 5 agents with individual start and goal positions
starts = [
    (0, 0),  # Agent 1: top-left
    (0, 9),  # Agent 2: top-right
    (9, 0),  # Agent 3: bottom-left
    (9, 9),  # Agent 4: bottom-right
    (4, 4)   # Agent 5: center
]

goals = [
    (9, 9),  # Agent 1: to bottom-right
    (9, 0),  # Agent 2: to bottom-left
    (0, 9),  # Agent 3: to top-right
    (0, 0),  # Agent 4: to top-left
    (1, 1)   # Agent 5: to near top-left
]

# Define meta-agent groups (optional - can group agents that might cooperate)
# For this example, we'll use individual agents, but you could group them like:
# meta_agent_groups = [[0, 1], [2, 3], [4]]  # Group agents 1&2, 3&4, agent 5 alone
meta_agent_groups = None  # Each agent is its own meta-agent

print("Setting up Multi-Agent Path Finding problem...")
print("Environment: 10x10 grid with obstacles")
print("Agents and their start/goal positions:")
for i, (start, goal) in enumerate(zip(starts, goals)):
    print(f"  Agent {i+1}: Start at {start} → Goal at {goal}")

print("\nSolving using Meta-Agent CBS...")

# Create and solve the MAPF problem
mapf_solver = MetaAgentCBS(grid, starts, goals, meta_agent_groups)
solution_paths = mapf_solver.solve()

if solution_paths:
    print("\nSolution found!")
    print("Agent paths:")
    for agent_id, path in solution_paths.items():
        print(f"  Agent {agent_id+1}: {path}")
    
    # Calculate statistics
    total_cost = sum(len(path) - 1 for path in solution_paths.values())
    makespan = max(len(path) - 1 for path in solution_paths.values())
    
    print(f"\nSolution Statistics:")
    print(f"  Total Cost (sum of individual costs): {total_cost}")
    print(f"  Makespan (maximum individual cost): {makespan}")
    
    # Visualize the solution
    print(f"\nShowing visualization...")
    print("Left plot: Static environment layout with obstacles")
    print("Right plot: Animated path execution")
    
    visualizer = MAPFVisualizer(grid, starts, goals, solution_paths)
    visualizer.show_with_animation(interval=800)  # 0.8 second intervals
    
else:
    print("No solution found!")
    # Show just the environment
    visualizer = MAPFVisualizer(grid, starts, goals)
    plt.tight_layout()
    plt.show()