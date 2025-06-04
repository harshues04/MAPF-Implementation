import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import heapq
from collections import defaultdict
import copy

class SpaceTimeAStar:
    def __init__(self, grid, starts, goals):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.rows, self.cols = len(grid), len(grid[0])
        self.agents = len(starts)
        
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
    
    def is_conflict(self, agent_paths, agent_id, new_pos, time, current_path):
        """Check for vertex and edge conflicts"""
        # Vertex conflict: same position at same time
        for other_agent, path in agent_paths.items():
            if other_agent != agent_id and len(path) > time:
                if path[time] == new_pos:
                    return True
                
                # Edge conflict: agents swap positions
                if (len(path) > time + 1 and 
                    len(current_path) > time - 1 and
                    time > 0 and
                    path[time] == current_path[time - 1] and
                    path[time - 1] == new_pos):
                    return True
        
        return False
    
    def space_time_astar(self, agent_id, existing_paths, max_time=50):
        """Space-Time A* for single agent"""
        start = self.starts[agent_id]
        goal = self.goals[agent_id]
        
        # Priority queue: (f_cost, g_cost, time, position, path)
        open_list = [(self.heuristic(start, goal), 0, 0, start, [start])]
        closed_set = set()
        
        while open_list:
            f_cost, g_cost, time, current_pos, path = heapq.heappop(open_list)
            
            # Goal reached
            if current_pos == goal and time > 0:
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
                
                # Check for conflicts with existing paths
                temp_path = path + [next_pos]
                if not self.is_conflict(existing_paths, agent_id, next_pos, new_time, path):
                    new_g_cost = g_cost + 1
                    new_f_cost = new_g_cost + self.heuristic(next_pos, goal)
                    
                    heapq.heappush(open_list, 
                                 (new_f_cost, new_g_cost, new_time, next_pos, temp_path))
        
        return None  # No path found
    
    def solve(self):
        """Solve MAPF using Space-Time A* with prioritized planning"""
        agent_paths = {}
        
        # Plan paths for agents in order (prioritized planning)
        for agent_id in range(self.agents):
            print(f"Planning path for Agent {agent_id}...")
            path = self.space_time_astar(agent_id, agent_paths)
            
            if path is None:
                print(f"No solution found for Agent {agent_id}")
                return None
            
            agent_paths[agent_id] = path
            print(f"Agent {agent_id} path: {path}")
        
        return agent_paths

class MAPFVisualizer:
    def __init__(self, grid, starts, goals, paths=None):
        self.grid = np.array(grid)
        self.starts = starts
        self.goals = goals
        self.paths = paths
        self.rows, self.cols = self.grid.shape
        
        # Set up the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.suptitle('Multi-Agent Path Finding: Space-Time A*', fontsize=16, fontweight='bold')
        
        # Colors for agents (matching your image)
        self.agent_colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        self.setup_static_plot()
        if paths:
            self.setup_animation()
    
    def setup_static_plot(self):
        """Setup the static environment plot"""
        self.ax1.set_title('White Cross Environment Layout', fontweight='bold')
        self.ax1.set_xlim(-0.5, self.cols - 0.5)
        self.ax1.set_ylim(-0.5, self.rows - 0.5)
        self.ax1.set_aspect('equal')
        
        # Draw the entire background as gray first
        background = patches.Rectangle((-0.5, -0.5), self.cols, self.rows, 
                                     facecolor='gray', edgecolor='none')
        self.ax1.add_patch(background)
        
        # Draw white squares for free spaces (forming the cross)
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 0:  # Free space - white
                    rect = patches.Rectangle((j - 0.5, self.rows - i - 1.5), 1, 1, 
                                           linewidth=1, edgecolor='black', facecolor='white')
                    self.ax1.add_patch(rect)
        
        # Draw starts and goals
        for agent_id, (start, goal) in enumerate(zip(self.starts, self.goals)):
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            
            # Start position - filled circle
            start_circle = patches.Circle((start[1], self.rows - start[0] - 1), 0.3, 
                                        facecolor=color, alpha=0.9, edgecolor='black', linewidth=2)
            self.ax1.add_patch(start_circle)
            self.ax1.text(start[1], self.rows - start[0] - 1, f'{agent_id+1}', 
                         ha='center', va='center', fontweight='bold', color='white', fontsize=12)
            
            # Goal position - semi-transparent square
            goal_square = patches.Rectangle((goal[1] - 0.35, self.rows - goal[0] - 1.35), 0.7, 0.7,
                                         facecolor=color, alpha=0.4, edgecolor=color, linewidth=2)
            self.ax1.add_patch(goal_square)
        
        # Add position labels
        positions = ['A', 'C', 'B', 'D', 'E']
        coords = [(0, 1), (1, 1), (1, 0), (1, 2), (2, 1)]
        
        for pos, (r, c) in zip(positions, coords):
            if self.grid[r, c] == 0:  # Only label free cells
                self.ax1.text(c - 0.4, self.rows - r - 0.6, pos, ha='center', va='center', 
                             fontsize=10, fontweight='bold', color='black')
        
        self.ax1.set_xticks(range(self.cols))
        self.ax1.set_yticks(range(self.rows))
        self.ax1.tick_params(labelbottom=False, labelleft=False)
    
    def setup_animation(self):
        """Setup the animation plot"""
        self.ax2.set_title('Path Execution Animation', fontweight='bold')
        self.ax2.set_xlim(-0.5, self.cols - 0.5)
        self.ax2.set_ylim(-0.5, self.rows - 0.5)
        self.ax2.set_aspect('equal')
        
        # Draw the entire background as gray first
        background = patches.Rectangle((-0.5, -0.5), self.cols, self.rows, 
                                     facecolor='gray', edgecolor='none')
        self.ax2.add_patch(background)
        
        # Draw white squares for free spaces
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 0:
                    rect = patches.Rectangle((j - 0.5, self.rows - i - 1.5), 1, 1, 
                                           linewidth=1, edgecolor='black', facecolor='white')
                    self.ax2.add_patch(rect)
        
        # Draw goal positions
        for agent_id, goal in enumerate(self.goals):
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            goal_square = patches.Rectangle((goal[1] - 0.35, self.rows - goal[0] - 1.35), 0.7, 0.7,
                                         facecolor=color, alpha=0.4, edgecolor=color, linewidth=2)
            self.ax2.add_patch(goal_square)
        
        self.ax2.set_xticks(range(self.cols))
        self.ax2.set_yticks(range(self.rows))
        self.ax2.tick_params(labelbottom=False, labelleft=False)
        
        # Initialize agent circles for animation
        self.agent_circles = []
        for agent_id in range(len(self.starts)):
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            circle = patches.Circle((0, 0), 0.3, facecolor=color, alpha=0.9, 
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
                    circle.set_alpha(0.5)  # Fade out completed agents
        
        return self.agent_circles + [self.time_text]
    
    def show_static(self):
        """Show static plot only"""
        plt.tight_layout()
        plt.show()
    
    def show_with_animation(self, interval=1000):
        """Show plot with animation"""
        if self.paths:
            self.ani = FuncAnimation(self.fig, self.animate, frames=self.max_time + 2, 
                                   interval=interval, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()

grid = [
    [1, 0, 1],  # Gray-White-Gray (A in middle)
    [0, 0, 0],  # White-White-White (B-C-D)
    [1, 0, 1]   # Gray-White-Gray (E in middle)
]

# Agent starting positions and goals (matching your image)
starts = [(0, 1), (1, 0)]  # Agent 1 at A (0,1), Agent 2 at B (1,0)
goals = [(2, 1), (1, 2)]   # Agent 1 to E (2,1), Agent 2 to D (1,2)

# Create and solve the MAPF problem
print("Setting up Multi-Agent Path Finding problem...")
print("Environment: White cross on gray background")
print("Agent 1 (Blue): Start at position A (0,1) → Goal at position E (2,1)")
print("Agent 2 (Green): Start at position B (1,0) → Goal at position D (1,2)")
print("\nSolving using Space-Time A*...")

mapf_solver = SpaceTimeAStar(grid, starts, goals)
solution_paths = mapf_solver.solve()

if solution_paths:
    print("\nSolution found!")
    print("Agent paths:")
    for agent_id, path in solution_paths.items():
        print(f"  Agent {agent_id}: {path}")
    
    # Calculate some statistics
    total_cost = sum(len(path) - 1 for path in solution_paths.values())
    makespan = max(len(path) - 1 for path in solution_paths.values())
    
    print(f"\nSolution Statistics:")
    print(f"  Total Cost (sum of individual costs): {total_cost}")
    print(f"  Makespan (maximum individual cost): {makespan}")
    
    # Visualize the solution
    visualizer = MAPFVisualizer(grid, starts, goals, solution_paths)
    print(f"\nShowing visualization...")
    print("Left plot: Static environment layout")
    print("Right plot: Animated path execution")
    visualizer.show_with_animation(interval=1500)  # 1.5 second intervals
    
else:
    print("No solution found!")
    # Show just the environment
    visualizer = MAPFVisualizer(grid, starts, goals)
    visualizer.show_static()