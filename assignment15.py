import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import matplotlib.gridspec as gridspec
import networkx as nx
import heapq

class Line:
    def __init__(self, slope, intercept, name):
        self.slope = slope
        self.intercept = intercept
        self.name = name
    
    def y_at(self, x):
        """Calculate y value at given x coordinate"""
        return self.slope * x + self.intercept
    
    def __str__(self):
        return f"{self.name} (m={self.slope:.2f}, b={self.intercept:.2f})"

class IntersectionPoint:
    def __init__(self, x, y, lines):
        self.x = x
        self.y = y
        self.lines = lines  # Pair of lines that intersect here
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.x < other.x
    
    def __str__(self):
        return f"Intersection at ({self.x:.2f}, {self.y:.2f}) between {self.lines[0].name} and {self.lines[1].name}"

class TopologicalPlaneSweep:
    def __init__(self, num_lines):
        self.num_lines = num_lines
        self.lines = []
        self.current_cut_order = []
        self.upper_horizon_tree = []
        self.lower_horizon_tree = []
        self.stack = []  # Priority queue for upcoming intersections
        self.processed_intersections = []
        self.current_x = 0
        self.step_count = 0
        self.colors = plt.cm.tab20(np.linspace(0, 1, num_lines))
        self.all_intersections = []  # For visualization only
        
    def generate_random_lines(self):
        """Generate random non-parallel lines with distinct slopes"""
        used_slopes = set()
        
        for i in range(self.num_lines):
            while True:
                # Generate random slope between -5 and 5, but not too close to 0
                slope = random.uniform(-5, 5)
                if abs(slope) < 0.1:  # Avoid nearly horizontal lines
                    continue
                
                # Check if this slope is already used (with some tolerance)
                if all(abs(slope - s) > 0.2 for s in used_slopes):
                    used_slopes.add(slope)
                    break
            
            # Generate random y-intercept between -10 and 10
            intercept = random.uniform(-10, 10)
            
            name = f"C{i+1}"
            self.lines.append(Line(slope, intercept, name))
    
    def calculate_intersection(self, line1, line2):
        """Calculate intersection point between two lines"""
        # Skip parallel lines
        if abs(line1.slope - line2.slope) < 1e-10:
            return None
        
        # Calculate intersection
        x = (line2.intercept - line1.intercept) / (line1.slope - line2.slope)
        y = line1.slope * x + line1.intercept
        
        return IntersectionPoint(x, y, [line1, line2])
    
    def initialize_sweep(self):
        """Initialize the sweep line at the leftmost possible position"""
        # Find a good starting x position (before all possible intersections)
        min_x = float('inf')
        for i in range(len(self.lines)):
            for j in range(i+1, len(self.lines)):
                intersection = self.calculate_intersection(self.lines[i], self.lines[j])
                if intersection:
                    min_x = min(min_x, intersection.x)
                    # Store for visualization only
                    self.all_intersections.append(intersection)
                    
        # Start slightly to the left of the leftmost intersection
        self.current_x = min_x - 2 if min_x != float('inf') else -10
            
        # Sort the lines by their y-value at the current_x (from top to bottom)
        self.current_cut_order = sorted(self.lines, key=lambda line: -line.y_at(self.current_x))
        
        # Rename the original self.lines based on the current cut order
        for i, line in enumerate(self.current_cut_order):
            line.name = f"L{i+1}"  # Assuming each line has a 'name' attribute
        
        # Initialize upper and lower horizon trees
        self.update_horizon_trees()
        
        # Find and add initial intersections to stack
        self.find_new_intersections()
    
    def update_horizon_trees(self):
        """Update the Upper and Lower Horizon Trees at the current cut position"""
        # Get line ordering at current position
        sorted_lines = self.current_cut_order.copy()
        
        # Upper horizon tree: for each line, which line is directly above it
        self.upper_horizon_tree = []
        for i in range(1, len(sorted_lines)):
            self.upper_horizon_tree.append((sorted_lines[i].name, sorted_lines[i-1].name))
        
        # Lower horizon tree: for each line, which line is directly below it
        self.lower_horizon_tree = []
        for i in range(len(sorted_lines) - 1):
            self.lower_horizon_tree.append((sorted_lines[i].name, sorted_lines[i+1].name))
    
    def find_new_intersections(self):
        """Find new intersection points between adjacent lines in the cut ordering"""
        for i in range(len(self.current_cut_order) - 1):
            line1 = self.current_cut_order[i]
            line2 = self.current_cut_order[i + 1]
            
            # Calculate intersection point
            intersection = self.calculate_intersection(line1, line2)
            
            # Only add if it's to the right of the current sweep position
            if intersection and intersection.x > self.current_x:
                # Check if this intersection is already in the stack
                is_duplicate = False
                for existing in self.stack:
                    if (abs(existing.x - intersection.x) < 1e-10 and 
                        abs(existing.y - intersection.y) < 1e-10 and
                        set([l.name for l in existing.lines]) == set([l.name for l in intersection.lines])):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    heapq.heappush(self.stack, intersection)
    
    def perform_elementary_step(self):
        """Perform one elementary step of the topological plane sweep"""
        if not self.stack:
            return False  # No more intersections to process
        
        # Get the next intersection point
        intersection = heapq.heappop(self.stack)
        self.processed_intersections.append(intersection)
        self.current_x = intersection.x
        self.step_count += 1
        
        # Find the indices of the intersecting lines in the current cut order
        line1, line2 = intersection.lines
        line1_idx = -1
        line2_idx = -1
        
        for i, line in enumerate(self.current_cut_order):
            if line.name == line1.name:
                line1_idx = i
            elif line.name == line2.name:
                line2_idx = i
        
        # Check if lines are adjacent in the current cut order
        if abs(line1_idx - line2_idx) != 1:
            # This can happen if multiple intersections occur at the same x-coordinate
            # In this case, we need to handle them in the correct order
            return True
        
        # Ensure line1 is above line2 in the current ordering
        if line1_idx > line2_idx:
            line1_idx, line2_idx = line2_idx, line1_idx
            line1, line2 = line2, line1
        
        # Swap the two lines in the current cut order
        self.current_cut_order[line1_idx], self.current_cut_order[line2_idx] = \
            self.current_cut_order[line2_idx], self.current_cut_order[line1_idx]
        
        # Update the horizon trees after the swap
        self.update_horizon_trees()
        
        # Find new potential intersections after the swap
        self.find_new_intersections()
        
        return True  # Successfully performed a step
    
    def get_animation_frames(self):
        """Generate all frames for the animation"""
        frames = []
        
        # Add initial state
        frames.append({
            'step': 0,
            'current_x': self.current_x,
            'cut_order': self.current_cut_order.copy(),
            'upper_horizon_tree': self.upper_horizon_tree.copy(),
            'lower_horizon_tree': self.lower_horizon_tree.copy(),
            'stack': self.stack.copy(),
            'processed': self.processed_intersections.copy(),
            'title': "Initial Arrangement"
        })
        
        # Perform all elementary steps and save frames
        step = 1
        while self.perform_elementary_step():
            frames.append({
                'step': step,
                'current_x': self.current_x,
                'cut_order': self.current_cut_order.copy(),
                'upper_horizon_tree': self.upper_horizon_tree.copy(),
                'lower_horizon_tree': self.lower_horizon_tree.copy(),
                'stack': self.stack.copy(),
                'processed': self.processed_intersections.copy(),
                'title': f"Step {step}: Intersection at x={self.current_x:.2f}"
            })
            step += 1
        
        return frames
    
    def create_horizon_tree_graph(self, horizon_tree, ax, title, node_color='skyblue'):
        """Create a directed graph visualization of a horizon tree"""
        G = nx.DiGraph()
        
        # Add nodes and edges
        for edge in horizon_tree:
            source, target = edge
            G.add_edge(source, target)
        
        # If the graph is empty, just add a dummy node
        if len(G.nodes()) == 0:
            G.add_node("No edges")
        
        # Position nodes in a vertical layout
        pos = {}
        node_list = list(G.nodes())
        
        # Create positions for nodes based on their position in the cut order
        node_to_pos = {}
        for i, line in enumerate(self.current_cut_order):
            node_to_pos[line.name] = i
        
        # Sort nodes by their position in the cut order
        sorted_nodes = sorted(node_list, key=lambda n: node_to_pos.get(n, 999))
        
        # Position nodes in a vertical layout
        for i, node in enumerate(sorted_nodes):
            # Offset alternating nodes for better edge visibility
            offset = 0.2 if i % 2 == 0 else -0.2
            pos[node] = (0.5 + offset, len(sorted_nodes) - i)
        
        # Draw the graph
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_color, 
                node_size=700, arrowsize=20, width=2, edge_color='red',
                font_weight='bold', font_size=10)
        
        ax.set_title(title)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, len(sorted_nodes) + 0.5)
        ax.axis('off')
    
    def visualize_animation(self):
        """Create an animation of the plane sweep algorithm"""
        # Generate all frames
        frames = self.get_animation_frames()
        
        # Create figure with gridspec for layout
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[3, 1])
        
        # Sweep area subplot (left side, top)
        ax_sweep = fig.add_subplot(gs[0, 0])
        
        # UHT visualization subplot (middle top)
        ax_uht = fig.add_subplot(gs[0, 1])
        
        # LHT visualization subplot (right top)
        ax_lht = fig.add_subplot(gs[0, 2])
        
        # Stack and Cut Order visualization subplot (left side bottom)
        ax_stack_cut = fig.add_subplot(gs[1, :])
        ax_stack_cut.axis('off')  # Hide axes for this panel
        
        # Calculate reasonable plot bounds
        min_x = min(frame['current_x'] for frame in frames) - 2
        max_x = max(frame['current_x'] for frame in frames) + 2
        
        # For y-axis bounds, consider all lines at min_x and max_x
        y_values = []
        for line in self.lines:
            y_values.append(line.y_at(min_x))
            y_values.append(line.y_at(max_x))
        min_y = min(y_values) - 2
        max_y = max(y_values) + 2
        
        # Prepare line objects
        line_objects = {}
        for i, line in enumerate(self.lines):
            y1 = line.y_at(min_x)
            y2 = line.y_at(max_x)
            line_obj, = ax_sweep.plot([min_x, max_x], [y1, y2], color=self.colors[i], label=line.name, lw=2)
            line_objects[line.name] = line_obj
        
        # Prepare intersection points scatter - for visualization only
        x_coords = [p.x for p in self.all_intersections]
        y_coords = [p.y for p in self.all_intersections]
        intersections_scatter = ax_sweep.scatter(x_coords, y_coords, color='red', s=50, zorder=3)
        
        # Prepare current cut line
        cut_line, = ax_sweep.plot([min_x, min_x], [min_y, max_y], 'g--', linewidth=2, label="Current Cut")
        
        # Set up sweep area
        ax_sweep.set_xlim(min_x, max_x)
        ax_sweep.set_ylim(min_y, max_y)
        ax_sweep.set_title("Topological Plane Sweep")
        ax_sweep.legend(loc='upper right')
        ax_sweep.set_xlabel("x")
        ax_sweep.set_ylabel("y")
        ax_sweep.grid(True, linestyle='--', alpha=0.7)
        
        # Text objects for labels in the current cut
        cut_labels = []
        for i in range(len(self.lines)):
            label = ax_sweep.text(0, 0, "", bbox=dict(facecolor='yellow', alpha=0.7), visible=False)
            cut_labels.append(label)
        
        # Create horizon tree graphs for initial state
        self.create_horizon_tree_graph(frames[0]['upper_horizon_tree'], ax_uht, "Upper Horizon Tree")
        self.create_horizon_tree_graph(frames[0]['lower_horizon_tree'], ax_lht, "Lower Horizon Tree", node_color='lightgreen')
        
        def update(frame_idx):
            nonlocal fig, ax_uht, ax_lht, ax_stack_cut
            frame = frames[frame_idx]
            
            # Update title
            fig.suptitle(frame['title'], fontsize=16)
            
            # Update sweep line position
            current_x = frame['current_x']
            cut_line.set_xdata([current_x, current_x])
            
            # Update line labels at current cut
            for i, label in enumerate(cut_labels):
                if i < len(frame['cut_order']):
                    line = frame['cut_order'][i]
                    y = line.y_at(current_x)
                    label.set_position((current_x + 0.5, y))
                    label.set_text(f"{i+1}: {line.name}")
                    label.set_visible(True)
                else:
                    label.set_visible(False)
            
            # Update horizon tree visualizations
            ax_uht.clear()
            ax_lht.clear()
            
            self.create_horizon_tree_graph(frame['upper_horizon_tree'], ax_uht, "Upper Horizon Tree")
            self.create_horizon_tree_graph(frame['lower_horizon_tree'], ax_lht, "Lower Horizon Tree", node_color='lightgreen')
            
            # Update stack and cut order visualization
            ax_stack_cut.clear()
            
            # Visualize current cut order
            cut_order_text = "Current Cut Order (top to bottom):\n"
            for i, line in enumerate(frame['cut_order']):
                cut_order_text += f"{i+1}. {line.name}\n"
            
            ax_stack_cut.text(0.05, 0.7, cut_order_text, fontsize=12, 
                           transform=ax_stack_cut.transAxes, verticalalignment='top',
                           bbox=dict(facecolor='lightblue', alpha=0.5))
            
            # Visualize stack - format as a priority queue showing next intersection first
            stack_text = "Intersection Stack (next at top):\n"
            if frame['stack']:
                # Get a sorted copy of the stack
                sorted_stack = sorted(frame['stack'], key=lambda x: x.x)
                for i, intersection in enumerate(sorted_stack):
                    line1, line2 = intersection.lines
                    stack_text += f"{line1.name} ∩ {line2.name} at x={intersection.x:.2f}\n"
            else:
                stack_text += "Empty"
            
            ax_stack_cut.text(0.4, 0.7, stack_text, fontsize=12, 
                           transform=ax_stack_cut.transAxes, verticalalignment='top',
                           bbox=dict(facecolor='lightgreen', alpha=0.5))
            
            # Visualize processed intersections
            processed_text = "Processed Intersections:\n"
            if frame['processed']:
                for i, intersection in enumerate(frame['processed']):
                    line1, line2 = intersection.lines
                    processed_text += f"{line1.name} ∩ {line2.name} at x={intersection.x:.2f}\n"
            else:
                processed_text += "None"
            
            ax_stack_cut.text(0.75, 0.7, processed_text, fontsize=12, 
                           transform=ax_stack_cut.transAxes, verticalalignment='top',
                           bbox=dict(facecolor='lightyellow', alpha=0.5))
            
            # Ensure the subplot has the correct dimensions
            ax_stack_cut.set_xlim(0, 1)
            ax_stack_cut.set_ylim(0, 1)
            ax_stack_cut.axis('off')
            
            # Return all updated artists
            artists = [cut_line, intersections_scatter]
            artists.extend(cut_labels)
            artists.extend(line_objects.values())
            return artists
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(frames), 
                                    interval=1500, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3)
        
        return ani, fig
    
    def run_algorithm(self):
        """Run the complete algorithm and create an animation"""
        print("Generating random lines...")
        self.generate_random_lines()
        
        print("Initializing sweep...")
        self.initialize_sweep()
        
        print("Creating animation...")
        ani, fig = self.visualize_animation()
        
        return ani, fig


# Example usage
if __name__ == "__main__":
    # Get input from user
    num_lines = int(input("Enter the number of lines (N): "))
    np.random.seed(42)
    random.seed(42)
    # Initialize and run the algorithm
    sweep = TopologicalPlaneSweep(num_lines)
    ani, fig = sweep.run_algorithm()
    
    # To save animation:
    # ani.save('topological_sweep.mp4', writer='ffmpeg', fps=1)
    
    # To display animation:
    plt.show()