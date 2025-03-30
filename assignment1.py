import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

class UpperHullVisualizer:
    def __init__(self, points):
        self.points = np.array(points)
        self.fig = plt.figure(figsize=(15, 8))
        self.ax1 = self.fig.add_subplot(121)  # Hull visualization
        self.ax2 = self.fig.add_subplot(122)  # Stack visualization
        self.stack_frames = []
        self.stack_operations = []
        self.discarded_edges = []  # Store discarded edges
        self.current_frame = 0
        self.final_hull = None  # Store the final hull vertices
        
    def ccw(self, p1, p2, p3):
        """Returns true if points make a counter-clockwise turn"""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    def compute_upper_hull(self):
        """Compute upper hull and record stack operations"""
        sorted_points = self.points[self.points[:, 0].argsort()]
        
        # Initialize stack with first two points
        stack = [sorted_points[0], sorted_points[1]]
        self.stack_frames.append(stack.copy())
        self.stack_operations.append(("Push", f"({sorted_points[0][0]:.1f}, {sorted_points[0][1]:.1f})"))
        self.stack_operations.append(("Push", f"({sorted_points[1][0]:.1f}, {sorted_points[1][1]:.1f})"))
        self.discarded_edges.append([])  # No discarded edges for first two points
        self.discarded_edges.append([])
        
        for point in sorted_points[2:]:
            current_discarded = []  # Store edges discarded in this step
            while len(stack) >= 2 and self.ccw(stack[-2], stack[-1], point) >= 0:
                popped = stack.pop()
                # Record the discarded edge
                current_discarded.append((stack[-1], popped))
                self.stack_frames.append(stack.copy())
                self.stack_operations.append(("Pop", f"({popped[0]:.1f}, {popped[1]:.1f})"))
                self.discarded_edges.append(current_discarded.copy())
            
            stack.append(point)
            self.stack_frames.append(stack.copy())
            self.stack_operations.append(("Push", f"({point[0]:.1f}, {point[1]:.1f})"))
            self.discarded_edges.append(current_discarded.copy())
        
        self.final_hull = stack
        return stack
    
    def draw_stack(self, ax, stack, frame):
        """Visualize the stack state"""
        ax.clear()
        ax.set_title("Stack State")
        ax.axis('off')
        
        # Draw stack background
        stack_height = 10
        stack_width = 3
        rect = Rectangle((-stack_width/2, 0), stack_width, stack_height, 
                        facecolor='lightgray', edgecolor='black')
        ax.add_patch(rect)
        
        # Draw stack elements
        for i, point in enumerate(stack):
            y_pos = 1 + i
            text = f"({point[0]:.1f}, {point[1]:.1f})"
            ax.text(0, y_pos, text, ha='center', va='center',
                   bbox=dict(facecolor='white', edgecolor='black'))
        
        # Show last operation
        if frame < len(self.stack_operations):
            op_type, value = self.stack_operations[frame]
            color = 'green' if op_type == "Push" else 'red'
            ax.text(0, stack_height + 1, f"{op_type}: {value}",
                   ha='center', va='center', color=color,
                   bbox=dict(facecolor='white', edgecolor=color))
        
        ax.set_ylim(0, stack_height + 2)
        ax.set_xlim(-stack_width, stack_width)
    
    def update(self, frame):
        """Update both hull visualization and stack state"""
        # Update hull visualization
        self.ax1.clear()
        
        # Get current stack
        stack = self.stack_frames[frame]
        
        # Draw lines from first vertex to current hull vertices
        first_point = self.points[0]
        if len(stack) >= 2:
            for point in stack[1:]:  # Skip the first point itself
                self.ax1.plot([first_point[0], point[0]], 
                            [first_point[1], point[1]], 
                            'purple', alpha=0.5, linestyle='-', linewidth=1.5)
        
        # Plot all points
        self.ax1.scatter(self.points[:, 0], self.points[:, 1], 
                        c='blue', label='Input Points')
        
        # Plot current hull
        if len(stack) >= 2:
            stack_array = np.array(stack)
            self.ax1.plot(stack_array[:, 0], stack_array[:, 1], 
                         'g-', label='Current Hull', linewidth=2)
        
        # Plot discarded edges
        for i in range(frame + 1):
            for edge in self.discarded_edges[i]:
                self.ax1.plot([edge[0][0], edge[1][0]], 
                            [edge[0][1], edge[1][1]], 
                            'r--', alpha=0.5, linewidth=1)
        
        # Highlight stack points
        stack_array = np.array(stack)
        self.ax1.scatter(stack_array[:, 0], stack_array[:, 1], 
                        c='green', s=100, label='Hull Points')
        
        # Highlight first vertex
        self.ax1.scatter(first_point[0], first_point[1], 
                        c='orange', s=150, label='First Vertex')
        
        # Set plot properties
        self.ax1.set_title(f'Upper Hull Construction - Step {frame + 1}')
        legend_elements = [
            plt.Line2D([0], [0], color='green', label='Current Hull'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Discarded Edges'),
            plt.Line2D([0], [0], color='purple', alpha=0.5, label='Connections to Hull Vertices'),
            plt.scatter([0], [0], c='blue', label='Input Points'),
            plt.scatter([0], [0], c='green', label='Hull Points'),
            plt.scatter([0], [0], c='orange', label='First Vertex')
        ]
        self.ax1.legend(handles=legend_elements)
        self.ax1.grid(True)
        
        # Set consistent axes limits
        x_min, x_max = self.points[:, 0].min() - 1, self.points[:, 0].max() + 1
        y_min, y_max = self.points[:, 1].min() - 1, self.points[:, 1].max() + 1
        self.ax1.set_xlim(x_min, x_max)
        self.ax1.set_ylim(y_min, y_max)
        
        # Update stack visualization
        self.draw_stack(self.ax2, stack, frame)
        
        # Adjust layout
        self.fig.tight_layout()
    
    def animate(self):
        """Create and display the animation"""
        self.compute_upper_hull()
        anim = animation.FuncAnimation(self.fig, self.update, 
                                     frames=len(self.stack_frames),
                                     interval=1000, repeat=False)
        plt.show()
        return anim

def main():
    # Manual points for better visualization
    points = np.array([
        [1, 2],    # Starting point
        [2, 5],    # Higher point
        [3, 3],    # Point that will be discarded
        [5, 6],    # Higher point
        [5, 4],    # Point that will be discarded
        [6, 7],    # Highest point
        [7, 5],    # Point that will be discarded
        [8, 6],    # Second highest point
        [9, 4],    # Lower point
        [10, 3],   # Ending point
    ])
    
    visualizer = UpperHullVisualizer(points)
    visualizer.animate()

if __name__ == "__main__":
    main()