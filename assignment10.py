import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Set Seaborn style
sns.set_theme(style="whitegrid")  # Use whitegrid style for better readability
sns.set_context("notebook", font_scale=1.2)  # Increase font size slightly

N = 1 ## global variable for tracking the splitter number

class Node:
    def __init__(self, point, axis, number, split_value):
        self.point = point        # The point stored at this node
        self.left = None          # Left child
        self.right = None         # Right child
        self.axis = axis          # 0 for x-axis split, 1 for y-axis split
        self.number = number      # Line number for visualization
        self.split_value = split_value  # Value where the splitting happens

def find_bounding_box(points):
    """Find the bounding box for a set of points."""
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    return min_x, max_x, min_y, max_y

def build_kdtree(points, depth=0):
    """Build a kd-tree from a set of points, with special handling for median points."""
    global N
    n = len(points)
    
    # Base case: empty set of points
    if n == 0:
        return None
    
    # Base case: only one point left
    if n == 1:
        # We don't partition further when we reach a single point
        return Node(points[0], depth % 2, 0, None)  # Number 0 means no partition line here
    
    # Alternate axis between x (0) and y (1)
    axis = depth % 2
    
    # Sort points by the current axis
    sorted_points = sorted(points, key=lambda point: point[axis])
    
    # Find split value based on median
    if n % 2 == 0:  # Even number of points
        # Take the mean of the two middle points
        middle1 = sorted_points[n//2 - 1]
        middle2 = sorted_points[n//2]
        split_value = (middle1[axis] + middle2[axis]) / 2
        median_point = (sorted_points[n//2 - 1] if axis == 0 else sorted_points[n//2])
    else:  # Odd number of points
        # Use the middle point
        split_value = sorted_points[n//2][axis]
        median_point = sorted_points[n//2]
    
    # Create node
    node = Node(median_point, axis, N, split_value)
    # Divide points into left and right groups
    # Left group includes all points with value <= split_value
    left_points = [p for p in sorted_points if p[axis] <= split_value]
    # Right group includes all points with value > split_value
    right_points = [p for p in sorted_points if p[axis] > split_value]
    
    # Increment partition number for children
    N += 1
    
    # Recursively build subtrees
    node.left = build_kdtree(left_points, depth + 1)
    
    # Update next partition number based on left subtree
    if node.left is not None and node.left.number > 0:
        N = max(N, node.left.number + 1)
    
    node.right = build_kdtree(right_points, depth + 1)
    
    return node

def create_kdtree_animation(points, root, bbox):
    """Create an animation of the kd-tree partitioning process with improved Seaborn styling."""
    # Unpack bounding box
    min_x, max_x, min_y, max_y = bbox
    
    # Add some padding to the bounding box
    padding = max(max_x - min_x, max_y - min_y) * 0.1
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding
    
    # Setup the figure and axes with Seaborn styling
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(min_x-2, max_x+2)
    ax.set_ylim(min_y-2, max_y+2)
    
    # Use Seaborn's color palette
    palette = sns.color_palette("hls")
    point_color = palette[0]  # First color for points
    line_colors = [palette[1], palette[2]]  # Different colors for vertical and horizontal lines
    
    # Set up figure title with Seaborn styling
    ax.set_title('2D-Tree building animation', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X-axis', fontweight='bold', labelpad=10)
    ax.set_ylabel('Y-axis', fontweight='bold', labelpad=10)
    
    # Plot all points with improved styling
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    scatter = ax.scatter(x_coords, y_coords, c=[point_color], s=50, 
                         edgecolor='white', linewidth=1.5, alpha=0.8, zorder=3)
    
    # Add point labels
    # for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    #     ax.text(x + 0.1, y + 0.1, f'{i+1}', fontsize=10, ha='left', va='bottom')
    
    # Draw initial bounding box with Seaborn style
    bbox_rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                          fill=False, edgecolor='gray', linestyle='--', 
                          linewidth=1.5, alpha=0.6, zorder=1)
    ax.add_patch(bbox_rect)
    
    # Storage for partition lines and their text labels
    partition_lines = []
    text_labels = []
    
    # Collect all partitioning lines
    def collect_partition_lines(node, x_min, x_max, y_min, y_max):
        if node is None or node.number == 0 or node.split_value is None:
            return
        
        if node.axis == 0:  # Vertical line (x-axis partition)
            split_x = node.split_value
            # Use the first color for vertical lines
            line, = ax.plot([split_x, split_x], [y_min, y_max], 
                           color=line_colors[0], linestyle='-', linewidth=2.5, alpha=0, zorder=2)
            text = ax.text(split_x, y_max + padding/4, str(node.number), 
                          ha='center', va='bottom', fontsize=12, fontweight='bold', 
                          color=line_colors[0], alpha=0)
            partition_lines.append((line, node.number))
            text_labels.append((text, node.number))
            
            # Continue recursion with updated bounds
            collect_partition_lines(node.left, x_min, split_x, y_min, y_max)
            collect_partition_lines(node.right, split_x, x_max, y_min, y_max)
        else:  # Horizontal line (y-axis partition)
            split_y = node.split_value
            # Use the second color for horizontal lines
            line, = ax.plot([x_min, x_max], [split_y, split_y], 
                           color=line_colors[1], linestyle='-', linewidth=2.5, alpha=0, zorder=2)
            text = ax.text(x_max + padding/4, split_y, str(node.number), 
                          ha='left', va='center', fontsize=12, fontweight='bold', 
                          color=line_colors[1], alpha=0)
            partition_lines.append((line, node.number))
            text_labels.append((text, node.number))
            
            # Continue recursion with updated bounds
            collect_partition_lines(node.left, x_min, x_max, y_min, split_y)
            collect_partition_lines(node.right, x_min, x_max, split_y, y_max)
    
    # Collect all the partition lines
    collect_partition_lines(root, min_x, max_x, min_y, max_y)
    
    # Sort partition lines by their number
    partition_lines.sort(key=lambda x: x[1])
    text_labels.sort(key=lambda x: x[1])
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=point_color, 
               markersize=10, label='Data Points'),
        Line2D([0], [0], color=line_colors[0], lw=2, label='Horizontal Partition'),
        Line2D([0], [0], color=line_colors[1], lw=2, label='Vertical Partition')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
              framealpha=0.9, edgecolor='lightgray')
    
    def init():
        return []
    
    def animate(i):
        if i < len(partition_lines):
            # Make the i-th line visible
            line, number = partition_lines[i]
            line.set_alpha(1)
            
            # Make the corresponding text visible
            for text, text_num in text_labels:
                if text_num == number:
                    text.set_alpha(1)
                    break
        
        visible_objects = [line for line, _ in partition_lines[:i+1]]
        visible_objects.extend([text for text, _ in text_labels if text.get_alpha() > 0])
        return visible_objects
    
    # Create the animation
    frames = len(partition_lines) + 5  # Extra frames at the end to see the final result
    ani = FuncAnimation(fig, animate, frames=frames,
                        init_func=init, blit=True, repeat=False, interval=800)
    
    plt.tight_layout()
    return fig, ani

def main():
    # Generate random points
    np.random.seed(100)  # For reproducibility
    N = int(input("Enter number of points : "))
    num_points = N  # Number of points
    
    points = np.random.rand(num_points, 2) * 10  # Random points in [0, 10] x [0, 10]
    points = [(p[0], p[1]) for p in points]  # Convert to list of tuples
    # points = [(0,0),(1,1),(0.8,0)]
    # Find bounding box
    bbox = find_bounding_box(points)
    print(f"Bounding Box: X range [{bbox[0]:.2f}, {bbox[1]:.2f}], Y range [{bbox[2]:.2f}, {bbox[3]:.2f}]")
    
    # Print points
    print("Points:")
    for i, p in enumerate(points):
        print(f"Point {i+1}: ({p[0]:.2f}, {p[1]:.2f})")
    
    # Build the KD-tree
    root = build_kdtree(points)
    
    # Create and display the animation
    fig, ani = create_kdtree_animation(points, root, bbox)
    
    # Save the animation if desired
    # ani.save('kdtree_animation.gif', writer='pillow', fps=1)
    
    # Display the animation
    plt.show()

if __name__ == "__main__":
    main()