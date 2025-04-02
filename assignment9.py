'''Assignment 9
============
1.Build a Quadtree T for a set P of N points. 
2.Build a numbering of the nodes of T(maybe adj. List)
3.Take two random node index and draw the disks enclosing
those two nodes (square/rectangle)'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys
from matplotlib.widgets import Button, TextBox
import seaborn as sns

class QuadNode:
    def __init__(self, boundary, max_points=4, depth=0):
        self.boundary = boundary  # [x_min, y_min, x_max, y_max]
        self.max_points = max_points
        self.points = []
        self.divided = False
        self.children = []  # northwest, northeast, southwest, southeast
        self.depth = depth
        self.label = None  # Will be assigned during inorder traversal
    
    def insert(self, point):
        # Check if point is within boundary
        if not self._contains_point(point):
            return False
        
        # If this node has space and is not divided, add the point here
        if len(self.points) < self.max_points and not self.divided:
            self.points.append(point)
            return True
        
        # Otherwise, we need to subdivide (if we haven't already)
        if not self.divided:
            self._subdivide()
        
        # Try to insert into children
        for child in self.children:
            if child.insert(point):
                return True
        
        # Should never reach here if boundary check is correct
        return False
    
    def _contains_point(self, point):
        x, y = point
        x_min, y_min, x_max, y_max = self.boundary
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def _subdivide(self):
        x_min, y_min, x_max, y_max = self.boundary
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        
        # Create boundaries for children
        nw = [x_min, y_mid, x_mid, y_max]  # Northwest
        ne = [x_mid, y_mid, x_max, y_max]  # Northeast
        sw = [x_min, y_min, x_mid, y_mid]  # Southwest
        se = [x_mid, y_min, x_max, y_mid]  # Southeast
        
        # Create children nodes
        self.children = [
            QuadNode(nw, self.max_points, self.depth + 1),
            QuadNode(ne, self.max_points, self.depth + 1),
            QuadNode(sw, self.max_points, self.depth + 1),
            QuadNode(se, self.max_points, self.depth + 1)
        ]
        
        # Move existing points to children
        for point in self.points:
            for child in self.children:
                if child.insert(point):
                    break
        
        self.points = []  # Clear points from this node
        self.divided = True
    
    def is_leaf(self):
        return not self.divided
    
    def get_all_nodes(self):
        """Get all nodes in the quadtree."""
        nodes = [self]
        if self.divided:
            for child in self.children:
                nodes.extend(child.get_all_nodes())
        return nodes
    
    def get_all_leaf_nodes(self):
        """Get all leaf nodes in the quadtree."""
        if self.is_leaf():
            return [self]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaf_nodes())
        return leaves


class QuadTree:
    def __init__(self, boundary, max_points=4):
        self.root = QuadNode(boundary, max_points)
        self.labeled_nodes = []  # Will store both internal and leaf nodes
        self.inserted_points = []
        self.animation_completed = False
    
    def insert(self, point):
        self.inserted_points.append(point)
        return self.root.insert(point)
    
    def insert_points(self, points):
        for point in points:
            self.insert(point)
    
    def get_inserted_points_count(self):
        return len(self.inserted_points)
    
    def label_all_nodes(self):
        """Label all nodes (both internal and leaf) using inorder traversal."""
        self.labeled_nodes = []
        self._inorder_traversal(self.root)
        return self.labeled_nodes
    
    def _inorder_traversal(self, node, index=0):
        if node is None:
            return index
        
        if len(node.children) > 0:
            # Process NW child
            index = self._inorder_traversal(node.children[0], index)
            
            # Process NE child
            index = self._inorder_traversal(node.children[1], index)
        
        # Process current node (both internal and leaf nodes)
        node.label = len(self.labeled_nodes)
        self.labeled_nodes.append(node)
        index += 1
        
        if len(node.children) > 0:
            # Process SW child
            index = self._inorder_traversal(node.children[2], index)
            
            # Process SE child
            index = self._inorder_traversal(node.children[3], index)
        
        return index

def create_quadtree_animation(points, boundary, max_points=1):
    """Create an animation of building the quadtree step by step."""
    # Create a separate quadtree for animation
    animation_quadtree = QuadTree(boundary, max_points)
    
    # Create the final quadtree with all points (for interactive phase)
    final_quadtree = QuadTree(boundary, max_points)
    for point in points:
        final_quadtree.insert(point)
    
    # Apply seaborn theme
    sns.set_theme(style="whitegrid")
    
    # Setup the figure and axes with seaborn styling
    fig, ax = plt.subplots(figsize=(10, 12))
    plt.subplots_adjust(bottom=0.3)  # Make more room for controls
    ax.set_xlim(boundary[0]-0.25, boundary[2]+0.25)
    ax.set_ylim(boundary[1]-0.25, boundary[3]+0.25)
    ax.set_title('Quadtree Building Animation and drawing disks by user interaction', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Update color palette based on seaborn
    palette = sns.color_palette()
    point_color = palette[0]  # First color from the palette
    rect_color = palette[6]   
    highlight_color1 = palette[3]  # Fourth color (usually red in many palettes)
    highlight_color2 = palette[4]
    text_color = palette[5]   # Fifth color
    
    # Initialize scatter plot with ALL points visible from the start
    scatter = ax.scatter([p[0] for p in points], [p[1] for p in points], 
                        c=point_color, s=40, alpha=0.7, zorder=2, edgecolor='white')
    
    rectangles = []
    texts = []
    
    # Create buttons and text boxes for node input (but keep them hidden initially)
    ax_reveal = plt.axes([0.1, 0.05, 0.2, 0.075])
    ax_clear = plt.axes([0.35, 0.05, 0.2, 0.075])
    ax_draw = plt.axes([0.6, 0.05, 0.2, 0.075])
    ax_text1 = plt.axes([0.15, 0.15, 0.1, 0.05])
    ax_text2 = plt.axes([0.55, 0.15, 0.1, 0.05])
    ax_submit = plt.axes([0.35, 0.15, 0.1, 0.05])
    ax_status = plt.figtext(0.5, 0.25, "", ha="center", va="center", 
                          bbox=dict(boxstyle="round,pad=0.5", fc=sns.color_palette("pastel")[2], alpha=0.7))
    
    # Hide all controls initially
    ax_reveal.set_visible(False)
    ax_clear.set_visible(False)
    ax_draw.set_visible(False)
    ax_text1.set_visible(False)
    ax_text2.set_visible(False)
    ax_submit.set_visible(False)
    ax_status.set_visible(False)
    
    # Create button objects (but don't connect them yet)
    btn_reveal = Button(ax_reveal, 'Reveal Labels', color=sns.color_palette("pastel")[0])
    btn_clear = Button(ax_clear, 'Clear Disks', color=sns.color_palette("pastel")[1])
    btn_draw = Button(ax_draw, 'Draw Disks', color=sns.color_palette("pastel")[3])
    btn_submit = Button(ax_submit, 'Submit', color=sns.color_palette("pastel")[4])
    text_node1 = TextBox(ax_text1, 'Node 1:', initial="")
    text_node2 = TextBox(ax_text2, 'Node 2:', initial="")
    
    # Store state variables
    state = {
        'labels_revealed': False,
        'circles': [],
        'node_selection_active': False,
        'selected_nodes': [],
        'status_text': ax_status,
        'buttons_connected': False,
        'valid_nodes': []
    }
    
    def init():
        # Draw initial boundary
        x_min, y_min, x_max, y_max = boundary
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=1.5, edgecolor=rect_color, facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        rectangles.append(rect)
        return [scatter] + rectangles + texts
    
    def animate(i):
        if i < len(points):
            # Insert the next point into the animation quadtree
            point = points[i]
            animation_quadtree.insert(point)
            
            # Clear previous rectangles except the initial boundary
            for rect in rectangles[1:]:
                rect.remove()
            rectangles[1:] = []
            
            # Draw all quadrants in the current state
            nodes = animation_quadtree.root.get_all_nodes()
            for node in nodes:
                if node != animation_quadtree.root:  # Skip the root node (already drawn)
                    x_min, y_min, x_max, y_max = node.boundary
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                            linewidth=1, edgecolor=rect_color, facecolor='none', alpha=0.6)
                    ax.add_patch(rect)
                    rectangles.append(rect)
            
            # On the last frame, set up interactive controls
            if i == len(points) - 1:
                # Label nodes (but don't show labels yet)
                final_quadtree.label_all_nodes()  # Now labels all nodes
                state['valid_nodes'] = list(range(len(final_quadtree.labeled_nodes)))
                setup_interactive_controls()
                
        return [scatter] + rectangles + texts
    
    def setup_interactive_controls():
        """Set up interactive controls after animation completes."""
        # Show all controls
        ax_reveal.set_visible(True)
        ax_clear.set_visible(True)
        ax_draw.set_visible(True)
        
        # Connect button events if not already connected
        if not state['buttons_connected']:
            btn_reveal.on_clicked(on_reveal)
            btn_clear.on_clicked(on_clear)
            btn_draw.on_clicked(on_draw)
            btn_submit.on_clicked(on_submit)
            fig.canvas.mpl_connect('button_press_event', on_mouse_click)
            state['buttons_connected'] = True
        
        update_status(f"Animation complete. You can draw disks for nodes in range {state['valid_nodes'][0]} - {state['valid_nodes'][-1]}.")
    
    def on_reveal(event):
        if not state['labels_revealed']:
            state['labels_revealed'] = True
            reveal_labels(final_quadtree, ax, text_color)
            update_status(f"Labels revealed. You can draw disks for nodes in range {state['valid_nodes'][0]} - {state['valid_nodes'][-1]}.")
        else:
            # state['labels_revealed'] = False
            update_status("Labels are already visible.")
    
    def on_clear(event):
        # Clear all disks
        for circle in state['circles']:
            circle.remove()
        state['circles'] = []
        fig.canvas.draw()
        update_status(f"Disks cleared. You can draw disks for nodes in range {state['valid_nodes'][0]} - {state['valid_nodes'][-1]}.")
        
        # Hide input elements
        ax_text1.set_visible(False)
        ax_text2.set_visible(False)
        ax_submit.set_visible(False)
        state['node_selection_active'] = False
    
    def on_draw(event):
        # Show input elements
        ax_text1.set_visible(True)
        ax_text2.set_visible(True)
        ax_submit.set_visible(True)
        state['node_selection_active'] = True
        update_status(f"Enter node numbers in range {state['valid_nodes'][0]} - {state['valid_nodes'][-1]} and click Submit.")
    
    def on_submit(event):
        if not state['node_selection_active']:
            return
        
        try:
            idx1 = int(text_node1.text)
            idx2 = int(text_node2.text)
            
            max_index = len(final_quadtree.labeled_nodes) - 1
            
            if idx1 < 0 or idx1 > max_index or idx2 < 0 or idx2 > max_index:
                update_status(f"Invalid node numbers. Please enter values between 0 and {max_index}.")
                return
            
            # Clear previous disks
            on_clear(None)
            
            # Draw disks for selected nodes
            node1 = final_quadtree.labeled_nodes[idx1]
            node2 = final_quadtree.labeled_nodes[idx2]
            
            circle1 = draw_disk(ax, node1, highlight_color1) 
            circle2 = draw_disk(ax, node2, highlight_color2)
            
            state['circles'].extend([circle1, circle2])
            
            # Hide input elements
            ax_text1.set_visible(False)
            ax_text2.set_visible(False)
            ax_submit.set_visible(False)
            state['node_selection_active'] = False
            
            fig.canvas.draw()
            update_status(f"Disks drawn for nodes {idx1} and {idx2}.")
            
        except ValueError:
            update_status(f"Please enter valid numbers between 0 and {len(final_quadtree.labeled_nodes)-1}.")
    
    def update_status(message):
        """Update the status text displayed on the figure."""
        state['status_text'].set_text(message)
        state['status_text'].set_visible(True)
        fig.canvas.draw()
    
    def on_mouse_click(event):
        if not state['node_selection_active']:
            return
            
        if event.inaxes != ax:
            return
            
        # Find the closest node to the click
        click_pos = (event.xdata, event.ydata)
        min_dist = float('inf')
        closest_node_idx = None
        
        for i, node in enumerate(final_quadtree.labeled_nodes):
            x_min, y_min, x_max, y_max = node.boundary
            node_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            dist = np.sqrt((node_center[0] - click_pos[0])**2 + (node_center[1] - click_pos[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_node_idx = i
        
        if closest_node_idx is not None:
            if not text_node1.text:
                text_node1.set_val(str(closest_node_idx))
                update_status(f"Selected Node 1: {closest_node_idx}. Select Node 2 or enter manually. Valid nodes: {state['valid_nodes']}")
            elif not text_node2.text:
                text_node2.set_val(str(closest_node_idx))
                update_status(f"Selected Node 2: {closest_node_idx}. Click Submit to draw disks.")
    
    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(points), init_func=init,
                        interval=500, repeat=False, blit=False)
    
    return fig, ax, ani, final_quadtree

def draw_disk(ax, node, color, label=None):
    """Draw a disk enclosing a node based on the maximum diameter of the partition."""
    x_min, y_min, x_max, y_max = node.boundary
    
    # Calculate center
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Calculate maximum diameter of the partition
    width = x_max - x_min
    height = y_max - y_min
    
    # Calculate diagonal distance (maximum possible distance between any two points in the rectangle)
    # Using the pythagorean theorem: diagonal = sqrt(width^2 + height^2)
    max_diameter = np.sqrt(width**2 + height**2)
    
    # Use half the diagonal as radius (covering all corners of the rectangle)
    radius = max_diameter / 2
    
    # Draw circle
    circle = patches.Circle((center_x, center_y), radius,
                          linewidth=2, edgecolor=color,
                          facecolor=color, alpha=0.2, zorder=4)
    ax.add_patch(circle)
    
    return circle

def reveal_labels(quadtree, ax, text_color):
    """Reveal labels of all nodes."""
    labeled_nodes = quadtree.labeled_nodes
    
    for node in labeled_nodes:
        x_min, y_min, x_max, y_max = node.boundary
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Different style for internal vs leaf nodes
        if node.is_leaf():
            ax.text(x_center, y_center, str(node.label), ha='center', va='center',
                   fontweight='bold', fontsize=7, color=text_color, zorder=3)
        else:
            # Internal nodes with a different style (e.g., square background)
            ax.text(x_center, y_center, str(node.label), ha='center', va='center',
                   fontweight='bold', fontsize=7, color=text_color, zorder=3,
                   bbox=dict(boxstyle="square,pad=0.3", fc='white', alpha=0.7))
    
    plt.draw()
    return True

def main():
    # Get number of points from user via terminal
    try:
        N = int(input("Enter the number of points for the quadtree: "))
        if N <= 0:
            print("Number of points must be positive. Using default value 50.")
            N = 50
    except ValueError:
        print("Invalid input. Using default value 50.")
        N = 50
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random points in range [0, 1]
    points = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(N)]
    
    # Define the boundary [x_min, y_min, x_max, y_max]
    boundary = [0, 0, 1, 1]
    
    # Create the animation
    fig, ax, ani, quadtree = create_quadtree_animation(points, boundary, max_points=1)
    
    # Show the plot and start the animation
    plt.show()

if __name__ == "__main__":
    main()