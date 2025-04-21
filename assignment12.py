import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import seaborn as sns
from itertools import chain, combinations
import time

class AxisAlignedRectangleVisualization:
    def __init__(self, n_points=4, seed=None):
        """
        Initialize the visualization with n random points.
        
        Args:
            n_points (int): Number of points to generate
            seed (int, optional): Random seed for reproducibility
        """
        # Set Seaborn style
        sns.set_theme(style="whitegrid")
        
        if seed is not None:
            np.random.seed(seed)
            
        self.n_points = n_points
        self.points = self.generate_random_points(n_points)
        self.all_subsets = self.generate_all_subsets()
        self.capturable_subsets = []
        self.non_capturable_subsets = []
        self.classify_subsets()
        
        # Combine both types of subsets for animation
        self.all_classified_subsets = [(subset, True) for subset in self.capturable_subsets] + \
                                      [(subset, False) for subset in self.non_capturable_subsets]
        
        # Set up the figure and axes with Seaborn styling
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('Axis-Aligned Rectangle Visualization')
        
        # Animation properties
        self.current_subset_idx = 0
        self.animation_running = False
        self.anim = None
        
    def generate_random_points(self, n):
        """Generate n random points in [0,1] x [0,1]"""
        points = []
        for i in range(n):
            points.append({
                'id': i + 1,
                'x': np.random.uniform(0.1, 0.9),
                'y': np.random.uniform(0.1, 0.9)
            })
        return points
    
    def generate_all_subsets(self):
        """Generate all non-empty subsets of points"""
        point_indices = range(len(self.points))
        # Use itertools.combinations to generate all subsets of all sizes
        all_subsets = []
        for r in range(1, len(self.points) + 1):
            for combo in combinations(point_indices, r):
                subset = [self.points[i] for i in combo]
                all_subsets.append(subset)
        return all_subsets
    
    def can_be_captured(self, subset):
        """Check if a subset can be captured by an axis-aligned rectangle"""
        if len(subset) <= 1:
            return True
        
        # Find min and max x,y coordinates
        min_x = min(p['x'] for p in subset)
        max_x = max(p['x'] for p in subset)
        min_y = min(p['y'] for p in subset)
        max_y = max(p['y'] for p in subset)
        
        # Check if the rectangle defined by these bounds captures exactly the subset
        for p in self.points:
            is_in_rect = (min_x <= p['x'] <= max_x) and (min_y <= p['y'] <= max_y)
            is_in_subset = p in subset
            
            if is_in_rect and not is_in_subset:
                return False
        
        return True
    
    def classify_subsets(self):
        """Classify subsets as capturable or non-capturable"""
        for subset in self.all_subsets:
            if self.can_be_captured(subset):
                self.capturable_subsets.append(subset)
            else:
                self.non_capturable_subsets.append(subset)
    
    def get_bounding_rect(self, subset):
        """Get the minimal bounding rectangle for a subset"""
        if not subset:
            return None
        
        min_x = min(p['x'] for p in subset)
        max_x = max(p['x'] for p in subset)
        min_y = min(p['y'] for p in subset)
        max_y = max(p['y'] for p in subset)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def format_subset(self, subset):
        """Format a subset for display"""
        return '{' + ', '.join(str(p['id']) for p in sorted(subset, key=lambda p: p['id'])) + '}'
    
    def draw_frame(self, subset_idx=0):
        """Draw a single frame of the visualization"""
        self.ax.clear()
        
        # Set axis limits
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        
        # Get current subset and its capturable status
        if subset_idx < len(self.all_classified_subsets):
            current_subset, is_capturable = self.all_classified_subsets[subset_idx]
            subset_str = self.format_subset(current_subset)
            status = "Capturable" if is_capturable else "NON-Capturable"
            self.ax.set_title(f'Subset {subset_str} - {status} ({subset_idx + 1}/{len(self.all_classified_subsets)})', fontsize=14)
        else:
            self.ax.set_title('No subset selected', fontsize=14)
            current_subset = []
            is_capturable = True
        
        # Define colors based on capturable status
        capturable_color = 'red'
        non_capturable_color = 'green'
        highlight_color = capturable_color if is_capturable else non_capturable_color
        
        # Extract points for plot
        x_values = [p['x'] for p in self.points]
        y_values = [p['y'] for p in self.points]
        
        # Plot all points first
        for i, p in enumerate(self.points):
            in_subset = p in current_subset
            color = highlight_color if in_subset else 'blue'
            self.ax.scatter(p['x'], p['y'], color=color, s=150, zorder=3)
            self.ax.text(p['x'] + 0.02, p['y'] + 0.02, str(p['id']), fontsize=12, 
                        weight='bold', color='black', zorder=4)
        
        # Draw bounding rectangle for current subset
        if current_subset:
            rect = self.get_bounding_rect(current_subset)
            if rect:
                rect_color = highlight_color
                self.ax.add_patch(
                    patches.Rectangle(
                        (rect[0], rect[1]), rect[2], rect[3],
                        linewidth=2,
                        edgecolor=rect_color,
                        facecolor=rect_color,
                        alpha=0.2,
                        zorder=2
                    )
                )
        
        # Add info text
        info_text = f"""
        Total points: {self.n_points}
        Capturable subsets: {len(self.capturable_subsets)}
        Non-capturable subsets: {len(self.non_capturable_subsets)}
        Current: {"Capturable (red)" if is_capturable else "NON-Capturable (green)"}
        """
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=5)
        
        # Set axis labels
        self.ax.set_xlabel('X-axis', fontsize=12)
        self.ax.set_ylabel('Y-axis', fontsize=12)
        
        plt.tight_layout()
        return self.ax
    
    def update_frame(self, frame):
        """Update function for animation"""
        return self.draw_frame(frame % len(self.all_classified_subsets))
    
    def toggle_animation(self):
        """Toggle animation play/pause"""
        if self.animation_running:
            self.anim.event_source.stop()
            self.animation_running = False
        else:
            self.anim.event_source.start()
            self.animation_running = True
    
    def print_all_subsets(self):
        """Print all subsets and their capturability"""
        print("\nAll subsets (excluding empty set):")
        for subset in self.all_subsets:
            subset_str = self.format_subset(subset)
            status = "capturable" if subset in self.capturable_subsets else "NOT capturable"
            print(f"{subset_str}: {status}")
    
    def run_visualization(self, animate=True, interval=1000):
        """Run the visualization with interactive controls"""
        # Draw initial frame
        self.draw_frame(0)
        
        # Set up animation
        if animate:
            self.anim = animation.FuncAnimation(
                self.fig, self.update_frame, 
                frames=len(self.all_classified_subsets),
                interval=interval, repeat=True
            )
            self.anim.event_source.stop()  # Start paused
            
            # Start animation
            self.anim.event_source.start()
            self.animation_running = True
        
        # Print all subsets
        self.print_all_subsets()
        
        plt.show()

def main():
    """Main function to run the visualization"""
    # You can adjust the number of points and the random seed
    n_points = 5
    # seed = 42  # None for random seed
    
    viz = AxisAlignedRectangleVisualization(n_points=n_points, seed=None)
    viz.run_visualization(animate=True, interval=600)  # Set interval (milliseconds) between frames

if __name__ == "__main__":
    main()