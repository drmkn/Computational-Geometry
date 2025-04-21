'''Given a list of 2D points (generated) randomly and a set of line segments(representing obstacles), show the points that are visible
from all points, considering line of sight obstruction by the segments.  Consider the invisible event in this section, take random n 
line segments in the plane, those segments will have 2 end points each so (2*n) end points are there, we have to find those points that 
are visible from all the other (2n-1) points. if such point doesn't exist then report that, if such point exists then show the line of 
sight (as dotted line) joining that point to all other 2n-1 point. You may add animations. You can use brute force techniques for this.
Input: set of n line segments with their respective end points (take lines as 2 point form).'''


import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

class VisibilityGraph:
    def __init__(self, num_segments=5):
        self.segments = []
        self.points = []
        self.num_segments = num_segments
        self.visible_from_all = []
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
    def generate_random_segments(self):
        """Generate random line segments as obstacles"""
        self.segments = []
        self.points = []
        
        for i in range(self.num_segments):
            x1, y1 = random.uniform(0, 80), random.uniform(0, 80)
            # Generate second point at a reasonable distance
            angle = random.uniform(0, 2 * np.pi)
            length = random.uniform(10, 20)
            x2 = x1 + length * np.cos(angle)
            y2 = y1 + length * np.sin(angle)
            
            self.segments.append(((x1, y1), (x2, y2)))
            self.points.append((x1, y1, f"P{i*2}"))
            self.points.append((x2, y2, f"P{i*2+1}"))
    
    def do_segments_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect"""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
            
        # Special Cases
        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True
            
        return False
    
    def is_visible(self, point1, point2):
        """Check if point2 is visible from point1"""
        # Extract coordinates (ignore labels)
        p1 = (point1[0], point1[1])
        p2 = (point2[0], point2[1])
        
        # Points are same
        if p1 == p2:
            return False
            
        # Check against all segments
        for segment in self.segments:
            seg_p1, seg_p2 = segment
            
            # Don't check segments that have either point as an endpoint
            if (p1 == seg_p1 or p1 == seg_p2 or p2 == seg_p1 or p2 == seg_p2):
                continue
                
            if self.do_segments_intersect(p1, p2, seg_p1, seg_p2):
                return False
                
        return True
    
    def find_visible_from_all(self):
        """Find points that are visible from all other points"""
        visible_from_all = []
        
        for i, point in enumerate(self.points):
            is_visible_from_all = True
            
            for j, other_point in enumerate(self.points):
                if i != j and not self.is_visible(point, other_point):
                    is_visible_from_all = False
                    break
                    
            if is_visible_from_all:
                visible_from_all.append(point)
                
        return visible_from_all
    
    def get_plot_limits(self):
        """Calculate appropriate plot limits based on point coordinates"""
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        
        if not x_coords or not y_coords:
            return (0, 100, 0, 100)
            
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding (10% of range)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        # Ensure minimum size of 10 units in each dimension
        if x_max - x_min < 10:
            x_padding = (10 - (x_max - x_min)) / 2
        if y_max - y_min < 10:
            y_padding = (10 - (y_max - y_min)) / 2
            
        return (x_min - x_padding, x_max + x_padding, 
                y_min - y_padding, y_max + y_padding)
    
    def draw_base_visualization(self):
        """Draw the segments and points without visibility lines"""
        self.ax.clear()
        
        # Set axis limits based on point coordinates
        x_min, x_max, y_min, y_max = self.get_plot_limits()
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        # Draw line segments (obstacles)
        for segment in self.segments:
            x_values = [segment[0][0], segment[1][0]]
            y_values = [segment[0][1], segment[1][1]]
            self.ax.plot(x_values, y_values, 'k-', linewidth=2)
        
        # Draw all points
        x_points = [p[0] for p in self.points]
        y_points = [p[1] for p in self.points]
        self.ax.scatter(x_points, y_points, color='blue', s=50, zorder=5)
        
        # Label the points
        for point in self.points:
            self.ax.annotate(point[2], (point[0], point[1]), 
                             textcoords="offset points", 
                             xytext=(0, 5), 
                             ha='center')
        
        # Highlight points visible from all others
        for point in self.visible_from_all:
            self.ax.scatter(point[0], point[1], color='green', s=100, zorder=10)
            
        if self.visible_from_all:
            title = f"Point(s) visible from all others: " + ", ".join([p[2] for p in self.visible_from_all])
        else:
            title = "No point is visible from all others"
            
        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        plt.tight_layout()
        
    def animate_visibility_lines(self):
        """Animate the visibility lines one by one"""
        if not self.visible_from_all:
            self.draw_base_visualization()
            plt.show()
            return
            
        # Draw base visualization
        self.draw_base_visualization()
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='Visible from all'),
            Line2D([0], [0], linestyle='--', color='r', label='Line of sight')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        # Number of frames = number of points visible from all * (number of other points)
        total_frames = sum([len(self.points) - 1 for _ in self.visible_from_all])
        
        # Initialize lines list to store the line artists
        lines = []
        
        # Animation update function
        def update(frame):
            # Determine which point and which visibility line to draw
            point_idx = 0
            remaining_frame = frame
            
            while point_idx < len(self.visible_from_all):
                lines_for_point = len(self.points) - 1
                if remaining_frame < lines_for_point:
                    # This is the point we're drawing lines for
                    break
                remaining_frame -= lines_for_point
                point_idx += 1
            
            if point_idx < len(self.visible_from_all):
                visible_point = self.visible_from_all[point_idx]
                
                # Find which other point we're connecting to
                other_points = [p for p in self.points if p != visible_point]
                other_point = other_points[remaining_frame]
                
                # Draw the visibility line
                line = self.ax.plot([visible_point[0], other_point[0]], 
                                   [visible_point[1], other_point[1]], 
                                   'r--', alpha=0.7)[0]
                lines.append(line)
            
            return lines
        
        # Create animation
        ani = FuncAnimation(self.fig, update, frames=total_frames, 
                            interval=500, blit=True, repeat=False)
        
        plt.show()
    
if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    
    n = int(input("Enter number of line segments :"))
    # Create visibility graph with 5 segments (10 points)
    vis_graph = VisibilityGraph(num_segments=n)
    vis_graph.generate_random_segments()
    vis_graph.visible_from_all = vis_graph.find_visible_from_all()
    
    # Animate the visibility lines
    vis_graph.animate_visibility_lines()