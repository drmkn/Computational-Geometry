import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.animation import FuncAnimation
import random
import math

class TriangleRobot:
    def __init__(self, vertices):
        """Initialize the triangle robot with its vertices"""
        self.vertices = np.array(vertices)
        # Calculate centroid
        self.centroid = np.mean(self.vertices, axis=0)
        # Calculate relative positions of vertices from centroid
        self.relative_vertices = self.vertices - self.centroid
    
    def get_vertices_at(self, position):
        """Get the vertices of the triangle at a specific position"""
        position = np.array(position)
        return self.relative_vertices + position
    
    def check_collision(self, position, obstacles):
        """Check if the triangle at the given position collides with any obstacle"""
        # Ensure position is a single point, not an array of points
        position = np.array(position)
        if position.ndim > 1:
            raise ValueError("Position should be a single point, not an array of points")
            
        vertices = self.get_vertices_at(position)
        
        # Create triangle edges for line segment intersection tests
        edges = [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0])
        ]
        
        # Check each obstacle
        for obstacle in obstacles:
            # Get obstacle coordinates
            (x_min, y_min), (x_max, y_max) = obstacle
            
            # Check if any vertex is inside the obstacle
            for vertex in vertices:
                if (x_min <= vertex[0] <= x_max) and (y_min <= vertex[1] <= y_max):
                    return True
            
            # Check if any edge of the triangle intersects with any edge of the rectangle
            rect_edges = [
                ((x_min, y_min), (x_max, y_min)),  # bottom
                ((x_max, y_min), (x_max, y_max)),  # right
                ((x_max, y_max), (x_min, y_max)),  # top
                ((x_min, y_max), (x_min, y_min))   # left
            ]
            
            for tri_edge in edges:
                for rect_edge in rect_edges:
                    if line_segment_intersect(tri_edge[0], tri_edge[1], rect_edge[0], rect_edge[1]):
                        return True
            
            # Check if rectangle is completely inside triangle (rare but possible)
            if point_in_triangle(vertices, (x_min, y_min)) and point_in_triangle(vertices, (x_max, y_min)) and \
               point_in_triangle(vertices, (x_max, y_max)) and point_in_triangle(vertices, (x_min, y_max)):
                return True
        
        return False

def line_segment_intersect(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect"""
    # Convert to numpy arrays for easier calculation
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    
    # Calculate directions
    d1 = p2 - p1
    d2 = p4 - p3
    
    # Calculate the denominator of the parametric equation
    den = d1[0] * d2[1] - d1[1] * d2[0]
    
    # If den is zero, lines are parallel
    if den == 0:
        return False
    
    # Calculate numerators
    num1 = (p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]
    num2 = (p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]
    
    # Calculate parameters
    t1 = num1 / den
    t2 = num2 / den
    
    # Check if intersection point is within both line segments
    return (0 <= t1 <= 1) and (0 <= t2 <= 1)

def point_in_triangle(triangle_vertices, point):
    """Check if a point is inside a triangle using barycentric coordinates"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(point, triangle_vertices[0], triangle_vertices[1])
    d2 = sign(point, triangle_vertices[1], triangle_vertices[2])
    d3 = sign(point, triangle_vertices[2], triangle_vertices[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def check_path_collision(robot, start, end, obstacles, step_size):
    """Check if there's a collision along the path from start to end"""
    dist = distance(start, end)
    num_checks = max(int(dist / step_size), 2)
    
    # Generate points along the path
    for i in range(1, num_checks):
        t = i / num_checks
        point = (1-t) * np.array(start) + t * np.array(end)
        if robot.check_collision(point, obstacles):
            return True  # Collision detected
    
    return False  # No collision

def rrt_path_planning(robot, start, goal, obstacles, max_iterations=5000, step_size=0.5):
    """RRT (Rapidly-Exploring Random Tree) algorithm for path planning"""
    # Initialize tree with start position
    tree = {tuple(start): None}  # node: parent
    
    # Check if goal is reachable directly
    if not robot.check_collision(start, obstacles) and not robot.check_collision(goal, obstacles):
        if distance(start, goal) < step_size or not check_path_collision(robot, start, goal, obstacles, step_size/2):
            path = [start, goal]
            return True, path
    
    for _ in range(max_iterations):
        # With some probability, sample the goal
        if random.random() < 0.1:
            random_point = goal
        else:
            # Sample random point in configuration space
            random_point = (
                random.uniform(min(start[0], goal[0]) - 5, max(start[0], goal[0]) + 5),
                random.uniform(min(start[1], goal[1]) - 5, max(start[1], goal[1]) + 5)
            )
        
        # Find nearest node in the tree
        nearest_node = min(tree.keys(), key=lambda node: distance(node, random_point))
        
        # Calculate direction toward random point
        direction = np.array(random_point) - np.array(nearest_node)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        
        # Generate new node at a fixed distance in the direction of the random point
        new_node = tuple(np.array(nearest_node) + step_size * direction / norm)
        
        # Check if the path to the new node is collision-free
        if not robot.check_collision(new_node, obstacles):
            # Check if the straight line path is collision-free
            if not check_path_collision(robot, nearest_node, new_node, obstacles, step_size/2):
                # Add new node to the tree
                tree[new_node] = nearest_node
                
                # Check if we can reach the goal from this new node
                if distance(new_node, goal) < step_size:
                    # Try to connect to the goal
                    if not check_path_collision(robot, new_node, goal, obstacles, step_size/2):
                        # Goal reached
                        tree[tuple(goal)] = new_node
                        
                        # Extract path
                        path = [goal]
                        node = tuple(goal)
                        
                        while node != tuple(start):
                            node = tree[node]
                            path.append(node)
                        
                        path.reverse()
                        
                        return True, path
    
    return False, []

def visualize_motion_planning(robot, path, obstacles, file_name="triangle_motion.gif"):
    """Create animation of the triangle moving along the path"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set plot limits
    all_x = [p[0] for p in path]
    all_y = [p[1] for p in path]
    
    min_x, max_x = min(all_x) - 2, max(all_x) + 2
    min_y, max_y = min(all_y) - 2, max(all_y) + 2
    
    # Adjust for obstacle positions
    for (x_min, y_min), (x_max, y_max) in obstacles:
        min_x = min(min_x, x_min - 1)
        min_y = min(min_y, y_min - 1)
        max_x = max(max_x, x_max + 1)
        max_y = max(max_y, y_max + 1)
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Draw obstacles
    for (x_min, y_min), (x_max, y_max) in obstacles:
        width = x_max - x_min
        height = y_max - y_min
        rect = Rectangle((x_min, y_min), width, height, facecolor='pink', edgecolor='red', alpha=0.7)
        ax.add_patch(rect)
    
    # Initialize the triangle at the start position
    triangle_vertices = robot.get_vertices_at(path[0])
    triangle = Polygon(triangle_vertices, facecolor='lightblue', edgecolor='blue', alpha=0.7)
    ax.add_patch(triangle)
    
    # Add path lines
    ax.plot([p[0] for p in path], [p[1] for p in path], 'k--', alpha=0.5)
    
    # Mark start and goal
    ax.plot(path[0][0], path[0][1], 'go', markersize=8, label='Start')
    ax.plot(path[-1][0], path[-1][1], 'ro', markersize=8, label='Goal')
    
    # Set title and legend
    ax.set_title('Motion Planning for Triangular Robot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    # Animation function
    def update(frame):
        if frame < len(path):
            # Update triangle position
            new_vertices = robot.get_vertices_at(path[frame])
            triangle.set_xy(new_vertices)
            return [triangle]
        return [triangle]
    
    # Create animation
    frames = len(path)
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    
    # Save animation (optional)
    ani.save(file_name, writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return ani

# Main function to run the motion planning
def main():
    # Define the triangle robot
    triangle_vertices = [(0, 0), (1, 2), (2, 0)]  # Example triangle
    robot = TriangleRobot(triangle_vertices)
    
    # Define start and goal positions
    start_position = (3, 3)
    goal_position = (15, 5)
    
    # Define obstacles as ((min_x, min_y), (max_x, max_y))
    obstacles = [
        ((6, 0), (9, 5)),    # Bottom center
        ((0, 8), (5, 12)),   # Top left
        ((12, 8), (16, 12)), # Top right
        ((6, 14), (9, 18))   # Top center
    ]
    
    # Run RRT path planning
    found_path, path = rrt_path_planning(robot, start_position, goal_position, obstacles)
    
    if found_path:
        print("Collision-free path found!")
        # Visualize the motion
        visualize_motion_planning(robot, path, obstacles)
    else:
        print("No collision-free path found within the maximum number of iterations.")

if __name__ == "__main__":
    main()