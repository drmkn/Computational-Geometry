import numpy as np
import matplotlib.pyplot as plt
import math

def is_less(X, Y, a, b, i, j, n):
    """
    Compare two vertices based on their projection onto the line perpendicular to ax + by + c = 0
    """
    Di = b * X[i % n] + -a * Y[i % n]
    Dj = b * X[j % n] + -a * Y[j % n]
    # print(i%n,j%n,Di,Dj)
    return (Di < Dj) or (Di == Dj and i < j)

def is_monotone(X, Y, a, b, c):
    """
    Check if a polygon defined by vertices (X, Y) is monotone with respect to the line ax + by + c = 0
    
    Args:
        X, Y: Lists of x and y coordinates of the polygon vertices
        a, b, c: Coefficients of the line equation ax + by + c = 0
        
    Returns:
        is_monotone: Boolean indicating if the polygon is monotone
        violation_points: List of indices where monotonicity is violated (excluding the expected starting point)
    """
    n = len(X)
    local_mins = []
    
    for i in range(n):
        prev = (i - 1) % n
        next = (i + 1) % n
        
        # Check if the current vertex is a local minimum
        if is_less(X, Y, a, b, i, prev, n) and is_less(X, Y, a, b, i, next, n):
            local_mins.append(i)
        # print(local_mins)    
    
    # A polygon should have exactly one local minimum to be monotone (the starting point)
    # If there are more, return the list of additional violation points
    
    if len(local_mins) == 1:
        # This is the expected starting point, polygon is monotone
        return True, None
    else:
        # Find which point is the "starting point" (the one that would exist in a monotone polygon)
        # Usually this is the one with minimum projection value onto the line
        min_proj_idx = None
        min_proj_value = float('inf')
        
        for idx in local_mins:
            proj_value = b * X[idx] + -a * Y[idx]
            if proj_value < min_proj_value:
                min_proj_value = proj_value
                min_proj_idx = idx
        
        # Remove the expected starting point from the list of violations
        if min_proj_idx in local_mins:
            local_mins.remove(min_proj_idx)
            
        # Return the first additional violation point (if any)
        violation_point = local_mins[0] if local_mins else None
        return len(local_mins) == 0, violation_point

def project_point_on_line(x, y, a, b, c):
    """
    Project a point (x, y) onto the line ax + by + c = 0
    Returns the coordinates of the projection
    """
    denominator = a * a + b * b
    numerator_x = (b * b * x - a * b * y - a * c)
    numerator_y = (a * a * y - a * b * x - b * c)
    
    proj_x = numerator_x / denominator
    proj_y = numerator_y / denominator
    
    return proj_x, proj_y

def distance_between_points(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_closest_neighbor_along_line(X, Y, a, b, c, violation_point, n):
    """
    Find the closest neighbor to the violation point along the direction of the line
    """
    vp_x, vp_y = X[violation_point], Y[violation_point]
    
    # Project violation point onto the line
    vp_proj_x, vp_proj_y = project_point_on_line(vp_x, vp_y, a, b, c)
    
    # Check both neighbors (previous and next)
    prev_idx = (violation_point - 1) % n
    next_idx = (violation_point + 1) % n
    
    # Project neighbors onto the line
    prev_x, prev_y = X[prev_idx], Y[prev_idx]
    next_x, next_y = X[next_idx], Y[next_idx]
    
    prev_proj_x, prev_proj_y = project_point_on_line(prev_x, prev_y, a, b, c)
    next_proj_x, next_proj_y = project_point_on_line(next_x, next_y, a, b, c)
    
    # Calculate distances along the line
    dist_to_prev = distance_between_points(vp_proj_x, vp_proj_y, prev_proj_x, prev_proj_y)
    dist_to_next = distance_between_points(vp_proj_x, vp_proj_y, next_proj_x, next_proj_y)
    
    # Return the index of the closest neighbor
    if dist_to_prev <= dist_to_next:
        return prev_idx, (prev_x + vp_x) / 2, (prev_y + vp_y) / 2
    else:
        return next_idx, (next_x + vp_x) / 2, (next_y + vp_y) / 2

def line_intersection(line1, line2, tol=1e-10):
    """
    Find the intersection of two lines defined by points (x1, y1)-(x2, y2) and (x3, y3)-(x4, y4)
    Returns intersection point (x, y) or None if lines are parallel or intersection is outside segments
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < tol:  # Lines are parallel
        return None
    
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    
    # Check if the intersection lies within both line segments
    t1 = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
    t2 = ((px - x3) * (x4 - x3) + (py - y3) * (y4 - y3)) / ((x4 - x3)**2 + (y4 - y3)**2)
    
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return px, py
    return None

def find_perpendicular_intersections(X, Y, a, b, c, midpoint_x, midpoint_y, x_min, x_max, y_min, y_max):
    """
    Find all intersections of the perpendicular line with the polygon edges
    """
    n = len(X)
    intersections = []
    
    # Define the perpendicular line through the midpoint
    # Perpendicular line has coefficients (-b, a) to be perpendicular to (a, b)
    perp_a, perp_b = -b, a
    perp_c = -perp_a * midpoint_x - perp_b * midpoint_y  # Line equation through midpoint
    
    # Generate points for the perpendicular line, extended beyond plot bounds
    padding = max(x_max - x_min, y_max - y_min) * 2  # Extend line significantly
    if abs(perp_b) > 1e-10:  # Not a vertical line
        x_values = np.array([x_min - padding, x_max + padding])
        y_values = (-perp_a * x_values - perp_c) / perp_b
    else:  # Vertical line
        x_values = np.array([-perp_c / perp_a, -perp_c / perp_a])
        y_values = np.array([y_min - padding, y_max + padding])
    
    # Define the perpendicular line segment
    perp_line = (x_values[0], y_values[0], x_values[1], y_values[1])
    
    # Check intersections with each edge of the polygon
    for i in range(n):
        j = (i + 1) % n
        edge = (X[i], Y[i], X[j], Y[j])
        intersection = line_intersection(perp_line, edge)
        if intersection is not None:
            intersections.append(intersection)
    
    return intersections

def plot_polygon_with_line(X, Y, a, b, c, violation_point=None, is_monotone=True):
    """
    Plot the polygon, the given line, and (if applicable) the perpendicular line at the violation point
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the polygon (connect adjacent points in order)
    # Close the polygon by adding the first point at the end
    X_closed = X + [X[0]]
    Y_closed = Y + [Y[0]]
    
    # Plot the polygon outline
    ax.plot(X_closed, Y_closed, 'k-', linewidth=2, label='Polygon')
    # Fill the polygon
    ax.fill(X, Y, alpha=0.2)
    
    # Add vertex numbers for clarity
    for i, (x, y) in enumerate(zip(X, Y)):
        ax.text(x, y, f' {i}', fontsize=12)
    
    # Find bounds for the line
    padding = 2  # Add some padding to the plot
    x_min, x_max = min(X) - padding, max(X) + padding
    y_min, y_max = min(Y) - padding, max(Y) + padding
    
    # Plot the given line ax + by + c = 0
    if abs(b) > 1e-10:  # Not a vertical line
        x_values = np.linspace(x_min, x_max, 100)
        y_values = (-a * x_values - c) / b
        ax.plot(x_values, y_values, 'r-', linewidth=2, label='Given Line')
    else:  # Vertical line
        x_value = -c / a
        ax.axvline(x=x_value, color='r', linewidth=2, label='Given Line')
    
    # Find the "starting point" (minimum vertex in the direction of the line)
    n = len(X)
    min_proj_idx = 0
    min_proj_value = float('inf')
    
    for i in range(n):
        proj_value = a * X[i] + b * Y[i]
        if proj_value < min_proj_value:
            min_proj_value = proj_value
            min_proj_idx = i
    
    # Highlight the starting point (always present, even in monotone polygons)
    # ax.plot(X[min_proj_idx], Y[min_proj_idx], 'bo', markersize=8, label='Starting Point')
    
    # If a violation point is provided (separate from the starting point), plot it
    if violation_point is not None and not is_monotone:
        print(f"Drawing perpendicular line for violation point {violation_point}")
        vp_x, vp_y = X[violation_point], Y[violation_point]
        
        # Highlight the violation point
        # ax.plot(vp_x, vp_y, 'ro', markersize=10, label='Violation Point')
        
        # Find closest neighbor along the line direction
        neighbor_idx, midpoint_x, midpoint_y = find_closest_neighbor_along_line(X, Y, a, b, c, violation_point, n)
        
        # Highlight the closest neighbor
        neighbor_x, neighbor_y = X[neighbor_idx], Y[neighbor_idx]
        # ax.plot(neighbor_x, neighbor_y, 'go', markersize=8, label='Closest Neighbor')
        
        # Draw a line between violation point and its neighbor
        ax.plot([vp_x, neighbor_x], [vp_y, neighbor_y], 'g--', linewidth=1)
        
        # Find intersections of the perpendicular line with the polygon
        intersections = find_perpendicular_intersections(X, Y, a, b, c, midpoint_x, midpoint_y, x_min, x_max, y_min, y_max)
        
        if intersections:
            print(f"Perpendicular line intersects polygon at {len(intersections)} points.")
            # Draw the perpendicular line at the midpoint
            perp_a, perp_b = -b, a
            perp_c = -perp_a * midpoint_x - perp_b * midpoint_y
            if abs(perp_b) > 1e-10:  # Not a vertical line
                perp_x_values = np.linspace(x_min, x_max, 100)
                perp_y_values = (-perp_a * perp_x_values - perp_c) / perp_b
                ax.plot(perp_x_values, perp_y_values, 'g-', linewidth=2, label='Perpendicular Line')
            else:  # Vertical line
                ax.axvline(x=-perp_c / perp_a, color='g', linewidth=2, label='Perpendicular Line')
            
            # Mark the midpoint
            # ax.plot(midpoint_x, midpoint_y, 'gX', markersize=8, label='Midpoint')
            
            # Plot intersection points
            inter_x, inter_y = zip(*intersections)
            ax.plot(inter_x, inter_y, 'y*', markersize=10, label='Intersections')
            
            if len(intersections) > 2:
                print("The perpendicular line intersects the polygon at more than two points, confirming non-monotonicity.")
            else:
                print("Warning: The perpendicular line does not intersect the polygon at more than two points. "
                      "This may not adequately demonstrate non-monotonicity.")
        else:
            print("Warning: No intersections found with the perpendicular line. "
                  "This may indicate an issue with the violation point selection or intersection detection.")
    
    # Set plot limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    # Set title based on monotonicity
    if is_monotone:
        ax.set_title(f'Polygon is Monotone with respect to {a}x + {b}y + {c} = 0')
    else:
        ax.set_title(f'Polygon is NOT Monotone with respect to {a}x + {b}y + {c} = 0')
    
    plt.tight_layout()
    return fig

def check_monotonicity(vertices, line_coefficients, name="Polygon"):
    """
    Main function to check monotonicity of a polygon with respect to a line
    
    Args:
        vertices: List of (x, y) coordinates defining the polygon
        line_coefficients: Tuple (a, b, c) for the line equation ax + by + c = 0
        name: Name of the test case for display purposes
        
    Returns:
        is_monotone: Boolean indicating if the polygon is monotone
        fig: Matplotlib figure
    """
    # Extract coordinates and coefficients
    X = [v[0] for v in vertices]
    Y = [v[1] for v in vertices]
    a, b, c = line_coefficients
    
    # Check monotonicity
    monotone, violation_point = is_monotone(X, Y, a, b, c)
    
    # Always create the plot
    fig = plot_polygon_with_line(X, Y, a, b, c, violation_point, monotone)
    # fig.suptitle(f"Test Case: {name}", fontsize=16)
    
    if monotone:
        print(f"True - {name} is monotone with respect to the given line.")
    else:
        print(f"False - {name} is not monotone with respect to the given line.")
    
    return monotone, fig

def get_user_input():
    """
    Get polygon vertices and line coefficients from user input
    """
    print("Enter polygon vertices (x,y), one per line. Enter 'done' when finished:")
    vertices = []
    while True:
        vertex_input = input()
        if vertex_input.lower() == 'done':
            break
        try:
            x, y = map(float, vertex_input.replace('(', '').replace(')', '').split(','))
            vertices.append((x, y))
        except ValueError:
            print("Invalid input. Please enter coordinates as 'x,y'")
    
    print("\nEnter line coefficients (a,b,c) for the line equation ax + by + c = 0:")
    line_input = input()
    try:
        a, b, c = map(float, line_input.replace('(', '').replace(')', '').split(','))
        line_coefficients = (a, b, c)
    except ValueError:
        print("Invalid input. Using default values (1,1,0)")
        line_coefficients = (1, 1, 0)
    
    return vertices, line_coefficients

def main():
    # Define test examples directly in the main function
    test_examples = [
        # {
        #     "name": "Square (monotone w.r.t. x-axis)",
        #     "vertices": [(0, 0), (2, 0), (2, 2), (0, 2)],
        #     "line": (0, 1, 0)  # y = 0 (x-axis)
        # },
        # {
        #     "name": "Triangle (monotone w.r.t. x-axis)",
        #     "vertices": [(0, 0), (4, 0), (2, 3)],
        #     "line": (0, 1, 0)  # x = 0 (y-axis)
        # },
        {
            "name": "Hexagon (monotone w.r.t. diagonal line)",
            "vertices": [(0, 0), (2, 0), (3, 2), (2, 4), (0, 4), (-1, 2)],
            "line": (1, 1, 0)  # x + y = 0 (diagonal line)
        },
        {
            "name": "U-shape (not monotone w.r.t. y-axis)",
            "vertices": [(0, 0), (0, 3), (2, 4), (2, 1), (4, 0.5), (4, 2), (6, 3), (6, 0)],
            "line": (0, 1, 0)  # x = 0 (y-axis)
        },
        {
            "name": "U-shape (not monotone w.r.t. y-axis)",
            "vertices": [(0, 0), (0, 3), (2, 4), (2, 1), (4, 0.5), (4, 2), (6, 3), (6, 0)],
            "line": (1, 0, 0)  # x = 0 (y-axis)
        },
        # {
        #     "name": "Concave polygon (not monotone w.r.t. y-axis)",
        #     "vertices": [(0, 2), (2, 0), (4, 2), (2, 4), (1, 3), (2, 2.5)],
        #     "line": (0, 1, 0)  # x = 0 (y-axis)
        # },
    {
        "name": "Highly concave polygon - Line 1",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (1, 0, 0)  # Vertical line (x-axis aligned)
    },
    {
        "name": "Highly concave polygon - Line 2",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (0, 1, 0)  # Horizontal line (y-axis aligned)
    },
    {
        "name": "Highly concave polygon - Line 3",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (1, -1, 0)  # Diagonal line (45-degree angle)
    },
    {
        "name": "Highly concave polygon - Line 4",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (2, -1, 0)  # Steeper diagonal line
    },
    {
        "name": "Highly concave polygon - Line 5",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (-1, 2, 0)  # Reverse slope diagonal line
    },
    {
        "name": "Highly concave polygon - Line 6",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (1, 1, 0)  # 45-degree diagonal line
    },
    {
        "name": "Highly concave polygon - Line 7",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (-1, -1, 0)  # Negative slope diagonal line
    },
    {
        "name": "Highly concave polygon - Line 8",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (3, -2, 0)  # Arbitrary slant line
    },
    {
        "name": "Highly concave polygon - Line 9",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (2, 1, 0)  # Mildly slanted line
    },
    {
        "name": "Highly concave polygon - Line 10",
        "vertices": [
            (0, 2), (1, 0), (2, 1), (3, -0.5), (4, 2), 
            (3, 4), (2, 3.5), (1, 4), (2, 2.5)
        ],
        "line": (-3, 1, 0)  # Negative diagonal line
    },

        {
            "name": "Star-like shape (not monotone w.r.t. x-axis)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (1, 0, 0)  # x = 0 (y-axis)
        },
        {
            "name": "Star-like shape (not monotone w.r.t. x-axis)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (0, 1, 0)  # y = 0 (x-axis)
        },
        {
        "name": "Star-like shape (line y=0)",
        "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
        "line": (-1, 1, 0)  # y = x
        },
        {
            "name": "Star-like shape (line y=x)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (1, 1, 0)  # y = -x
        },
        # {
        #     "name": "Star-like shape (line y=-x)",
        #     "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 2)],
        #     "line": (1, 1, 0)  # y = -x
        # },
        {
            "name": "Star-like shape (line x=2)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (1, -1.5, -2)  # x = 2
        },
        {
            "name": "Star-like shape (line x=-1)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (-1.5, 1, 1)  # x = -1
        },
        {
            "name": "Star-like shape (line y=2)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (2, 1, -2)  # y = 2
        },
        {
            "name": "Star-like shape (line y=-2)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (1, 2, 2)  # y = -2
        },
        {
            "name": "Star-like shape (line y=2x+1)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (-2, 1, -1)  # y = 2x + 1
        },
        {
            "name": "Star-like shape (line y=-0.5x+3)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (0.5, 1, -3)  # y = -0.5x + 3
        },
        {
            "name": "Star-like shape (line y=3x-4)",
            "vertices": [(0, 0), (2, 5), (4, 0), (3, 2), (1, 1)],
            "line": (-3, 1, 4)  # y = 3x - 4
        }
    ]
    
    # print("Choose an option:")
    # print("1. Enter your own polygon and line")
    # print("2. Test with example polygons")
    
    # choice = input().strip()
    
    # if choice == "1":
    #     vertices, line_coefficients = get_user_input()
    #     if len(vertices) < 3:
    #         print("Error: A polygon must have at least 3 vertices.")
    #         return
        
    #     monotone, fig = check_monotonicity(vertices, line_coefficients)
    #     plt.figure(fig.number)
    #     plt.show()
    # elif choice == "2":
    for example in test_examples:
        print(f"\nTesting: {example['name']}")
        vertices = example["vertices"]
        line = example["line"]
        
        monotone, fig = check_monotonicity(vertices, line)
        plt.figure(fig.number)
        plt.show()
    # else:
    #     print("Invalid choice. Running test examples by default.")
    #     # If invalid choice, run the test examples anyway
    #     for example in test_examples:
    #         print(f"\nTesting: {example['name']}")
    #         vertices = example["vertices"]
    #         line = example["line"]
            
    #         monotone, fig = check_monotonicity(vertices, line, example['name'])
    #         plt.figure(fig.number)
    #         plt.show()

if __name__ == "__main__":
    main()