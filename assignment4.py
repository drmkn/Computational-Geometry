import matplotlib.pyplot as plt
import numpy as np

def calculate_angle(p1, p2, p3):
    """
    Computes the interior angle at vertex p2 formed by the edges (p1, p2) and (p2, p3)
    using the orientation test.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])  # Vector from p2 to p1
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])  # Vector from p2 to p3

    # Compute the cross product (orientation test)
    cross_product = np.cross(v1, v2)
    
    # Compute the dot product for angle calculation
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid floating-point errors

    # Convert angle to degrees
    angle_deg = np.degrees(theta)

    # If cross_product is negative, the angle is reflex (>180 degrees)
    if cross_product >= 0:
        return 360 - angle_deg
    else:
        return angle_deg

def draw_polygon(vertices, title="Polygon"):
    """
    Draws a polygon and marks split, merge, and regular vertices with a proper legend.
    """
    x, y = zip(*vertices)
    x = list(x) + [x[0]]
    y = list(y) + [y[0]]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'k-', linewidth=2, label="Polygon")

    split_vertices = []
    merge_vertices = []

    for i in range(len(vertices)):
        prev_idx = (i - 1) % len(vertices)
        next_idx = (i + 1) % len(vertices)
        prev_x, prev_y = vertices[prev_idx]
        curr_x, curr_y = vertices[i]
        next_x, next_y = vertices[next_idx]

        interior_angle = calculate_angle(vertices[prev_idx], vertices[i], vertices[next_idx])
        print(f"Vertex {vertices[i]}: Interior Angle = {interior_angle:.2f}°")

        if curr_x < prev_x and curr_x < next_x and interior_angle > 180:
            split_vertices.append((curr_x, curr_y))
        elif curr_x > prev_x and curr_x > next_x and interior_angle > 180:
            merge_vertices.append((curr_x, curr_y))
        else:
            plt.scatter(curr_x, curr_y, color='green', s=50)
    
    # Plot split vertices if they exist
    if split_vertices:
        sx, sy = zip(*split_vertices)
        plt.scatter(sx, sy, color='blue', s=100, label="Split Vertex")
    
    # Plot merge vertices if they exist
    if merge_vertices:
        mx, my = zip(*merge_vertices)
        plt.scatter(mx, my, color='red', s=100, label="Merge Vertex")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to define test cases and visualize the polygons.
    """
    test_cases = [
        [(2, 1), (6, 2), (7, 5), (5, 7), (3, 6), (2, 4), (4, 3)],  
        [(1, 2), (4, 1), (6, 3), (4.5, 6), (5, 8), (2, 7), (1.5, 4),(3,2)],  
        [(2, 3), (5, 2), (4, 3), (6, 5), (3.5, 5.5), (4, 7), (2, 6), (3, 4)],
        [(3, 1), (7, 2), (9, 5), (6, 8), (4, 7), (2, 5), (3, 3), (5, 4)],  # Non-x-monotone polygon
        [(1, 3), (4, 1), (3.5, 4), (8, 5), (5, 6), (3, 8), (1, 6), (2, 4)]   # Another complex polygon
    ]
    
    for i, vertices in enumerate(test_cases):
        print(f"Test Case {i+1}: {vertices}")
        draw_polygon(vertices, title=f"Test Case {i+1}")

if __name__ == "__main__":
    main()