import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

# Point class to represent 2D points
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __lt__(self, other):
        if self.x == other.x:
            return self.y < other.y
        return self.x < other.x
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __hash__(self):  # Added to make Point hashable
        return hash((round(self.x / 1e-9), round(self.y / 1e-9)))
    
    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

# Line segment class to represent edges of the polygons
class Segment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        if p2 < p1:
            self.p1, self.p2 = p2, p1

# Function to compute the orientation of three points (p, q, r)
def orientation(p, q, r):
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else -1

# Function to check if two segments intersect
def get_intersection(seg1, seg2):
    p1, q1 = seg1.p1, seg1.p2
    p2, q2 = seg2.p1, seg2.p2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        denom = (p1.x - q1.x) * (p2.y - q2.y) - (p1.y - q1.y) * (p2.x - q2.x)
        if abs(denom) < 1e-9:
            return None
        
        t = ((p1.x - p2.x) * (p2.y - q2.y) - (p1.y - p2.y) * (p2.x - q2.x)) / denom
        u = -((p1.x - q1.x) * (p1.y - p2.y) - (p1.y - q1.y) * (p1.x - p2.x)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = p1.x + t * (q1.x - p1.x)
            y = p1.y + t * (q1.y - p1.y)
            return Point(x, y)
    return None

# Function to check if a point is inside a convex polygon
def is_point_inside_polygon(point, polygon):
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        if orientation(p1, p2, point) != -1:  # Not strictly to the left (counterclockwise)
            return False
    return True

# Function to compute the convex hull of a set of points
def convex_hull(points):
    if len(points) < 3:
        return points

    points = sorted(points)  # Sort by x, then y
    start = points[0]
    remaining = points[1:]

    def polar_angle(p):
        return math.atan2(p.y - start.y, p.x - start.x)

    remaining.sort(key=polar_angle)

    hull = [start]
    for p in remaining:
        while len(hull) > 1 and orientation(hull[-2], hull[-1], p) != -1:
            hull.pop()
        hull.append(p)
    return hull

# Function to split a convex polygon into upper and lower chains
def split_into_chains(polygon):
    n = len(polygon)
    # Find leftmost and rightmost points
    leftmost_idx = min(range(n), key=lambda i: (polygon[i].x, polygon[i].y))
    rightmost_idx = max(range(n), key=lambda i: (polygon[i].x, polygon[i].y))

    upper_chain = []
    lower_chain = []

    # Traverse from leftmost to rightmost (upper chain)
    idx = leftmost_idx
    while idx != rightmost_idx:
        upper_chain.append((polygon[idx], idx))
        idx = (idx + 1) % n

    # Include the rightmost point
    upper_chain.append((polygon[rightmost_idx], rightmost_idx))

    # Traverse from rightmost to leftmost (lower chain)
    idx = rightmost_idx
    while idx != leftmost_idx:
        lower_chain.append((polygon[idx], idx))
        idx = (idx + 1) % n

    # Include the leftmost point
    lower_chain.append((polygon[leftmost_idx], leftmost_idx))

    return upper_chain, lower_chain

# Function to compute the intersection of two convex polygons using sweep line
def intersect_polygons_with_events(P, Q):
    n, m = len(P), len(Q)
    if n < 3 or m < 3:
        return None, []

    # Split polygons into upper and lower chains
    P_upper, P_lower = split_into_chains(P)
    Q_upper, Q_lower = split_into_chains(Q)

    # Initialize sweep line status (4-element list: P upper, P lower, Q upper, Q lower)
    sweep_line_status = [None, None, None, None]  # [P_upper, P_lower, Q_upper, Q_lower]

    # Initialize event queue with all vertices
    event_queue = []
    intersection_points = []
    event_log = []
    intersection_count = 0

    # Add all vertices of P and Q to the event queue
    for i in range(n):
        event_queue.append((P[i].x, 'P', i))
    for i in range(m):
        event_queue.append((Q[i].x, 'Q', i))

    # Sort event queue by x-coordinate
    event_queue.sort()

    # Initialize indices for upper and lower chains
    P_upper_idx = 0
    P_lower_idx = 0
    Q_upper_idx = 0
    Q_lower_idx = 0

    # Current sweep line position
    current_x = float('-inf')

    # Process events
    while event_queue:
        # Get the next event (leftmost x-coordinate)
        x, event_type, idx = event_queue.pop(0)

        # Ensure the sweep line moves left to right
        if x < current_x:
            continue  # Skip events that would move the sweep line backward
        current_x = x

        # Log the current state
        event_log.append((x, [s[0] if s else None for s in sweep_line_status[:2]],
                         [s[0] if s else None for s in sweep_line_status[2:]],
                         intersection_points.copy()))

        # Process the event (vertex event)
        if event_type == 'P':
            # Check if this vertex is part of the upper or lower chain of P
            # Update P upper
            if P_upper_idx < len(P_upper) and P_upper[P_upper_idx][1] == idx:
                if P_upper_idx < len(P_upper) - 1:
                    seg = Segment(P_upper[P_upper_idx][0], P_upper[P_upper_idx + 1][0])
                    sweep_line_status[0] = (seg, P_upper_idx)
                    P_upper_idx += 1
                else:
                    sweep_line_status[0] = None
            # Update P lower
            if P_lower_idx < len(P_lower) and P_lower[P_lower_idx][1] == idx:
                if P_lower_idx < len(P_lower) - 1:
                    seg = Segment(P_lower[P_lower_idx][0], P_lower[P_lower_idx + 1][0])
                    sweep_line_status[1] = (seg, P_lower_idx)
                    P_lower_idx += 1
                else:
                    sweep_line_status[1] = None
        elif event_type == 'Q':
            # Update Q upper
            if Q_upper_idx < len(Q_upper) and Q_upper[Q_upper_idx][1] == idx:
                if Q_upper_idx < len(Q_upper) - 1:
                    seg = Segment(Q_upper[Q_upper_idx][0], Q_upper[Q_upper_idx + 1][0])
                    sweep_line_status[2] = (seg, Q_upper_idx)
                    Q_upper_idx += 1
                else:
                    sweep_line_status[2] = None
            # Update Q lower
            if Q_lower_idx < len(Q_lower) and Q_lower[Q_lower_idx][1] == idx:
                if Q_lower_idx < len(Q_lower) - 1:
                    seg = Segment(Q_lower[Q_lower_idx][0], Q_lower[Q_lower_idx + 1][0])
                    sweep_line_status[3] = (seg, Q_lower_idx)
                    Q_lower_idx += 1
                else:
                    sweep_line_status[3] = None
        elif event_type.startswith('I'):
            # Intersection event, already added to intersection_points
            pass

        # Collect active segments
        active_segments_P = [s[0] for s in sweep_line_status[:2] if s is not None]
        active_segments_Q = [s[0] for s in sweep_line_status[2:] if s is not None]

        # Check for intersections between all active segments
        new_intersections = []
        for seg_P in active_segments_P:
            for seg_Q in active_segments_Q:
                inter = get_intersection(seg_P, seg_Q)
                if inter and inter not in intersection_points and inter not in new_intersections:
                    new_intersections.append(inter)

        # Add new intersection points to the intersection_points list and event queue
        for inter in new_intersections:
            intersection_points.append(inter)
            if inter.x >= current_x:
                event_queue.append((inter.x, f'I{intersection_count}', intersection_count))
                intersection_count += 1

        # Sort event queue by x-coordinate
        event_queue.sort()

        # Ensure the event queue has at most 8 events (though this is not strictly enforced here)
        # In practice, the number of events is bounded by the number of vertices plus intersection points

        event_log.append((x, [s[0] if s else None for s in sweep_line_status[:2]],
                         [s[0] if s else None for s in sweep_line_status[2:]],
                         intersection_points.copy()))

    # Add vertices of P that are inside Q
    for p in P:
        if is_point_inside_polygon(p, Q) and p not in intersection_points:
            intersection_points.append(p)

    # Add vertices of Q that are inside P
    for q in Q:
        if is_point_inside_polygon(q, P) and q not in intersection_points:
            intersection_points.append(q)

    # Remove duplicates
    intersection_points = list(dict.fromkeys(intersection_points))

    # If no intersection points, return empty result
    if not intersection_points:
        return [], event_log

    # Compute the convex hull of the intersection points
    intersection_points = convex_hull(intersection_points)

    # Ensure the points are in counterclockwise order
    if intersection_points:
        center_x = sum(p.x for p in intersection_points) / len(intersection_points)
        center_y = sum(p.y for p in intersection_points) / len(intersection_points)

        def angle(p):
            return math.atan2(p.y - center_y, p.x - center_x)

        intersection_points.sort(key=angle)

    return intersection_points, event_log

# Function to animate the sweep line
def animate_sweep_line(P, Q, event_log, title):
    # Create a figure with two subplots: one for the sweep line (left), one for the event queue (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(title, fontsize=14)

    # Plot the polygons on ax1
    P_x = [p.x for p in P] + [P[0].x]
    P_y = [p.y for p in P] + [P[0].y]
    Q_x = [q.x for q in Q] + [Q[0].x]
    Q_y = [q.y for q in Q] + [Q[0].y]
    ax1.plot(P_x, P_y, 'b-', label='Polygon P (C1)')
    ax1.plot(Q_x, Q_y, 'r-', label='Polygon Q (C2)')

    # Highlight vertices of P inside Q and Q inside P
    P_inside_Q = [p for p in P if is_point_inside_polygon(p, Q)]
    Q_inside_P = [q for q in Q if is_point_inside_polygon(q, P)]
    if P_inside_Q:
        ax1.plot([p.x for p in P_inside_Q], [p.y for p in P_inside_Q], 'b*', label='P vertices inside Q')
    if Q_inside_P:
        ax1.plot([q.x for q in Q_inside_P], [q.y for q in Q_inside_P], 'r*', label='Q vertices inside P')

    # Set plot limits for ax1 (main plot)
    all_x = P_x + Q_x
    all_y = P_y + Q_y
    x_min = min(all_x) - 1
    x_max = max(all_x) + 1
    y_min = min(all_y) - 1
    y_max = max(all_y) + 1
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend()
    ax1.set_title('Sweep Line Algorithm')
    ax1.grid(True)

    # Initialize sweep line and intersection points on ax1
    sweep_line, = ax1.plot([], [], 'g--', label='Sweep Line')
    inter_points, = ax1.plot([], [], 'ko', label='Intersection Points')

    # Since we can have multiple active segments, we'll create lists to hold the plot objects
    active_P_plots = []
    active_Q_plots = []
    max_segments = 2  # At most 2 segments per polygon (upper and lower)
    for _ in range(max_segments):
        p_plot, = ax1.plot([], [], 'b-', linewidth=2)
        q_plot, = ax1.plot([], [], 'r-', linewidth=2)
        active_P_plots.append(p_plot)
        active_Q_plots.append(q_plot)

    # Initialize the fill for the intersection polygon
    intersection_fill = None

    # Initialize event queue visualization on ax2
    ax2.set_title('Event Queue (Vertices and Intersections)')
    ax2.set_xlim(0, 1)
    ax2.set_yticks([])
    ax2.set_xticks([])

    event_bars = []
    event_texts = []

    def init():
        nonlocal intersection_fill
        sweep_line.set_data([], [])
        for p_plot in active_P_plots:
            p_plot.set_data([], [])
        for q_plot in active_Q_plots:
            q_plot.set_data([], [])
        inter_points.set_data([], [])
        # Clear event queue visualization
        for bar in event_bars:
            bar.remove()
        for text in event_texts:
            text.remove()
        event_bars.clear()
        event_texts.clear()
        # Remove previous fill if it exists
        if intersection_fill is not None:
            for artist in intersection_fill:
                artist.remove()
            intersection_fill = None
        return [sweep_line] + active_P_plots + active_Q_plots + [inter_points] + event_bars + event_texts

    def update(frame):
        nonlocal intersection_fill
        x, segs_P, segs_Q, inters = event_log[frame]
        
        # Update sweep line
        sweep_line.set_data([x, x], [y_min, y_max])
        
        # Update active segments for P
        for i, p_plot in enumerate(active_P_plots):
            if i < len(segs_P) and segs_P[i] is not None:
                seg = segs_P[i]
                p_plot.set_data([seg.p1.x, seg.p2.x], [seg.p1.y, seg.p2.y])
            else:
                p_plot.set_data([], [])
        
        # Update active segments for Q
        for i, q_plot in enumerate(active_Q_plots):
            if i < len(segs_Q) and segs_Q[i] is not None:
                seg = segs_Q[i]
                q_plot.set_data([seg.p1.x, seg.p2.x], [seg.p1.y, seg.p2.y])
            else:
                q_plot.set_data([], [])
        
        # Update intersection points
        if inters:
            inter_x = [p.x for p in inters]
            inter_y = [p.y for p in inters]
            inter_points.set_data(inter_x, inter_y)
        else:
            inter_points.set_data([], [])

        # Update event queue visualization on ax2
        for bar in event_bars:
            bar.remove()
        for text in event_texts:
            text.remove()
        event_bars.clear()
        event_texts.clear()

        # Reconstruct the event queue for visualization based on the current frame
        event_x_coords = []
        event_labels = []
        # Add vertices that are still in the event queue (future events)
        if frame < len(event_log) - 1:
            # Collect all future vertex events
            for i in range(len(P)):
                if P[i].x >= x:
                    found = False
                    for j, existing_x in enumerate(event_x_coords):
                        if abs(existing_x - P[i].x) < 1e-9:
                            event_labels[j] += f", P{i} ({P[i].x:.2f})"
                            found = True
                            break
                    if not found:
                        event_x_coords.append(P[i].x)
                        event_labels.append(f"P{i} ({P[i].x:.2f})")
            for i in range(len(Q)):
                if Q[i].x >= x:
                    found = False
                    for j, existing_x in enumerate(event_x_coords):
                        if abs(existing_x - Q[i].x) < 1e-9:
                            event_labels[j] += f", Q{i} ({Q[i].x:.2f})"
                            found = True
                            break
                    if not found:
                        event_x_coords.append(Q[i].x)
                        event_labels.append(f"Q{i} ({Q[i].x:.2f})")
            # Add intersection points that were added in this frame
            if frame > 0:
                prev_inters = event_log[frame - 1][3]
                curr_inters = inters
                new_inters = [p for p in curr_inters if p not in prev_inters]
                for i, inter in enumerate(new_inters):
                    if inter.x >= x:
                        found = False
                        for j, existing_x in enumerate(event_x_coords):
                            if abs(existing_x - inter.x) < 1e-9:
                                event_labels[j] += f", I{i} ({inter.x:.2f})"
                                found = True
                                break
                        if not found:
                            event_x_coords.append(inter.x)
                            event_labels.append(f"I{i} ({inter.x:.2f})")

        # Filter events to show only those that haven't been processed yet
        remaining_events = [(x_coord, label) for x_coord, label in zip(event_x_coords, event_labels) if x_coord >= x]
        y_positions = range(len(remaining_events), 0, -1)

        for i, (x_coord, label) in enumerate(remaining_events):
            bar = ax2.barh(y_positions[i], 1, align='center', color='skyblue')[0]
            text = ax2.text(0.5, y_positions[i], label, ha='center', va='center', fontsize=8)
            event_bars.append(bar)
            event_texts.append(text)

        ax2.set_ylim(0, len(remaining_events) + 1)

        # Color the interior of the intersection polygon in the last frame
        if frame == len(event_log) - 1 and inters and len(inters) >= 3:
            if intersection_fill is not None:
                for artist in intersection_fill:
                    artist.remove()
            inter_x = [p.x for p in inters]
            inter_y = [p.y for p in inters]
            intersection_fill = ax1.fill(inter_x, inter_y, 'yellow', alpha=0.5, label='Intersection Polygon')

        return [sweep_line] + active_P_plots + active_Q_plots + [inter_points] + event_bars + event_texts
# Function to display the convex hull of the intersection polygon separately
def display_intersection_polygon(P, Q, intersection_polygon, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title + " - Intersection Polygon", fontsize=14)

    P_x = [p.x for p in P] + [P[0].x]
    P_y = [p.y for p in P] + [P[0].y]
    Q_x = [q.x for q in Q] + [Q[0].x]
    Q_y = [q.y for q in Q] + [Q[0].y]
    ax.plot(P_x, P_y, 'b-', label='Polygon P (C1)')
    ax.plot(Q_x, Q_y, 'r-', label='Polygon Q (C2)')

    all_x = P_x + Q_x
    all_y = P_y + Q_y
    if intersection_polygon:
        inter_x = [p.x for p in intersection_polygon]
        inter_y = [p.y for p in intersection_polygon]
        all_x.extend(inter_x)
        all_y.extend(inter_y)
    x_min = min(all_x) - 1
    x_max = max(all_x) + 1
    y_min = min(all_y) - 1
    y_max = max(all_y) + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('Intersection Polygon (Convex Hull)')
    ax.grid(True)

    if intersection_polygon and len(intersection_polygon) >= 3:
        inter_x = [p.x for p in intersection_polygon]
        inter_y = [p.y for p in intersection_polygon]
        ax.fill(inter_x, inter_y, 'yellow', alpha=0.5, label='Intersection Polygon (Convex Hull)')
        ax.plot(inter_x, inter_y, 'k-', linewidth=1)
        ax.plot(inter_x, inter_y, 'ko', markersize=5)

    ax.legend()
    plt.tight_layout()
    return fig

# Test cases
def run_test_cases():
    num_points = 15
    scenarios = [
        {
            "description": "One Polygon Inside Another",
            "center_P": (0, 0), "radius_P": 5,
            "center_Q": (0, 0), "radius_Q": 2,
            "seed": 12}
        # },
        # {
        #     "description": "Intersecting Polygons (Partial Overlap)",
        #     "center_P": (0, 0), "radius_P": 3,
        #     "center_Q": (4, 0), "radius_Q": 3,
        #     "seed": 13
        # },
        # {
        #     "description": "One Polygon Inside Another",
        #     "center_P": (4, 3), "radius_P": 4,
        #     "center_Q": (0, 0), "radius_Q": 10,
        #     "seed": 12
        # },
        # {
        #     "description": "Non-Intersecting Polygons (Disjoint)",
        #     "center_P": (-5, 0), "radius_P": 2,
        #     "center_Q": (5, 0), "radius_Q": 2,
        #     "seed": 11
        # }
    ]

    for test_idx, scenario in enumerate(scenarios):
        print(f"\nRunning {scenario['description']}")

        np.random.seed(scenario['seed'])

        points_P = []
        center_P_x, center_P_y = scenario['center_P']
        radius_P = scenario['radius_P']
        for _ in range(num_points):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, radius_P)
            x = center_P_x + distance * np.cos(angle)
            y = center_P_y + distance * np.sin(angle)
            points_P.append(Point(x, y))

        points_Q = []
        center_Q_x, center_Q_y = scenario['center_Q']
        radius_Q = scenario['radius_Q']
        for _ in range(num_points):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, radius_Q)
            x = center_Q_x + distance * np.cos(angle)
            y = center_Q_y + distance * np.sin(angle)
            points_Q.append(Point(x, y))

        P = convex_hull(points_P)
        Q = convex_hull(points_Q)

        if len(P) < 3 or len(Q) < 3:
            print("One of the polygons has fewer than 3 vertices. Skipping this test case.")
            continue

        result, event_log = intersect_polygons_with_events(P, Q)

        if not result:
            print("No intersection")
        else:
            print("Intersection points (counterclockwise order, convex hull):")
            for p in result:
                print(p)

        title = f"{scenario['description']}"
        fig, ani = animate_sweep_line(P, Q, event_log, title)
        plt.show(block=True)
        plt.close(fig)

        inter_fig = display_intersection_polygon(P, Q, result, title)
        plt.show(block=True)
        plt.close(inter_fig)

if __name__ == "__main__":
    run_test_cases()