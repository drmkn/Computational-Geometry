'''*Problem Statement:Given n points in a 2D plane, perform Delaunay Triangulation using incremental insertion. The user specifies a step i (1≤i≤n), and the program displays the edge flip process occurring at that step. If no edge flip occurs at step i, display "No edge flip".    **Input*: 1. An integer n representing the number of points.
2. n points given as (𝑥𝑖,𝑦𝑖) coordinates, where:
    - Each 𝑥𝑖,𝑦𝑖 is a real number.
    - Points are given in arbitrary order.
3. An integer i representing the step at which the user wants to    inspect the edge flip process.                                                                           *Output*: 1. If an edge flip occurs at step i:
                   - Show the affected edges before and after the flip.
                   - Display the updated triangulation after the flip.
                   2. If no edge flip occurs at step i:
                    - Print "No edge flip".
                   3. The final Delaunay Triangulation after all points have been inserted.'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from matplotlib.widgets import Button

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Triangle:
    def __init__(self, a, b, c):
        self.vertices = [a, b, c]
        self.neighbors = [None, None, None]
        self.circumcenter = None
        self.circumradius = None

    def compute_circumcircle(self):
        """
        Compute the circumcenter and circumradius of the triangle
        """
        ax, ay = self.vertices[0].x, self.vertices[0].y
        bx, by = self.vertices[1].x, self.vertices[1].y
        cx, cy = self.vertices[2].x, self.vertices[2].y

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        
        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

        self.circumcenter = Point(ux, uy)
        self.circumradius = math.sqrt((ux - ax)**2 + (uy - ay)**2)

class DelaunayTriangulation:
    def __init__(self, plot_range=2.0):
        self.triangles = []
        self.super_triangle_vertices = None
        self.plot_range = plot_range
        self.init_super_triangle()
        self.intermediate_states = []
        self.is_paused = False
        self.edge_flip_occurred = False
        self.processed_points = set()

    def init_super_triangle(self):
        # Expanded super triangle to cover entire plot
        p1 = Point(0, self.plot_range * 4)
        p2 = Point(-self.plot_range * 4, -self.plot_range * 4)
        p3 = Point(self.plot_range * 4, -self.plot_range * 4)
        
        # Store super triangle vertices separately
        self.super_triangle_vertices = {p1, p2, p3}
        
        super_triangle = Triangle(p1, p2, p3)
        super_triangle.compute_circumcircle()
        self.triangles.append(super_triangle)

    def incircle_test(self, a, b, c, d):
        """
        Implement the incircle test using determinant method
        Returns True if point d is inside the circumcircle of triangle abc
        """
        # Extract coordinates
        ax, ay = a.x, a.y
        bx, by = b.x, b.y
        cx, cy = c.x, c.y
        dx, dy = d.x, d.y

        # Compute determinant
        matrix = np.array([
            [ax, ay, ax**2 + ay**2, 1],
            [bx, by, bx**2 + by**2, 1],
            [cx, cy, cx**2 + cy**2, 1],
            [dx, dy, dx**2 + dy**2, 1]
        ])
        
        return np.linalg.det(matrix) > 0

    def is_edge_legal(self, triangle1, triangle2, shared_edge):
        """
        Check if the shared edge between two triangles is legal
        """
        # Find the vertices of the two triangles
        vertices1 = set(triangle1.vertices)
        vertices2 = set(triangle2.vertices)
        
        # Find the vertices not on the shared edge
        fourth_vertex1 = list(vertices1 - set(shared_edge))[0]
        fourth_vertex2 = list(vertices2 - set(shared_edge))[0]
        
        # Check if the fourth vertices are inside each other's circumcircles
        if (not triangle1.circumcenter):
            triangle1.compute_circumcircle()
        if (not triangle2.circumcenter):
            triangle2.compute_circumcircle()
        
        # If the fourth vertex of either triangle is inside the circumcircle of the other triangle, 
        # then the edge needs to be flipped
        return not (
            self.incircle_test(
                triangle1.vertices[0], 
                triangle1.vertices[1], 
                triangle1.vertices[2], 
                fourth_vertex2
            ) or 
            self.incircle_test(
                triangle2.vertices[0], 
                triangle2.vertices[1], 
                triangle2.vertices[2], 
                fourth_vertex1
            )
        )

    def find_bad_triangles(self, point):
        """
        Find triangles whose circumcircle contains the new point
        """
        bad_triangles = []
        for triangle in self.triangles:
            # First compute circumcircle if not already done
            if not triangle.circumcenter:
                triangle.compute_circumcircle()
            
            if self.incircle_test(triangle.vertices[0], 
                                   triangle.vertices[1], 
                                   triangle.vertices[2], 
                                   point):
                bad_triangles.append(triangle)
        
        return bad_triangles

    def find_boundary_edges(self, bad_triangles):
        """
        Find the boundary of the hole created by bad triangles
        """
        polygon_edges = []
        for triangle in bad_triangles:
            for i in range(3):
                edge = (triangle.vertices[i], triangle.vertices[(i+1)%3])
                
                edge_count = sum(1 for t in bad_triangles 
                                 for j in range(3) 
                                 if (t.vertices[j], t.vertices[(j+1)%3]) == edge or 
                                    (t.vertices[(j+1)%3], t.vertices[j]) == edge)
                
                if edge_count == 1:
                    polygon_edges.append(edge)
        
        return polygon_edges

    def insert_point(self, point):
        """
        Insert a point into the triangulation
        """
        # Find bad triangles (those whose circumcircle contains the new point)
        bad_triangles = self.find_bad_triangles(point)
        
        # Find boundary of the hole
        polygon_edges = self.find_boundary_edges(bad_triangles)
        
        # Store current state for animation
        current_state = {
            'point': point,
            'bad_triangles': bad_triangles.copy(),
            'polygon_edges': polygon_edges.copy(),
            'triangles': self.triangles.copy(),
            'show_incircle': True,  # Flag to show incircle during test
            'edge_flip_occurred': False
        }
        self.intermediate_states.append(current_state)
        
        # Store state after incircle test (without incircle)
        current_state = {
            'point': point,
            'bad_triangles': bad_triangles.copy(),
            'polygon_edges': polygon_edges.copy(),
            'triangles': self.triangles.copy(),
            'show_incircle': False,  # Hide incircle after test
            'edge_flip_occurred': False
        }
        self.intermediate_states.append(current_state)
        
        # Remove bad triangles
        for triangle in bad_triangles:
            if triangle in self.triangles:
                self.triangles.remove(triangle)
        
        # Create new triangles connecting boundary to the new point
        new_triangles = []
        for edge in polygon_edges:
            new_triangle = Triangle(edge[0], edge[1], point)
            new_triangle.compute_circumcircle()
            new_triangles.append(new_triangle)
        
        self.triangles.extend(new_triangles)
        
        # Store final state after point insertion
        current_state = {
            'point': point,
            'bad_triangles': [],
            'polygon_edges': [],
            'triangles': self.triangles.copy(),
            'show_incircle': False,
            'edge_flip_occurred': False
        }
        self.intermediate_states.append(current_state)

    def legalize_triangulation(self):
        """
        Legalize the triangulation by checking and flipping illegal edges
        """
        # Track if any changes are made
        changes_made = True
        iteration = 0
        max_iterations = len(self.triangles) * 2  # Prevent infinite loop

        while changes_made and iteration < max_iterations:
            changes_made = False
            iteration += 1

            # Create a copy of triangles to iterate over
            current_triangles = self.triangles.copy()

            # Check each triangle against its neighbors
            for triangle in current_triangles:
                for i in range(3):
                    # Current edge
                    edge = (triangle.vertices[i], triangle.vertices[(i+1)%3])
                    
                    # Find adjacent triangles
                    adjacent_triangles = [
                        t for t in current_triangles 
                        if t != triangle and 
                        set(edge).issubset(set(t.vertices))
                    ]

                    # Check each adjacent triangle
                    for adj_triangle in adjacent_triangles:
                        if not self.is_edge_legal(triangle, adj_triangle, edge):
                            # Perform edge flip
                            self.edge_flip_occurred = True
                            changes_made = True

                            # Find the vertices not on the shared edge
                            vertices1 = set(triangle.vertices)
                            vertices2 = set(adj_triangle.vertices)
                            
                            fourth_vertex1 = list(vertices1 - set(edge))[0]
                            fourth_vertex2 = list(vertices2 - set(edge))[0]

                            # Remove old triangles
                            self.triangles.remove(triangle)
                            self.triangles.remove(adj_triangle)

                            # Create new triangles
                            new_triangle1 = Triangle(fourth_vertex1, fourth_vertex2, edge[0])
                            new_triangle2 = Triangle(fourth_vertex1, fourth_vertex2, edge[1])

                            # Compute new circumcircles
                            new_triangle1.compute_circumcircle()
                            new_triangle2.compute_circumcircle()

                            # Add new triangles
                            self.triangles.extend([new_triangle1, new_triangle2])

                            # Break out of adjacent triangle loop
                            break

            # Create a state for each iteration of legalization
            current_state = {
                'point': None,
                'bad_triangles': [],
                'polygon_edges': [],
                'triangles': self.triangles.copy(),
                'show_incircle': False,
                'edge_flip_occurred': self.edge_flip_occurred
            }
            self.intermediate_states.append(current_state)

    def create_animation_and_final_plot(self, points):
        """
        Create an animation of the Delaunay Triangulation process and a final plot
        """
        # Reset processed points
        self.processed_points = set()

        # Perform triangulation and collect intermediate states
        for point in points:
            self.insert_point(point)

        # Legalize the triangulation after all points are inserted
        self.legalize_triangulation()

        # Create two subplots
        fig, (ax_anim, ax_final) = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(bottom=0.2, wspace=0.3)  # Make room for buttons and add space between plots

        # Animation plot setup
        ax_anim.set_xlim(-4.1*self.plot_range, 4.1*self.plot_range)
        ax_anim.set_ylim(-4.1*self.plot_range, 4.1*self.plot_range)
        ax_anim.set_title('Delaunay Triangulation - Incremental Insertion')
        ax_anim.set_xlabel('X')
        ax_anim.set_ylabel('Y')
        ax_anim.grid(True, linestyle='--', alpha=0.7)

        # Final plot setup
        ax_final.set_xlim(-1.1*self.plot_range, 1.1*self.plot_range)
        ax_final.set_ylim(-1.1*self.plot_range, 1.1*self.plot_range)
        ax_final.set_title('Final Delaunay Triangulation')
        ax_final.set_xlabel('X')
        ax_final.set_ylabel('Y')
        ax_final.grid(True, linestyle='--', alpha=0.7)

        def update(frame):
            ax_anim.clear()
            ax_anim.set_xlim(-4.1*self.plot_range, 4.1*self.plot_range)
            ax_anim.set_ylim(-4.1*self.plot_range, 4.1*self.plot_range)
            
            # Get current state
            state = self.intermediate_states[frame]
            
            # Update title with iteration and edge flip information
            title_text = f'Delaunay Triangulation - Iteration {frame}'
            # edge_flip_text = 'Edge Flip: ' + ('Yes' if state.get('edge_flip_occurred', False) else 'No')
            ax_anim.set_title(f'{title_text}')
            ax_anim.grid(True, linestyle='--', alpha=0.7)

            # Draw all triangles
            for triangle in state['triangles']:
                # Triangle edges
                coords = [(v.x, v.y) for v in triangle.vertices]
                coords.append(coords[0])
                xs, ys = zip(*coords)
                ax_anim.plot(xs, ys, color='blue', linewidth=1, alpha=0.5)

            # Draw only the circumcircle of the current iteration's triangle
            if state.get('point'):
                for triangle in state['triangles']:
                    # Check if the triangle contains the current point
                    if state['point'] in triangle.vertices and triangle.circumcenter:
                        circle = plt.Circle(
                            (triangle.circumcenter.x, triangle.circumcenter.y), 
                            triangle.circumradius, 
                            color='black',  
                            fill=False, 
                            linestyle=':', 
                            alpha=0.5
                        )
                        ax_anim.add_artist(circle)

            # Update processed points
            if state.get('point'):
                self.processed_points.add(state['point'])

            # Draw bad triangles during incircle test if applicable
            if state.get('show_incircle', False):
                for triangle in state.get('bad_triangles', []):
                    coords = [(v.x, v.y) for v in triangle.vertices]
                    coords.append(coords[0])
                    xs, ys = zip(*coords)
                    ax_anim.plot(xs, ys, color='orange', linewidth=2, alpha=0.7)

            # Draw polygon edges if applicable
            for edge in state.get('polygon_edges', []):
                ax_anim.plot([edge[0].x, edge[1].x], 
                             [edge[0].y, edge[1].y], 
                             color='green', linewidth=3, alpha=0.7)
            
            # Draw current point if exists
            if state.get('point'):
                current_point = state['point']
                ax_anim.scatter([current_point.x], [current_point.y], 
                                color='yellow', s=200, zorder=11, 
                                edgecolor='black')

            # Draw points - processed in red, unprocessed in light grey
            processed_x = [p.x for p in self.processed_points]
            processed_y = [p.y for p in self.processed_points]
            unprocessed_x = [p.x for p in points if p not in self.processed_points]
            unprocessed_y = [p.y for p in points if p not in self.processed_points]
            
            ax_anim.scatter(processed_x, processed_y, color='red', s=100, zorder=10)
            ax_anim.scatter(unprocessed_x, unprocessed_y, color='lightgrey', s=100, zorder=10)

        # def pause(event):
        #     self.is_paused = not self.is_paused
        #     pause_button.label.set_text('Resume' if self.is_paused else 'Pause')
        #     fig.canvas.draw_idle()

        # # Add pause button
        # ax_pause = plt.axes([0.7, 0.05, 0.2, 0.075])
        # pause_button = Button(ax_pause, 'Pause')
        # pause_button.on_clicked(pause)

        # Create animation
        anim = animation.FuncAnimation(
            fig, update, 
            frames=len(self.intermediate_states), 
            interval=1000, 
            repeat=True,
            cache_frame_data=False
        )

        # Find final triangles that do NOT include super triangle vertices
        final_triangles = [
            triangle for triangle in self.triangles 
            if not any(v in self.super_triangle_vertices for v in triangle.vertices)
        ]

        # Plot final triangulation
        # Plot triangles
        for triangle in final_triangles:
            # Compute circumcircle if not already done
            if not triangle.circumcenter:
                triangle.compute_circumcircle()
            
            # Triangle edges
            coords = [(v.x, v.y) for v in triangle.vertices]
            coords.append(coords[0])
            xs, ys = zip(*coords)
            ax_final.plot(xs, ys, color='blue', linewidth=1, alpha=0.5)
            
            # Draw circumcircle
            circle = plt.Circle(
                (triangle.circumcenter.x, triangle.circumcenter.y), 
                triangle.circumradius, 
                color='black',  
                fill=False, 
                linestyle=':', 
                alpha=0.5
            )
            ax_final.add_artist(circle)

        # Extract all unique points (original input points)
        all_points = points
        
        # Plot points with red color
        ax_final.scatter(
            [p.x for p in all_points], 
            [p.y for p in all_points], 
            color='red', 
            s=100, 
            zorder=11, 
            edgecolor='black'
        )

        plt.tight_layout()
        return anim

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate random points
    num_points = 5  # Increased number of points for a more interesting visualization
    points = [
        Point(
            random.uniform(-2, 2), 
            random.uniform(-2, 2)
        ) for _ in range(num_points)
    ]

    # Create Delaunay Triangulation
    dt = DelaunayTriangulation(plot_range=2.0)

    # Create and display animation with final plot
    anim = dt.create_animation_and_final_plot(points)
    plt.show()

if __name__ == "__main__":
    main()