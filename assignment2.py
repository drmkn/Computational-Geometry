import numpy as np
import matplotlib.pyplot as plt

class ConvexPolygonQuery:
    def __init__(self, points):
        self.polygon = np.array(self._sort_counterclockwise(points))
        
    def _cross_product(self, p, q, r):
        """Calculate cross product (p, q) × (p, r)"""
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    
    def _sort_counterclockwise(self, points):
        """Sort points in counterclockwise order around their centroid"""
        centroid = np.mean(points, axis=0)
        
        # Sort points by angle from centroid
        return sorted(points, 
                     key=lambda p: np.arctan2(p[1] - centroid[1], 
                                            p[0] - centroid[0]))
    
    def contains_point(self, q):
        n = len(self.polygon)
        # Check if point is outside the first or last triangle
        if (self._cross_product(self.polygon[0], self.polygon[1], q) < 0 or 
            self._cross_product(self.polygon[0], self.polygon[-1], q) > 0):
            return False
        
        # Binary search for the triangle containing q
        left, right = 1, n - 1
        while left + 1 < right:
            mid = (left + right) // 2
            if self._cross_product(self.polygon[0], self.polygon[mid], q) >= 0:
                left = mid
            else:
                right = mid
                
        return self._cross_product(self.polygon[left], self.polygon[right], q) >= 0
    
    def _is_tangent(self, q, prev, curr, next, is_left):
        """Check if current vertex forms a tangent with query point"""
        if is_left:
                return (self._cross_product(curr, q, prev) > 0 and 
                        self._cross_product(curr, q, next) >= 0)
        else:
            return (self._cross_product(curr, q, prev) < 0 and 
            self._cross_product(curr, q, next) <= 0)
    
    def find_tangents(self, q):
        if self.contains_point(q):
            return None
        
        # Binary search for left tangent
        def find_single_tangent(is_left):
            lo, hi = 0, len(self.polygon)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                prev = self.polygon[mid - 1]
                curr = self.polygon[mid]
                next = self.polygon[(mid + 1) % len(self.polygon)]
                
                if self._is_tangent(q, prev, curr, next, is_left):
                    return curr
                
                if is_left:
                    if self._cross_product(q, curr, prev) < 0:
                        hi = mid
                    else:
                        lo = mid
                else:
                    if self._cross_product(q, curr, prev) > 0:
                        hi = mid
                    else:
                        lo = mid
            
            return self.polygon[lo]
        
        left_tangent = find_single_tangent(True)
        right_tangent = find_single_tangent(False)
        return left_tangent, right_tangent
    
    def visualize(self, query_point=None, tangents=None):
        """Visualize the polygon, query point, and tangents if provided"""
        plt.figure(figsize=(8, 8))
        
        # Plot polygon
        polygon_closed = np.vstack([self.polygon, self.polygon[0]])
        plt.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'bo-', 
                label="Convex Polygon")
        
        if query_point is not None:
            plt.plot(query_point[0], query_point[1], 'ro', markersize=8, 
                    label="Query Point")
            
            if tangents is not None:
                left_tangent, right_tangent = tangents
                if left_tangent is not None and right_tangent is not None:
                    # Plot tangent lines
                    plt.plot([query_point[0], left_tangent[0]], 
                           [query_point[1], left_tangent[1]], 
                           'g--', linewidth=2, label="Left Tangent")
                    plt.plot([query_point[0], right_tangent[0]], 
                           [query_point[1], right_tangent[1]], 
                           'm--', linewidth=2, label="Right Tangent")
                    
                    # Plot tangent points
                    plt.plot(left_tangent[0], left_tangent[1], 'go', 
                           markersize=8, label="Left Tangent Point")
                    plt.plot(right_tangent[0], right_tangent[1], 'mo', 
                           markersize=8, label="Right Tangent Point")
        
        plt.legend()
        plt.grid(True)
        plt.title("Convex Polygon Query Visualization")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.axis('equal')
        plt.show()

def main():
    polygon_points = np.array([
        [1, 1], [3, 0], [5, 1], [4, 4], [2, 3]
    ])
    
    polygon_query = ConvexPolygonQuery(polygon_points)
    
    query_points = [
        np.array([5, 2]),
        np.array([3, 2]), 
        np.array([6, 4]),
        np.array([0, 2]),
        np.array([5,1]),

    ]
    
    for q in query_points:
        print(f"\nTesting query point {q}")
        is_inside = polygon_query.contains_point(q)
        print(f"Point is inside polygon: {is_inside}")
        
        if not is_inside:
            tangents = polygon_query.find_tangents(q)
            print(f"Tangent points: {tangents}")
            polygon_query.visualize(q, tangents)
        else:
            polygon_query.visualize(q)
if __name__ == "__main__":
    main()
