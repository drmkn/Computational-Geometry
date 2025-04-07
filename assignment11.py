'''Generate n random points in 2D and find the smallest directional width. Show the direction. Show the corresponding rectangular 
strip. Mark the two points for which the width is smallest.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math

# Set random seed for reproducibility
np.random.seed(42)

# Generate random points in primal space
n = 30
points = np.random.rand(n, 2) * 10

# Create figure with two subplots - primal and dual spaces
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Plot points in primal space
ax1.scatter(points[:, 0], points[:, 1], color='blue', s=50, label='Original points')
ax1.set_title("Primal Space (Original Points)")
ax1.grid(True)
ax1.set_aspect('equal')

# Compute convex hull (only needed for visualization in primal space)
hull = ConvexHull(points)
for simplex in hull.simplices:
    ax1.plot(points[simplex, 0], points[simplex, 1], 'k-')

# Transform points to dual space representation
# We use a parametric approach: for slopes m, find the y-intercepts b
# For a point (a,b), the dual line is y = ax - b
# We sample different slope values
m_values = np.linspace(-5, 5, 1000)  # Range of slopes to consider

# For each point, compute the dual line's y-intercept for each slope value
dual_lines = []
for point in points:
    a, b = point
    # For each slope m, compute y-intercept c where: c = am - b
    c_values = a * m_values - b
    dual_lines.append((m_values, c_values))

# Plot dual lines
for m_vals, c_vals in dual_lines:
    ax2.plot(m_vals, c_vals, 'b-', alpha=0.2)

ax2.set_title("Dual Space (Lines)")
ax2.set_xlabel("m (slope)")
ax2.set_ylabel("c (y-intercept)")
ax2.grid(True)

# Compute upper and lower envelopes in dual space
upper_envelope = np.ones_like(m_values) * float('-inf')
lower_envelope = np.ones_like(m_values) * float('inf')
upper_point_idx = np.zeros_like(m_values, dtype=int)
lower_point_idx = np.zeros_like(m_values, dtype=int)

for i, (m_vals, c_vals) in enumerate(dual_lines):
    for j, (m, c) in enumerate(zip(m_vals, c_vals)):
        if c > upper_envelope[j]:
            upper_envelope[j] = c
            upper_point_idx[j] = i
        if c < lower_envelope[j]:
            lower_envelope[j] = c
            lower_point_idx[j] = i

# Plot the upper and lower envelopes
ax2.plot(m_values, upper_envelope, 'r-', linewidth=2, label='Upper envelope')
ax2.plot(m_values, lower_envelope, 'g-', linewidth=2, label='Lower envelope')
ax2.legend()

# Find the minimum vertical gap between envelopes
vertical_gaps = upper_envelope - lower_envelope
min_gap_idx = np.argmin(vertical_gaps)
min_gap = vertical_gaps[min_gap_idx]
min_slope = m_values[min_gap_idx]

# Convert minimum gap to the actual width in primal space
# For a slope m, the perpendicular direction has slope -1/m
# The width w = gap / sqrt(1 + m²)
if abs(min_slope) < 1e-10:  # Practically zero slope
    min_width = min_gap
    angle = np.pi/2  # 90 degrees
else:
    min_width = min_gap / np.sqrt(1 + min_slope**2)
    angle = np.arctan(-1/min_slope)
    if angle < 0:
        angle += np.pi  # Ensure angle is in [0, π)

# Mark the minimum gap in dual space
ax2.plot([min_slope, min_slope], [lower_envelope[min_gap_idx], upper_envelope[min_gap_idx]], 
         'k-', linewidth=3, label=f'Min gap: {min_gap:.4f} at m={min_slope:.4f}')
ax2.legend()

# Find the two points in primal space that define the width
p1_idx = lower_point_idx[min_gap_idx]
p2_idx = upper_point_idx[min_gap_idx]
p1 = points[p1_idx]
p2 = points[p2_idx]

# Calculate the direction vector in primal space
# The direction is perpendicular to the slope
if abs(min_slope) < 1e-10:  # Practically zero slope
    direction = np.array([0, 1])  # Vertical direction
else:
    # Direction perpendicular to line with slope min_slope
    direction = np.array([1, -1/min_slope])
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

# Mark these points in primal space
ax1.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='red', s=150, marker='*', 
           label='Width-defining points')

# Draw the direction vector from center of points
center = np.mean(points, axis=0)
ax1.arrow(center[0], center[1], direction[0], direction[1], 
          head_width=0.3, head_length=0.3, fc='green', ec='green', 
          label=f'Min width direction (angle = {angle:.4f} rad)')

# Calculate and plot the strip with minimum width
# Direction perpendicular to our width direction
perp_direction = np.array([-direction[1], direction[0]])

# Define the two parallel lines passing through p1 and p2
t = np.linspace(-15, 15, 100)
line1_x = p1[0] + t * perp_direction[0]
line1_y = p1[1] + t * perp_direction[1]
line2_x = p2[0] + t * perp_direction[0]
line2_y = p2[1] + t * perp_direction[1]

# Plot the strip
ax1.plot(line1_x, line1_y, 'g--', label='Minimum width strip')
ax1.plot(line2_x, line2_y, 'g--')
ax1.legend()

# Set reasonable axis limits
ax1_xmin, ax1_xmax = ax1.get_xlim()
ax1_ymin, ax1_ymax = ax1.get_ylim()
ax1.set_xlim(min(ax1_xmin, -1), max(ax1_xmax, 11))
ax1.set_ylim(min(ax1_ymin, -1), max(ax1_ymax, 11))

plt.tight_layout()

# Print the results
print(f"Minimum directional width: {min_width:.4f}")
print(f"Direction angle: {angle:.4f} radians ({angle * 180/np.pi:.2f} degrees)")
print(f"Points defining the width: {p1} and {p2}")
print(f"Slope in dual space where minimum occurs: {min_slope:.4f}")