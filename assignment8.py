import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import random

# Apply Seaborn theme
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Step 1: Get user input for n and k
def get_user_input():
    while True:
        try:
            n = int(input("Enter the number of lines (n): "))
            if n < 1:
                print("n must be at least 1. Please try again.")
                continue
            k = int(input(f"Enter the number of lines to stab (k, 1 <= k <= {n}): "))
            if k < 1 or k > n:
                print(f"k must be between 1 and {n}. Please try again.")
                continue
            return n, k
        except ValueError:
            print("Please enter valid integers.")

n, k = get_user_input()

# Step 2: Generate random lines
lines = [(random.uniform(-10, 10), random.uniform(-5, 5)) for _ in range(n)]
# lines += ([(1,1),(1,1)])

# Compute all intersection points (events)
events = sorted(set((c2 - c1) / (m1 - m2) for i, (m1, c1) in enumerate(lines) 
                     for j, (m2, c2) in enumerate(lines) if i < j and m1 != m2))
if not events:
    events = [-10, 10]  # Default range if no intersections
else:
    events = [min(events) - 1] + events + [max(events) + 1]

# Step 3: Setup the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(min(events) - 1, max(events) + 1)
ax.set_ylim(-50, 50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Shortest Vertical k-Stabber (k={k}) Animation')

x_range = np.linspace(min(events) - 1, max(events) + 1, 100)
for m, c in lines:
    ax.plot(x_range, m * x_range + c, color='cyan', alpha=0.6)  # Muted cyan for lines

sweep_line, = ax.plot([], [], color='blue', linestyle='dashed', label='Sweep Line')  # Blue
stabber, = ax.plot([], [], color='red', linewidth=2, label='Current k-Stabber')  # Red
event_dots, = ax.plot([], [], 'ro', markersize=6, label="Event Points")  # Red dots
ax.legend()
current_length_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top')
min_length_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, verticalalignment='top')

# Step 4: Animation function
min_length = float('inf')
min_stabber = None
all_stabbers = []

# Print the equations of the lines
print("\nGenerated Lines (y = mx + c):")
for i, (m, c) in enumerate(lines):
    print(f"Line {i+1}: y = {m:.3f}x + {c:.3f}")


def update(frame):
    global min_length, min_stabber
    x = events[frame]
    y_sorted = sorted(m * x + c for m, c in lines)
    if len(y_sorted) < k:
        return sweep_line, stabber, event_dots, current_length_text, min_length_text
    
    best_length = float('inf')
    y_low, y_high = None, None
    for i in range(len(y_sorted) - k + 1):
        length = y_sorted[i + k - 1] - y_sorted[i]
        if length < best_length:
            best_length = length
            y_low, y_high = y_sorted[i], y_sorted[i + k - 1]
    
    sweep_line.set_data([x, x], [ax.get_ylim()[0], ax.get_ylim()[1]])
    stabber.set_data([x, x], [y_low, y_high])
    event_dots.set_data(events[:frame+1], [0] * (frame + 1))  # Update event points
    current_length_text.set_text(f'Current Length: {best_length:.2f}')
    
    all_stabbers.append((x, y_low, y_high))  # Store for final plot
    
    if best_length < min_length:
        min_length = best_length
        min_stabber = (x, y_low, y_high)
    min_length_text.set_text(f'Min Length: {min_length:.2f}')
    
    return sweep_line, stabber, event_dots, current_length_text, min_length_text

ani = FuncAnimation(fig, update, frames=len(events), interval=500, blit=True)

# Step 5: Final optimal stabber plot with all considered stabbers
def show_final_plot():
    if min_stabber is None:
        print("No valid stabber found.")
        return
    fig_final, ax_final = plt.subplots(figsize=(10, 6))
    ax_final.set_xlim(min(events) - 1, max(events) + 1)
    ax_final.set_ylim(-10, 10)
    ax_final.set_xlabel('x')
    ax_final.set_ylabel('y')
    ax_final.set_title(f'Shortest Vertical k-Stabber (k={k}) - Final Result')
    
    for m, c in lines:
        ax_final.plot(x_range, m * x_range + c, color='cyan', alpha=0.6)  # Muted cyan for lines
    
    # Plot all candidate stabbers in light gray
    for x, y_low, y_high in all_stabbers:
        ax_final.plot([x, x], [y_low, y_high], color='gray', linestyle='dotted', alpha=0.5)  # Light gray
    
    # Plot the optimal stabber in magenta
    min_x, min_y_low, min_y_high = min_stabber
    ax_final.plot([min_x, min_x], [min_y_low, min_y_high], color='magenta', linewidth=3, label=f'Optimal {k}-Stabber')  # Magenta
    
    # Mark event points in black
    ax_final.scatter(events, [0] * len(events), color='black', s=40, label="Event Points")  

    ax_final.legend()
    ax_final.text(0.02, 0.98, f'Optimal Length: {min_length:.2f}', transform=ax_final.transAxes, verticalalignment='top', fontsize=12, color='magenta')  # Display length
    plt.show()

plt.show()
show_final_plot()


