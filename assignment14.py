'''Given a set of n random points P={(a_i,b_i)∣1≤i≤n} with all a_i distinct, compute the number of pairs (i<j) such that the slope
s_ij=(b_j−b_i)/(a_j−a_i) is greater than or equal to 0. You must design an O(nlogn) time algorithm using inversion counting. No O(n^2) 
brute-force solutions are allowed.'''

import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec
import time

def merge_sort_and_count_with_visualization(arr):
    """
    Merge sort algorithm that counts inversions with matplotlib visualization.
    """
    # Store all steps for animation
    animation_steps = []
    current_arrays = {}  # Track arrays at each level
    
    # Track recursion depth for indentation and visualization
    def _merge_sort_and_count(arr, depth=0, position=""):
        # Record division step
        step = {
            "type": "divide",
            "depth": depth,
            "position": position,
            "array": arr.copy(),
            "message": f"Dividing: {arr}"
        }
        animation_steps.append(step)
        current_arrays[position] = arr.copy()
        
        if len(arr) <= 1:
            step = {
                "type": "base",
                "depth": depth,
                "position": position,
                "array": arr.copy(),
                "message": f"Base case: {arr}, inversions = 0"
            }
            animation_steps.append(step)
            return arr, 0
        
        # Split the array
        mid = len(arr) // 2
        left_half, right_half = arr[:mid], arr[mid:]
        
        # Recursively sort and count inversions in left and right halves
        left_sorted, left_inv = _merge_sort_and_count(left_half, depth + 1, position + "L")
        right_sorted, right_inv = _merge_sort_and_count(right_half, depth + 1, position + "R")
        
        # Merge and count split inversions
        merged, split_inv = _merge_and_count(left_sorted, right_sorted, depth, position)
        
        total_inv = left_inv + right_inv + split_inv
        
        step = {
            "type": "merge_complete",
            "depth": depth,
            "position": position,
            "array": merged.copy(),
            "left": left_sorted.copy(),
            "right": right_sorted.copy(),
            "result": merged.copy(),
            "message": f"After merge: {merged}, total inversions = {total_inv}",
            "total_inversions": total_inv,
            "left_inversions": left_inv,
            "right_inversions": right_inv,
            "split_inversions": split_inv
        }
        animation_steps.append(step)
        current_arrays[position] = merged.copy()
        
        return merged, total_inv
    
    def _merge_and_count(left, right, depth, position):
        step = {
            "type": "merge_start",
            "depth": depth,
            "position": position,
            "left": left.copy(),
            "right": right.copy(),
            "message": f"Merging: {left} and {right}"
        }
        animation_steps.append(step)
        
        result = []
        inversions = 0
        i = j = 0
        
        # Show initial state
        step = {
            "type": "merge_step",
            "depth": depth,
            "position": position,
            "left": left.copy(),
            "right": right.copy(),
            "result": result.copy(),
            "inversions": inversions,
            "left_idx": i,
            "right_idx": j,
            "message": f"L={left}, R={right}, Result=[], Inversions=0"
        }
        animation_steps.append(step)
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                step = {
                    "type": "merge_step",
                    "depth": depth,
                    "position": position,
                    "left": left.copy(),
                    "right": right.copy(),
                    "result": result.copy(),
                    "inversions": inversions,
                    "left_idx": i,
                    "right_idx": j,
                    "take_from": "left",
                    "message": f"Take L[{i}]={left[i]}, no inversions"
                }
                animation_steps.append(step)
                i += 1
            else:
                result.append(right[j])
                # Count inversions: all remaining elements in left form an inversion with right[j]
                new_inversions = len(left) - i
                inversions += new_inversions
                
                # Format the inversions for display
                inv_pairs = [(left[k], right[j]) for k in range(i, len(left))]
                
                step = {
                    "type": "merge_step",
                    "depth": depth,
                    "position": position,
                    "left": left.copy(),
                    "right": right.copy(),
                    "result": result.copy(),
                    "inversions": inversions,
                    "left_idx": i,
                    "right_idx": j,
                    "take_from": "right",
                    "new_inversions": new_inversions,
                    "inv_pairs": inv_pairs,
                    "message": f"Take R[{j}]={right[j]}, found {new_inversions} inversions"
                }
                animation_steps.append(step)
                j += 1
            
            step = {
                "type": "merge_update",
                "depth": depth,
                "position": position,
                "left": left.copy(),
                "right": right.copy(),
                "result": result.copy(),
                "inversions": inversions,
                "left_idx": i,
                "right_idx": j,
                "message": f"L={left}, R={right}, Result={result}, Inversions={inversions}"
            }
            animation_steps.append(step)
        
        # Add remaining elements
        if i < len(left):
            step = {
                "type": "merge_remainder",
                "depth": depth,
                "position": position,
                "source": "left",
                "remaining": left[i:].copy(),
                "message": f"Adding remaining left elements: {left[i:]}"
            }
            animation_steps.append(step)
            result.extend(left[i:])
        
        if j < len(right):
            step = {
                "type": "merge_remainder",
                "depth": depth,
                "position": position,
                "source": "right",
                "remaining": right[j:].copy(),
                "message": f"Adding remaining right elements: {right[j:]}"
            }
            animation_steps.append(step)
            result.extend(right[j:])
        
        step = {
            "type": "merge_final",
            "depth": depth,
            "position": position,
            "result": result.copy(),
            "inversions": inversions,
            "message": f"Final merge result: {result}, split inversions: {inversions}"
        }
        animation_steps.append(step)
        
        return result, inversions
    
    result, inversions = _merge_sort_and_count(arr)
    
    # Create and show the animation
    create_merge_sort_animation(animation_steps, arr, result, inversions)
    
    return result, inversions

def create_merge_sort_animation(steps, original_arr, final_arr, total_inversions):
    """
    Create and display a matplotlib animation of the merge sort process with inversion counts below subarrays.
    """
    # Setup figure
    plt.rcParams.update({'font.size': 10})  # Reduced default font size
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Subplot for recursion tree
    ax_tree = fig.add_subplot(gs[0])
    ax_tree.set_title("Merge Sort Recursion Tree")
    ax_tree.axis('off')
    
    # Subplot for merge operation
    ax_merge = fig.add_subplot(gs[1])
    ax_merge.set_title("Current Merge Operation")
    ax_merge.axis('off')
    
    # Subplot for inversion formula
    ax_formula = fig.add_subplot(gs[2])
    ax_formula.set_title("Inversion Counting")
    ax_formula.axis('off')
    
    # Get maximum depth
    max_depth = max(step["depth"] for step in steps)
    
    # Calculate positioning
    box_width = 0.6  # Reduced for smaller boxes
    box_height = 0.4  # Reduced to fit closer nodes
    level_height = 0.8  # Reduced for closer vertical spacing
    
    # Initialize with empty elements
    tree_boxes = []
    merge_elements = []
    formula_text = ax_formula.text(0.5, 0.5, "", 
                                  ha='center', va='center',
                                  transform=ax_formula.transAxes)
    step_text = ax_formula.text(0.5, 0.2, "", 
                               ha='center', va='center',
                               transform=ax_formula.transAxes)
    
    # Dictionary to track positions and inversion counts of arrays in the tree
    array_positions = {}  # Format: {position: (x, y, arr, inversions)}
    
    def init():
        # Initialize axes without clearing
        ax_tree.set_xlim(0, 1.5 ** max_depth)  # Adjusted for tighter horizontal spacing
        ax_tree.set_ylim(-1, (max_depth + 1) * level_height)
        ax_tree.axis('off')
        ax_tree.set_title("Merge Sort Recursion Tree")
        
        ax_merge.set_xlim(0, 10)
        ax_merge.set_ylim(0, 3)
        ax_merge.axis('off')
        ax_merge.set_title("Current Merge Operation")
        
        ax_formula.set_xlim(0, 1)
        ax_formula.set_ylim(0, 1)
        ax_formula.axis('off')
        ax_formula.set_title("Inversion Counting")
        
        formula_text.set_text("")
        step_text.set_text("")
        
        return [formula_text, step_text]
    
    def calculate_position(position_str, depth):
        """Calculate x position in the tree based on position string with reduced spacing."""
        if not position_str:  # Root
            return (1.5 ** max_depth) / 2
        
        x = (1.5 ** max_depth) / 2
        width = (1.5 ** max_depth) / 2
        
        for char in position_str:
            width /= 2
            if char == 'L':
                x -= width
            else:  # 'R'
                x += width
        
        return x
    
    def draw_array_box(ax, x, y, arr, inversions, color='skyblue', alpha=1.0, highlighted=False):
        """Draw a box with the array elements and inversion count below with smaller text."""
        box_width_scaled = 0.6 * max(1, len(arr) * 0.3)  # Adjusted for smaller boxes
        rect = Rectangle((x - box_width_scaled/2, y - box_height/2), 
                        box_width_scaled, box_height, 
                        facecolor=color, alpha=alpha, edgecolor='black', lw=1.5)
        ax.add_patch(rect)
        
        element_width = box_width_scaled / max(1, len(arr))
        for i, val in enumerate(arr):
            element_x = x - box_width_scaled/2 + i * element_width + element_width/2
            if highlighted:
                ax.text(element_x, y, str(val), ha='center', va='center', 
                        fontweight='bold', color='red', fontsize=10)  # Reduced fontsize
            else:
                ax.text(element_x, y, str(val), ha='center', va='center', fontsize=10)  # Reduced fontsize
        
        # Display inversion count below the box
        ax.text(x, y - box_height/2 - 0.15, f"Inv: {inversions}", 
                ha='center', va='center', fontsize=7, color='black')  # Reduced fontsize
        
        return rect
    
    def draw_merge_operation(ax, step):
        """Draw the current merge operation with smaller text."""
        ax.clear()
        ax.set_title("Current Merge Operation")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')
        
        elements = []
        
        if "type" not in step:
            return elements
        
        if step["type"] == "divide":
            ax.text(5, 2.5, f"Dividing Array", ha='center', va='center', fontsize=12, fontweight='bold')
            if "array" in step:
                elements.append(draw_array_box(ax, 5, 1.5, step["array"], inversions=0))
                ax.text(5, 0.5, step["message"], ha='center', va='center', fontsize=8)
        
        elif step["type"].startswith("merge"):
            ax.text(5, 2.8, f"Merging Arrays", ha='center', va='center', fontsize=12, fontweight='bold')
            
            if "left" in step and "right" in step:
                left_len = len(step["left"])
                right_len = len(step["right"])
                
                left_box = draw_array_box(ax, 2.5, 2, step["left"], 
                                       inversions=array_positions.get(step["position"] + "L", (0, 0, [], 0))[3], 
                                       color='lightgreen', alpha=0.7)
                elements.append(left_box)
                ax.text(2.5, 2.3, "Left", ha='center', va='center', fontsize=8)
                
                right_box = draw_array_box(ax, 7.5, 2, step["right"], 
                                         inversions=array_positions.get(step["position"] + "R", (0, 0, [], 0))[3], 
                                         color='lightcoral', alpha=0.7)
                elements.append(right_box)
                ax.text(7.5, 2.3, "Right", ha='center', va='center', fontsize=8)
                
                if "left_idx" in step and "right_idx" in step:
                    if step["left_idx"] < len(step["left"]):
                        left_x = 2.5 - (left_len * 0.3)/2 + step["left_idx"] * 0.3 + 0.15
                        ax.add_patch(Rectangle((left_x - 0.2, 1.85), 0.4, 0.3, 
                                            facecolor='yellow', alpha=0.5, edgecolor='black'))
                    
                    if step["right_idx"] < len(step["right"]):
                        right_x = 7.5 - (right_len * 0.3)/2 + step["right_idx"] * 0.3 + 0.15
                        ax.add_patch(Rectangle((right_x - 0.2, 1.85), 0.4, 0.3, 
                                            facecolor='yellow', alpha=0.5, edgecolor='black'))
                
                if "result" in step:
                    result_box = draw_array_box(ax, 5, 1, step["result"], 
                                             inversions=step.get("inversions", 0), 
                                             color='lightskyblue', alpha=0.7)
                    elements.append(result_box)
                    ax.text(5, 1.3, "Result", ha='center', va='center', fontsize=8)
                
                if "take_from" in step:
                    if step["take_from"] == "left" and step["left_idx"] < len(step["left"]):
                        src_x = 2.5 - (left_len * 0.3)/2 + step["left_idx"] * 0.3 + 0.15
                        dst_x = 5 - (len(step["result"]) * 0.3)/2 + (len(step["result"]) - 1) * 0.3 + 0.15
                        arrow = FancyArrowPatch((src_x, 1.85), (dst_x, 1.15), 
                                               connectionstyle="arc3,rad=.2", 
                                               color='green', linewidth=2, arrowstyle='->')
                        ax.add_patch(arrow)
                        elements.append(arrow)
                    
                    elif step["take_from"] == "right" and step["right_idx"] < len(step["right"]):
                        src_x = 7.5 - (right_len * 0.3)/2 + step["right_idx"] * 0.3 + 0.15
                        dst_x = 5 - (len(step["result"]) * 0.3)/2 + (len(step["result"]) - 1) * 0.3 + 0.15
                        arrow = FancyArrowPatch((src_x, 1.85), (dst_x, 1.15), 
                                               connectionstyle="arc3,rad=-.2", 
                                               color='red', linewidth=2, arrowstyle='->')
                        ax.add_patch(arrow)
                        elements.append(arrow)
                
                if "new_inversions" in step and step["new_inversions"] > 0:
                    inversion_text = f"Found {step['new_inversions']} inversions!"
                    ax.text(5, 0.5, inversion_text, ha='center', va='center', 
                          color='red', fontweight='bold', fontsize=8)
                    
                    if "inv_pairs" in step:
                        inv_pairs_str = ", ".join(f"({a},{b})" for a, b in step["inv_pairs"])
                        ax.text(5, 0.3, f"Inversion pairs: {inv_pairs_str}", 
                              ha='center', va='center', fontsize=7)
            
            if "message" in step:
                ax.text(5, 0.1, step["message"], ha='center', va='center', fontsize=8)
        
        return elements
    
    def update(frame):
        """Update function for animation frames."""
        if frame >= len(steps):
            return []
        
        step = steps[frame]
        
        # Clear tree axis
        ax_tree.clear()
        ax_tree.set_xlim(0, 1.5 ** max_depth)
        ax_tree.set_ylim(-1, (max_depth + 1) * level_height)
        ax_tree.axis('off')
        ax_tree.set_title("Merge Sort Recursion Tree")
        
        # Update array positions and inversion counts
        if step["type"] in ["divide", "base"]:
            x = calculate_position(step["position"], step["depth"])
            y = (max_depth - step["depth"]) * level_height
            array_positions[step["position"]] = (x, y, step["array"], 0)  # Initialize inversions to 0
        elif step["type"] == "merge_complete":
            x = calculate_position(step["position"], step["depth"])
            y = (max_depth - step["depth"]) * level_height
            array_positions[step["position"]] = (x, y, step["array"], step["total_inversions"])  # Set final inversions
        
        # Draw all arrays in the tree
        tree_elements = []
        for pos, (x, y, arr, inversions) in array_positions.items():
            highlighted = (pos == step["position"])
            color = 'yellow' if highlighted else 'skyblue'
            alpha = 1.0 if highlighted else 0.7
            
            box = draw_array_box(ax_tree, x, y, arr, inversions, color=color, alpha=alpha, highlighted=highlighted)
            tree_elements.append(box)
            
            ax_tree.text(x, y + 0.3, pos if pos else "root", ha='center', va='center', fontsize=8)  # Adjusted position and fontsize
            
            if len(pos) > 0:
                parent_pos = pos[:-1]
                if parent_pos in array_positions:
                    parent_x, parent_y, _, _ = array_positions[parent_pos]
                    arrow = FancyArrowPatch((parent_x, parent_y - 0.3), (x, y + 0.3),
                                         connectionstyle="arc3,rad=0", 
                                         color='black', linewidth=1, arrowstyle='->')
                    ax_tree.add_patch(arrow)
                    tree_elements.append(arrow)
        
        # Draw merge operation
        merge_elements = draw_merge_operation(ax_merge, step)
        
        # Update inversion formula
        formula_str = "Inversion counting formula:\n\n"
        formula_str += "C = C_L + C_R + C_split\n\n"
        
        if "total_inversions" in step:
            formula_str += f"C_L (left inversions) = {step['left_inversions']}\n"
            formula_str += f"C_R (right inversions) = {step['right_inversions']}\n"
            formula_str += f"C_split (split inversions) = {step['split_inversions']}\n\n"
            formula_str += f"C (total inversions) = {step['total_inversions']}"
        elif "inversions" in step:
            formula_str += f"Current inversions: {step['inversions']}"
        
        formula_text.set_text(formula_str)
        step_text.set_text(f"Step {frame+1} of {len(steps)}")
        
        return tree_elements + merge_elements + [formula_text, step_text]
    
    # Create animation with blitting disabled to avoid Tkinter issues
    anim = FuncAnimation(fig, update, frames=len(steps), init_func=init, blit=False, interval=1000)
    
    # Ensure figure is retained
    plt.tight_layout()
    plt.show()
    
    # Keep a reference to the animation to prevent garbage collection
    return anim

def brute_force_non_negative_slopes(points):
    """
    Compute the number of non-negative slopes using a brute-force O(n^2) approach.
    Returns the count of pairs (i, j) where i < j and slope s_ij >= 0.
    """
    n = len(points)
    non_negative_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            a_i, b_i = points[i]
            a_j, b_j = points[j]
            slope = (b_j - b_i) / (a_j - a_i)
            if slope >= 0:
                non_negative_count += 1
    
    return non_negative_count

def visualize_non_negative_slopes_calculation(points):
    """
    Visualize the calculation of non-negative slopes using inversion counting.
    Includes brute-force verification after animation.
    """
    print("\n" + "="*80)
    print("CALCULATING NON-NEGATIVE SLOPES USING INVERSION COUNTING")
    print("="*80)
    
    # Plot the initial points
    plt.figure(figsize=(10, 6))
    a_values = [p[0] for p in points]
    b_values = [p[1] for p in points]
    plt.scatter(a_values, b_values, color='blue', s=50)
    
    # Label each point
    for i, (a, b) in enumerate(points):
        plt.annotate(f"P{i+1}: ({a:.2f}, {b:.2f})", (a, b), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title(f"Randomly chosen {len(points)} points")
    plt.xlabel("X coordinate (a)")
    plt.ylabel("Y coordinate (b)")
    plt.grid(True)
    plt.show()
    
    print("\nPoints (before sorting):")
    for i, (a, b) in enumerate(points):
        print(f"  Point {i+1}: ({a:.2f}, {b:.2f})")
    
    # Sort points by a_i (x-coordinates)
    print("\nSorting points by x-coordinates (a values)...")
    points.sort(key=lambda p: p[0])
    
    # Plot the sorted points
    # plt.figure(figsize=(10, 6))
    a_values = [p[0] for p in points]
    b_values = [p[1] for p in points]
    # plt.scatter(a_values, b_values, color='green', s=50)
    
    # Label each point
    # for i, (a, b) in enumerate(points):
    #     plt.annotate(f"P{i+1}: ({a:.2f}, {b:.2f})", (a, b), 
    #                 textcoords="offset points", xytext=(0,10), ha='center')
    
    # plt.title("Points after sorting by X-coordinate")
    # plt.xlabel("X coordinate (a)")
    # plt.ylabel("Y coordinate (b)")
    # plt.grid(True)
    # plt.show()
    
    print("\nPoints (after sorting):")
    for i, (a, b) in enumerate(points):
        print(f"  Point {i+1}: ({a:.2f}, {b:.2f})")
    
    # Extract b_i values in the sorted order
    b_values = [point[1] for point in points]
    print("\nExtracted b values (y-coordinates) in sorted order:")
    print(f"  B values: {b_values}")
    
    print("\nCounting inversions using merge sort with animation...")
    print("-"*80)
    sorted_b, inversion_count = merge_sort_and_count_with_visualization(b_values)
    print("-"*80)
    
    # Calculate non-negative slopes
    n = len(points)
    total_pairs = n * (n - 1) // 2
    non_negative_slopes = total_pairs - inversion_count
    
    print("\nCalculating non-negative slopes (O(n log n) method):")
    print(f"  Total possible pairs: n(n-1)/2 = {n}({n-1})/2 = {total_pairs}")
    print(f"  Inversion count (negative slopes): {inversion_count}")
    print(f"  Non-negative slopes: total_pairs - inversion_count = {total_pairs} - {inversion_count} = {non_negative_slopes}")
    
    # Brute-force verification
    print("\nVerifying with brute-force O(n^2) method...")
    brute_force_count = brute_force_non_negative_slopes(points)
    print(f"  Non-negative slopes (brute-force): {brute_force_count}")
    print(f"  Verification: {'Passed' if brute_force_count == non_negative_slopes else 'Failed'}")
    
    # Calculate all slopes and identify positive/negative
    negative_pairs = []
    non_negative_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            a_i, b_i = points[i]
            a_j, b_j = points[j]
            slope = (b_j - b_i) / (a_j - a_i)
            
            if slope >= 0:
                non_negative_pairs.append((i, j, slope, (a_i, b_i), (a_j, b_j)))
            else:
                negative_pairs.append((i, j, slope, (a_i, b_i), (a_j, b_j)))
    
    # Plot points with slope lines
    plt.figure(figsize=(12, 10))
    
    # Create two subplots
    plt.subplot(2, 1, 1)
    plt.scatter(a_values, b_values, color='blue', s=50)
    # Draw negative slope lines
    for _, _, _, (a_i, b_i), (a_j, b_j) in negative_pairs:
        plt.plot([a_i, a_j], [b_i, b_j], 'r-', alpha=0.5)
    plt.title(f"Negative Slopes (Inversions): {inversion_count}")
    plt.xlabel("X coordinate (a)")
    plt.ylabel("Y coordinate (b)")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.scatter(a_values, b_values, color='blue', s=50)
    # Draw non-negative slope lines
    for _, _, _, (a_i, b_i), (a_j, b_j) in non_negative_pairs:
        plt.plot([a_i, a_j], [b_i, b_j], 'g-', alpha=0.5)
    plt.title(f"Non-negative Slopes: {non_negative_slopes}")
    plt.xlabel("X coordinate (a)")
    plt.ylabel("Y coordinate (b)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)

# Example usage
if __name__ == "__main__":
    # Show a simple example of merge sort animation
    # print("Example 1: Simple merge sort animation with array [2, 3, 4, 7]")
    # merge_sort_and_count_with_visualization([2, 3, 4, 7])
    
    # # Example from the whiteboard image
    # print("\nExample 2: Merge sort with inversion counting on array [7, 8, 1, 2]")
    # merge_sort_and_count_with_visualization([7, 8, 1, 2])
    
    # # Another example from the whiteboard image
    # print("\nExample 3: Merge sort with inversion counting on array [3, 4, 2, 7]")
    # merge_sort_and_count_with_visualization([3, 4, 2, 7])
    
    # User input for random points
    try:
        n = int(input("\nInput the number of points: "))
        example_points = [
            (round(random.uniform(-100,100),2), round(random.uniform(-100,100),2)) for _ in range(n)
        ]
        
        visualize_non_negative_slopes_calculation(example_points)
    except ValueError:
        print("Please enter a valid number. Using default of 5 points.")
        example_points = [
            (round(random.uniform(-100,100),2), round(random.uniform(-100,100),2)) for _ in range(5)
        ]
        
        visualize_non_negative_slopes_calculation(example_points)