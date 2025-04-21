'''Given a set of n random points P={(a_i,b_i)∣1≤i≤n} with all a_i distinct, compute the number of pairs (i<j) such that the slope
s_ij=(b_j−b_i)/(a_j−a_i) is greater than or equal to 0. You must design an O(nlogn) time algorithm using inversion counting. No O(n^2) 
brute-force solutions are allowed.'''

import random
def merge_sort_and_count_with_visualization(arr):
    """
    Merge sort algorithm that counts inversions with terminal-based visualization.
    """
    # Track recursion depth for indentation
    def _merge_sort_and_count(arr, depth=0):
        indent = "    " * depth
        print(f"{indent}Dividing: {arr}")
        
        if len(arr) <= 1:
            print(f"{indent}Base case: {arr}, inversions = 0")
            return arr, 0
        
        # Split the array
        mid = len(arr) // 2
        left_half, right_half = arr[:mid], arr[mid:]
        
        # Recursively sort and count inversions in left and right halves
        left_sorted, left_inv = _merge_sort_and_count(left_half, depth + 1)
        right_sorted, right_inv = _merge_sort_and_count(right_half, depth + 1)
        
        # Merge and count split inversions
        merged, split_inv = _merge_and_count(left_sorted, right_sorted, depth)
        
        total_inv = left_inv + right_inv + split_inv
        print(f"{indent}After merge: {merged}, total inversions = {total_inv}")
        return merged, total_inv
    
    def _merge_and_count(left, right, depth):
        indent = "    " * depth
        print(f"{indent}Merging: {left} and {right}")
        
        result = []
        inversions = 0
        i = j = 0
        
        # Show initial state
        print(f"{indent}  L={left}, R={right}, Result=[], Inversions=0")
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                print(f"{indent}  Take L[{i}]={left[i]}, no inversions")
                i += 1
            else:
                result.append(right[j])
                # Count inversions: all remaining elements in left form an inversion with right[j]
                new_inversions = len(left) - i
                inversions += new_inversions
                
                # Format the inversions for display
                inv_pairs = [(left[k], right[j]) for k in range(i, len(left))]
                inv_str = ", ".join(f"({a},{b})" for a, b in inv_pairs)
                
                print(f"{indent}  Take R[{j}]={right[j]}, found {new_inversions} inversions: {inv_str}")
                j += 1
            
            print(f"{indent}  L={left}, R={right}, Result={result}, Inversions={inversions}")
        
        # Add remaining elements
        if i < len(left):
            print(f"{indent}  Adding remaining left elements: {left[i:]}")
            result.extend(left[i:])
        
        if j < len(right):
            print(f"{indent}  Adding remaining right elements: {right[j:]}")
            result.extend(right[j:])
        
        print(f"{indent}  Final merge result: {result}, split inversions: {inversions}")
        return result, inversions
    
    return _merge_sort_and_count(arr)

def visualize_non_negative_slopes_calculation(points):
    """
    Visualize the calculation of non-negative slopes using inversion counting.
    """
    print("\n" + "="*80)
    print("CALCULATING NON-NEGATIVE SLOPES USING INVERSION COUNTING")
    print("="*80)
    
    print("\nPoints (before sorting):")
    for i, (a, b) in enumerate(points):
        print(f"  Point {i+1}: ({a:.2f}, {b:.2f})")
    
    # Sort points by a_i (x-coordinates)
    print("\nSorting points by x-coordinates (a values)...")
    points.sort(key=lambda p: p[0])
    
    print("\nPoints (after sorting):")
    for i, (a, b) in enumerate(points):
        print(f"  Point {i+1}: ({a:.2f}, {b:.2f})")
    
    # Extract b_i values in the sorted order
    b_values = [point[1] for point in points]
    print("\nExtracted b values (y-coordinates) in sorted order:")
    print(f"  B values: {b_values}")
    
    print("\nCounting inversions using merge sort...")
    print("-"*80)
    _, inversion_count = merge_sort_and_count_with_visualization(b_values)
    print("-"*80)
    
    # Calculate non-negative slopes
    n = len(points)
    total_pairs = n * (n - 1) // 2
    non_negative_slopes = total_pairs - inversion_count
    
    print("\nCalculating non-negative slopes:")
    print(f"  Total possible pairs: n(n-1)/2 = {n}({n-1})/2 = {total_pairs}")
    print(f"  Inversion count (negative slopes): {inversion_count}")
    print(f"  Non-negative slopes: total_pairs - inversion_count = {total_pairs} - {inversion_count} = {non_negative_slopes}")
    
    print("\nVerification with brute force:")
    count = 0
    negative_pairs = []
    non_negative_pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            a_i, b_i = points[i]
            a_j, b_j = points[j]
            slope = (b_j - b_i) / (a_j - a_i)
            
            if slope >= 0:
                count += 1
                non_negative_pairs.append((i, j, slope))
            else:
                negative_pairs.append((i, j, slope))
    
    print(f"  Non-negative slopes (brute force): {count}")
    
    print("\nNegative slope pairs (inversions):")
    for i, j, slope in negative_pairs:
        a_i, b_i = points[i]
        a_j, b_j = points[j]
        print(f"  Points ({a_i:.2f}, {b_i:.2f}) and ({a_j:.2f}, {b_j:.2f}): slope = {slope:.2f}")
    
    print("\nNon-negative slope pairs:")
    for i, j, slope in non_negative_pairs:
        a_i, b_i = points[i]
        a_j, b_j = points[j]
        print(f"  Points ({a_i:.2f}, {b_i:.2f}) and ({a_j:.2f}, {b_j:.2f}): slope = {slope:.2f}")
    
    print("\n" + "="*80)

# Example usage
if __name__ == "__main__":
    # Use a small example for clarity
    n = int(input("Input the number of points : "))
    example_points = [
        (round(random.uniform(-100,100),2),round(random.uniform(-100,100),2)) for _ in range(n)
    ]
    
    visualize_non_negative_slopes_calculation(example_points)