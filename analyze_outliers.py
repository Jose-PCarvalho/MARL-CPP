import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
def load_memory(memory_path):
    with open(memory_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)
def load_data(path):
    T_overlaps = []
    # Load data for each k value
    T_overlaps.append(load_memory('stats/' + path + '/timesave.pkl'))
    T_overlaps_selected = []
    for T_overlap in T_overlaps:
        T_overlap_selected = [T_overlap[i] for i in range(5, 46)]
        T_overlaps_selected.append(T_overlap_selected)
    return T_overlaps_selected[0]

# Define map sizes
map_sizes = list(range(10, 51))

# Load data for "No AM"
meu_overlap_no_am = load_data('No AM')
meu_overlap_si_k = load_data('SI K=0.25')

# Initialize comparison counts
comparison_count = 0
total_comparisons = 0

# Compare "No AM" and "SI K=0.25" overlaps
for no_am, si_k in zip(meu_overlap_no_am, meu_overlap_si_k):
    for no_am_val, si_k_val in zip(no_am, si_k):
        total_comparisons += 1
        if no_am_val == si_k_val:
            comparison_count += 1

# Calculate the percentage where "No AM" < "SI K=0.25"
percentage_no_am_less_si_k = (comparison_count / total_comparisons) * 100

# Print results
print(f"Total Comparisons: {total_comparisons}")
print(f"'No AM' < 'SI K=0.25' Episodes: {comparison_count}")
print(f"Percentage of Episodes where 'No AM' < 'SI K=0.25': {percentage_no_am_less_si_k:.2f}%")

outliers_no_am = [sum(1 for value in sublist if value > 0.70) for sublist in meu_overlap_no_am]
unfinished_no_am = [50 - len(sublist) for sublist in meu_overlap_no_am]

# Calculate percentages
total_points = 50 * len(map_sizes)  # Total tasks (50 per map size)
total_outliers = sum(outliers_no_am)
total_unfinished = sum(unfinished_no_am)

percentage_outliers = (total_outliers / total_points) * 100
percentage_unfinished = (total_unfinished / total_points) * 100

unfinished_after_30 = sum(unfinished_no_am[i] for i, size in enumerate(map_sizes) if size > 30)
percentage_unfinished_after_30 = (unfinished_after_30 / total_unfinished) * 100



print(f"Total Data Points: {total_points}")
print(f"Percentage of Outliers: {percentage_outliers:.2f}%")
print(f"Percentage of Unfinished: {percentage_unfinished:.2f}%")
print(f"Combined Percentage (Outliers + Unfinished): {percentage_outliers + percentage_unfinished:.2f}%")
print(f"Unfinished Episodes After Size > 30: {unfinished_after_30}")
print(f"Percentage of Unfinished Episodes After Size > 30: {percentage_unfinished_after_30:.2f}%")

# Create a stacked bar plot
fig, ax = plt.subplots(figsize=(9, 5))

# Plot stacked bars
ax.bar(map_sizes, outliers_no_am, label=f'Outliers ($\lambda>0.70$)', color='steelblue')
ax.bar(map_sizes, unfinished_no_am, bottom=outliers_no_am, label='Unfinished', color='darkorange')

# Add a vertical line at map_size = 30
#ax.axvline(x=30, color='red', linestyle='--', linewidth=1.5, label='Map Size = 30')

# Add labels, title, and legend
ax.set_xlabel('Map Size')
ax.set_ylabel('Count')
ax.set_title('Outliers and Unfinished Tasks by Map Size With no Safety and Robustness Filter')
ax.set_xticks(map_sizes)
ax.set_xticklabels(map_sizes, rotation=45)
ax.legend()

# Add a grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()