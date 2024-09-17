import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def load_memory(memory_path):
    with open(memory_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def save_memory(memory, memory_path):
    with open(memory_path, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)

# Initialize lists to store data for different k values
k_values = ['5AS_complete','5AS_incomplete','k025_old','k025_new','k025_old_4','new_curriculum']# '05', '075', '1']  # Values of k without leading zeros
T_overlaps = []
plt.rcParams.update({'font.size': 12})

# Load data for each k value
for k in k_values:

    T_overlaps.append(load_memory(f'stats/timesave_2{k}.pkl'))

# Select data for map sizes from 5 to 20
map_sizes = list(range(10, 25))
T_overlaps_selected = []
for T_overlap in T_overlaps:
    T_overlap_selecte = [T_overlap[i] for i in range(5, 20)]
    T_overlaps_selected.append(T_overlap_selecte)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5.5))
averages = []
for T_overlap in T_overlaps_selected:
   averages.append([sum(x)/len(x) for x in T_overlap])
#averages = [np.mean(np.array(T_overlap) ) for T_overlap in T_overlaps_selected]
# Plot the data for different k values
labels = ['5AS_complete','5AS_incomplete','k025_old','k025_new','025_old_4','new_curriculum']
for i, k in enumerate(k_values):

    ax.plot(map_sizes, averages[i], label=f'K={labels[i]}')

# Set labels and title
ax.set_xlabel('Map Size')
ax.set_ylabel('Mean Time Save Factor')
ax.set_title('Mean Time Save Factor Evolution for Different K Values with SI Reward Structure ')

# Set legend
ax.legend()

# Force y-axis ticks to be integers
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid()
# Show the plot
plt.show()