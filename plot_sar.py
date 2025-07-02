import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator

def load_memory(memory_path):
    with open(memory_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def save_memory(memory, memory_path):
    with open(memory_path, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)


def plot_median_for_maps_separate(maps, base_path="stats"):
    file_types = ["timesave", "saved"]
    lista = [0, 0, 788, 698, 1240, 1802]
    labels = ["Town", "Manhattan", "Cal", "Urban"]
    plt.rcParams.update({'font.size': 12})

    for file_type in file_types:
        fig, ax = plt.subplots(figsize=(9, 5))

        for i, map_name in enumerate(maps):
            file_path = os.path.join(base_path, map_name, f"{file_type}.pkl")

            # Load the data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            indices = range(2, 6)  # Indices to process
            medians = [np.mean(data[i]) for i in indices]  # Compute means

            # Plot results
            label = labels[i]
            ax.plot(indices, medians, marker='o', label=label)

        # Formatting
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Time Save Factor")
        ax.set_title("Impact of Number of Agents on the Time Save Factor")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.grid(True)

        plt.show()


# Example usage
maps = ['vdn_ood_map1', 'vdn_ood_map2', 'vdn_ood_map3', 'vdn_ood_map4']
plot_median_for_maps_separate(maps)