import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
from sympy.printing.pretty.pretty_symbology import line_width

matplotlib.use('TkAgg')
def load_memory(memory_path):
    with open(memory_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def save_memory(memory, memory_path):
    with open(memory_path, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)

def load_data(name,timesave=True):
    data = []
    for k in name:
        if timesave:
            data.append(load_memory('stats/' + k + '/timesave.pkl'))
        else:
            data.append(load_memory('stats/' + k + '/overlap.pkl'))
    return data
def prepare_data(data):
    T_overlaps_selected = []
    for T_overlap in data:
        T_overlap_selecte = [T_overlap[i] for i in range(5, 46)]
        T_overlaps_selected.append(T_overlap_selecte)
    averages = []
    for T_overlap in T_overlaps_selected:
        # averages.append([sum(x)/len(x) for x in T_overlap])
        averages.append([np.mean(x) for x in T_overlap])
    return averages


def plot(k_values, labels, title, ymin=None, ymax=None,timesave=True):
    T_overlaps = load_data(k_values,timesave)
    averages = prepare_data(T_overlaps)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, k in enumerate(k_values):
        ax.plot(map_sizes, averages[i], label=labels[i],linewidth=2)

    if timesave:
        str = "Time Save Factor"
    else:
        str = "Overlap"
    ax.set_xlabel('Map Size')
    ax.set_ylabel('Average '+ str)
    ax.set_title(title)
    ax.legend()

    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(20, 50)

    # Set y-axis limits if specified
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)


    plt.grid()
    plt.show()


#########################################################################################################################################
plt.rcParams.update({'font.size': 12})
map_sizes = list(range(10, 51))
k_values = ['vdn_0%','vdn_2','vdn_10%','greedy']
labels = ['0% Obstacles','5% Obstacles','10% Obstacles','Fallback Policy @ 5%']
title = r'Average Time Save Factor in i.i.d Zero Shot Generalization (VDN)'

# Load data and select the relevant range
data = load_data(k_values)
T_overlaps_selected = []
for T_overlap in data:
    T_overlap_selected = [T_overlap[i] for i in range(5, 46)]  # Select data between index 5 and 45
    T_overlaps_selected.append(T_overlap_selected)

# Calculate the averages (means) and IQRs
averages = []
iqrs = []
for T_overlap in T_overlaps_selected:
    averages.append([np.mean(x) for x in T_overlap])
    q25 = [np.percentile(x, 25) for x in T_overlap]
    q75 = [np.percentile(x, 75) for x in T_overlap]
    iqrs.append([q75[i] - q25[i] for i in range(len(q25))])

# Create the plot
fig, ax = plt.subplots(figsize=(9, 5))

# Plot the means
for i, k in enumerate(k_values):
    ax.plot(map_sizes, averages[i], label=labels[i], linewidth=2)

# Plot the IQR as shaded area with label
ax.fill_between(map_sizes, np.array(averages[1]) - np.array(iqrs[1])/2,
                np.array(averages[1]) + np.array(iqrs[1])/2,
                alpha=0.3, label='5% Obstacles IQR',color='orange')  # Add label to the IQR region

# Set plot labels and title
ax.set_xlabel('Map Size')
ax.set_ylabel('Average Time Save Factor')
ax.set_title(title)

# Remove duplicate labels (the IQR label will be repeated for each line)
handles, labels = ax.get_legend_handles_labels()
unique_handles = list(dict.fromkeys(handles))  # Remove duplicates
unique_labels = list(dict.fromkeys(labels))    # Remove duplicates

ax.legend(unique_handles, unique_labels)  # Update the legend with unique entries
ax.set_ylim(0.5, 0.7)
# Customize y-axis
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Display grid and show plot
plt.grid()
plt.show()

#########################################################################################################################################
plt.rcParams.update({'font.size': 12})
map_sizes = list(range(10, 51))
k_values = ['0% obstacles','SI K=0.25','10% obstacles','greedy']
labels = ['0% Obstacles','5% Obstacles','10% Obstacles','Fallback Policy @ 5%']
title = r'Average Time Save Factor in i.i.d Zero Shot Generalization (Rainbow)'

# Load data and select the relevant range
data = load_data(k_values)
T_overlaps_selected = []
for T_overlap in data:
    T_overlap_selected = [T_overlap[i] for i in range(5, 46)]  # Select data between index 5 and 45
    T_overlaps_selected.append(T_overlap_selected)

# Calculate the averages (means) and IQRs
averages = []
iqrs = []
for T_overlap in T_overlaps_selected:
    averages.append([np.mean(x) for x in T_overlap])
    q25 = [np.percentile(x, 25) for x in T_overlap]
    q75 = [np.percentile(x, 75) for x in T_overlap]
    iqrs.append([q75[i] - q25[i] for i in range(len(q25))])

# Create the plot
fig, ax = plt.subplots(figsize=(9, 5))

# Plot the means
for i, k in enumerate(k_values):
    ax.plot(map_sizes, averages[i], label=labels[i], linewidth=2)

# Plot the IQR as shaded area with label
ax.fill_between(map_sizes, np.array(averages[1]) - np.array(iqrs[1])/2,
                np.array(averages[1]) + np.array(iqrs[1])/2,
                alpha=0.3, label='5% Obstacles IQR',color='orange')  # Add label to the IQR region

# Set plot labels and title
ax.set_xlabel('Map Size')
ax.set_ylabel('Average Time Save Factor')
ax.set_title(title)

# Remove duplicate labels (the IQR label will be repeated for each line)
handles, labels = ax.get_legend_handles_labels()
unique_handles = list(dict.fromkeys(handles))  # Remove duplicates
unique_labels = list(dict.fromkeys(labels))    # Remove duplicates

ax.legend(unique_handles, unique_labels)  # Update the legend with unique entries
ax.set_ylim(0.5, 0.7)
# Customize y-axis
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Display grid and show plot
plt.grid()
plt.show()

#########################################################################################################################################
k_values = ['vdn_2','vdn_3','vdn_4','vdn_5','vdn_10']
labels = ['2 agents','3 agents','4 agents','5 agents','10 agents']
title = 'Average Time Save for Different Number of Agents (VDN) '

plot(k_values,labels,title,0.1,0.6,timesave=True)
#########################################################################################################################################
k_values = ['vdn_2','vdn_3','vdn_4','vdn_5','vdn_10']
labels = ['2 agents','3 agents','4 agents','5 agents','10 agents']
title = 'Average Overlap for Different Number of Agents (VDN) '

plot(k_values,labels,title,0,1.1,timesave=False)
#########################################################################################################################################
k_values = ['SI K=0.25','3 agents','4 agents','5 agents','10 agents']
labels = ['2 agents ','3 agents','4 agents','5 agents','10 agents']
title = 'Average Time Save Factor for Different Number of Agents (Rainbow)'

plot(k_values,labels,title,0.1,0.6,timesave=True)
#########################################################################################################################################
k_values = ['SI K=0.25','3 agents','4 agents','5 agents','10 agents']
labels = ['2 agents','3 agents','4 agents','5 agents','10 agents']
title = 'Average Overlap for Different Number of Agents (Rainbow)'

plot(k_values,labels,title,0,1.1,timesave=False)