import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')
def load_memory(memory_path):
    with open(memory_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def save_memory(memory, memory_path):
    with open(memory_path, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)

def plot(k_values,labels,title):
    T_overlaps = []
    plt.rcParams.update({'font.size': 12})
    # Load data for each k value
    for k in k_values:
        T_overlaps.append(load_memory('stats/'+k+'/timesave.pkl'))
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

    for i, k in enumerate(k_values):
        ax.plot(map_sizes, averages[i], label=labels[i])
    # Set labels and title
    ax.set_xlabel('Map Size')
    ax.set_ylabel('Mean Time Save Factor')
    ax.set_title(title)
    # Set legend
    ax.legend()
    # Force y-axis ticks to be integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid()
    # Show the plot
    plt.show()

#########################################################################################################################################

k_values = ['SI K=0','SI K=0.25','SI K=0.5','SI K=0.75','SI K=1','new','plspls']
labels = ['SI K=0','SI K=0.25','SI K=0.5','SI K=0.75','SI K=1','NEW','pls']
title = 'Mean Time Save Factor for Different K Values with SI Reward Structure '

plot(k_values,labels,title)

#########################################################################################################################################

k_values = ['SI K=0.25','NSI K=0.25','NSI K=0.5','NSI K=0.75','NSI K=1']
labels = ['SI K=0.25','NSI K=0.25','NSI K=0.5','NSI K=0.75','NSI K=1']

title = 'Mean Time Save Factor for Different K Values with NSI Reward Structure '

plot(k_values,labels,title)

#########################################################################################################################################
k_values = ['No Random','1 agent % 0.25','SI K=0.25','1 agent % 0.75']
labels = [
    r'1 agents $\chi=0\%$',
    r'1 agent $\chi=75\%$',
    r'1 agent $\chi=50\%$',
    r'1 agent $\chi=25\%$'
]
title = r'Mean Time Save Factor for Different $\chi$ Values, 2 Agents'

plot(k_values,labels,title)

#########################################################################################################################################
title = 'Mean Time Save Factor for Different K Values, 5 Agents'
k_values = ['5 agents K=0.25','5 agents K=1N']
labels =  ['5 agents K=0.25','5 agents K=1/J']

plot(k_values,labels,title)

#########################################################################################################################################
title = r'Mean Time Save Factor for Different $\chi$ Values, 5 Agents'

# X-axis labels for your data points
k_values = ['5 agents % 0.25', '5 agents % 0.5', '5 agents K=0.25']

# Legend labels with proper LaTeX formatting for the Greek letter chi
labels = [
    r'5 agents $\chi=0.25\%$',
    r'5 agents $\chi=0.5\%$',
    r'5 agents $\chi=0.8\%$'
]

plot(k_values,labels,title)

#########################################################################################################################################

k_values = ['No Base','No Curriculum','No Random','SI K=0.25 old','SI K=0.25','K025_newnew_batch','SI K=0.25NOH']
labels = ['No Pre-Trained Model','No Curriculum Learning','No Random Number Agents','No Wait Action','Baseline','What','NOH']
title = 'Mean Time Save Factor for the Ablation Tests, 2 Agents'
plot(k_values,labels,title)

#########################################################################################################################################

k_values = ['K025_newnew_batch','SI K=0.25NOH','new_batch_meu','new']
labels = ['new_batch_4090','NOH','new_batch_meu','new']
title = 'Mean Time Save Factor for the Ablation Tests, 2 Agents'
plot(k_values,labels,title)

