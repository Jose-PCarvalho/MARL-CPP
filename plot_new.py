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
    return data
def prepare_data(data):
    T_overlaps_selected = []
    for T_overlap in data:
        T_overlap_selecte = [T_overlap[i] for i in range(5, 46)]
        T_overlaps_selected.append(T_overlap_selecte)
    averages = []
    for T_overlap in T_overlaps_selected:
        # averages.append([sum(x)/len(x) for x in T_overlap])
        averages.append([np.median(x) for x in T_overlap])
    return averages


def plot(k_values, labels, title, ymin=None, ymax=None):
    T_overlaps = load_data(k_values)
    averages = prepare_data(T_overlaps)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, k in enumerate(k_values):
        ax.plot(map_sizes, averages[i], label=labels[i],linewidth=2)

    ax.set_xlabel('Map Size')
    ax.set_ylabel('Median Time Save Factor')
    ax.set_title(title)
    ax.legend()

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Set y-axis limits if specified
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    plt.grid()
    plt.show()

plt.rcParams.update({'font.size': 12})
map_sizes = list(range(10, 51))
#########################################################################################################################################

k_values = ['SI K=0.25','K=0','SI K=0.5','SI K=0.75','SI K=1']
labels = ['SI K=0.25','SI K=0','SI K=0.5','SI K=0.75','SI K=1']
title = 'Average Time Save Factor for Different K Values with SI Reward Structure '

plot(k_values,labels,title,0.5,0.95)

#########################################################################################################################################

k_values = ['SI K=0.25','NSI K=0.25','NSI K=0.5','NSI K=0.75','NSI K=1']
labels = ['SI K=0.25','NSI K=0.25','NSI K=0.5','NSI K=0.75','NSI K=1']
title = 'Average Time Save Factor for Different K Values with NSI Reward Structure '

plot(k_values,labels,title,0.5,0.95)
#########################################################################################################################################

k_values = ['2 agents chi=0.75','SI K=0.25','2 agents chi=0.25','2 agents chi=0']
labels = [r'$\chi$ = 75%',r'$\chi$ = 50%',r'$\chi$ = 25%',r'$\chi$ = 0%']
title = r'Average Time Save Factor for Different $\chi$ Values'

plot(k_values,labels,title,0.5,1.5)

#########################################################################################################################################

k_values = ['SI K=0.25','No AM','2 agents chi=0','No Base','No Curriculum']
labels = ['Baseline','No Safety and Robustness Filter','2 agents chi=0%','No Transfer Learning','No Curriculum']
title = 'Median Time Save Factor for the Ablation Tests '

plot(k_values,labels,title,0.5,1.5)

k_values = ['SI K=0.25','3 agents','4 agents','5 agents','10 agents']
labels = ['2 agents','3 agents','4 agents','5 agents','10 agents']
title = 'Average Time Save Factor for Different K Values with SI Reward Structure '

plot(k_values,labels,title)