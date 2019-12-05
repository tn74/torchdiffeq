from matplotlib import pyplot as plt
import math
import torch
import numpy as np

def axis_oned_state(state, ax=None, low_bound=None, high_bound=None, bound_scaling=0.1,
                      xlabel="X", ylabel="Temperature (F)", title="Temperature Over Time", dt=None):
    temps = state[0]
    heat_source = np.array([(x, temps[x]) for x, src in enumerate(state[1]) if abs(src - 1) < .01])
    low_temp, high_temp = low_bound or torch.max(temps), high_bound or torch.max(temps)
    temp_range = high_temp - low_temp
    low_bound, high_bound = low_temp - bound_scaling * temp_range, high_temp + bound_scaling * temp_range
    x = ax or plt.gca()
    ax.set_ylim(low_bound, high_bound)
    ax.plot(temps)
    if len(heat_source > 0):
        ax.scatter(heat_source[:, 0], heat_source[:, 1], c='r')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return ax

def graph_oned_evolution(evolution, plot_count=4, grid_shape=None, ax_size=(5, 2), figtitle = "Evolution"):
    temps = evolution[:, 0]
    bounds = torch.min(temps).item(), torch.max(temps).item()
    timesteps = evolution.size()[0]
    plot_count = plot_count or 4
    pcroot = int(math.sqrt(plot_count))
    nrows, ncols = grid_shape or (pcroot, pcroot + int(plot_count % pcroot > 0))
    figsize = (ax_size[0]*nrows, ax_size[1]*ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False, figsize=figsize)
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            dt = min(timesteps - 1, idx * timesteps//plot_count)
            ylabel = ""
            xlabel = ""
            if r == nrows-1:
                xlabel = "X"
            if c == 0:
                ylabel = "Temperature (K)"

            axis_oned_state(evolution[dt], ax=axes[r][c],
                            low_bound = bounds[0], high_bound=bounds[1],
                            title=f"DT {dt}", xlabel = xlabel, ylabel=ylabel)
            idx += 1
    fig.suptitle(figtitle)
    return fig

