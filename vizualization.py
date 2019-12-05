from matplotlib import pyplot as plt

def one_dim_heat_axis(state):
    heat_source = state[1]
    temps = state[0]
    print(state)
    low_temp, high_temp = min(temps), max(temps)
    ax = plt.gca()
    ax.plot(temps)
    return ax
