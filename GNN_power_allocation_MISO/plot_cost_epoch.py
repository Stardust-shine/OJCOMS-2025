import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_cost(TEST_COST, ax, test_cost=0):
    sns.set_style("whitegrid")
    MAX_EPOCHS = TEST_COST.shape[0]
    ax.cla()
    ax.plot(np.linspace(1, MAX_EPOCHS, MAX_EPOCHS), TEST_COST)
    ax.axis([0, MAX_EPOCHS, 0, np.nanmax(TEST_COST) * 1.2])
    if test_cost == 0:
        title = '传参错误'
    else:
        title = 'TEST COST: ' + '%.6f' % test_cost
    plt.title(title)
    plt.pause(0.01)


def plot_fig(dataX, dataY, ax, markerline, color, label, markersize=10, is_fill_between=False):
    ax.plot(dataX, np.sort(np.nanmean(dataY, axis=1)), markerline, color=color, label=label)
    if is_fill_between is True:
        ax.fill_between(dataX, np.nanmin(dataY, axis=1), np.nanmax(dataY, axis=1),
                        color=(229 / 256, 204 / 256, 249 / 256), alpha=0.9)
