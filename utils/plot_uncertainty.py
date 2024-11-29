import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmo
import seaborn as sns
import numpy as np

def plot_nparray():
    """
    From a numpy array denoting the values taken by the variable that we want to represent,
    the mask, and the name of the variable, produce the map of that variable.
    """
    matrix = np.load("uncertainty_matrix.npy")
    mask = np.load("mask.npy")

    matrix = matrix[0, 0, :, :]
    mask = mask[0,0,:,:]

    vmin = np.min(matrix[matrix != 0])
    vmax = np.max(matrix[matrix != 0])

    cMask = mcolors.ListedColormap(['#ffffff00', '#bfbfbfff'])
    #cMap = cmo.algae

    sns.heatmap(matrix,  vmin=vmin, vmax=vmax)
    #plt.colorbar()
    plt.imshow(mask, cmap=cMask, origin='lower')
    plt.axis('off')

    # Save the plot to an image file
    plt.title('Heat_map')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    #plt.imshow()
    plt.savefig("heat_map_test2", bbox_inches='tight')
    plt.close()

plot_nparray()
