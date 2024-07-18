import matplotlib.colors as mcolors 
import matplotlib.pyplot as plt
import cmocean as cmo

def variables_data(var):
    """
    For each variable, it returns a min value and a max value to
    help in the visualization, and a label that can be used in the figure
    """
    if var == "chl":
        return -1, 1, "Chlorophyll-α concentration [mg m⁻³]"
        
    elif var == "so":
        return 35, 39, "Salinity"
     
    elif var == "thetao":
        return 8, 15, "Temperature"
    
    elif var == "dissic":
        return 2250, 2350, "dissic"
    
    elif var == "nh4":
        return 0, 1, "nh4"
 
    elif var == "no3":
        return 0, 15, "no3"
        
    elif var == "po4":
        return 0, 0.5, "po4"

    elif var == "phyc":
        return 0, 40, "phyc"

    elif var == "talk":
        return  2640, 2690, "talk"
    
    elif var == "o2":
        return 250, 280, "o2"

def plot_nparray(matrix, mask, var):
    """
    From a numpy array denoting the values taken by the variable that we want to represent, 
    the mask, and the name of the variable, produce the map of that variable. 
    """
    vmin, vmax, _ = variables_data(var)

    cMask = mcolors.ListedColormap(['#ffffff00', '#bfbfbfff'])
    cMap = cmo.cm.algae

    plt.imshow(matrix, cmap=cMap, vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar()
    plt.imshow(mask, cmap=cMask, origin='lower')
    plt.axis('off')

    # Show the plot
    plt.title('Matrix')
    #plt.xlabel('X-axis Label')
    #plt.ylabel('Y-axis Label')
    plt.show()