import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmo
import seaborn as sns
import numpy as np
import netCDF4 as nc
import os


OGS_PATH = "/data/working_data/tests/OGS"
CMS_PATH = "/data/working_data/tests/iCMS"
PREDICTION_PATH = "/data/test_unet/yesriv_1var"

MAX_LAYER = 27

def statistics(var:str):
    """ Returns mean and standard deviation for a given variable.

    """
    if var == "chl":
        return 0.46845704381393677, 0.3397931775262973

    elif var == "no3":
        return 1.6804625274144198, 2.9038157913000764

    elif var == "po4":
        return 0.05270198820228757, 0.07341956821421901

    elif var == "thetao":
        return 16.025248636708163, 4.370690887046396

    elif var == "so":
        return 38.158791917433454, 1.048470331190192


def set_label(var) -> str:
    """Returns the title of a plot, given a variable
    """
    if var == "chl":
        return "Seasonal Chlorophyll-α concentration [mg m⁻³]"

    elif var == "no3":
        return 'Seasonal nitrate concentration [mmol m⁻³]'

    elif var == "po4":
        return 'Seasonal phosphate concentration [mmol m⁻³]'

    elif var == "thetao":
        return "Seasonal temperature [°C]"

    elif var == "so":
        return 'Seasonal salinity concentration [psu]'

    else:
        raise ValueError(f"Unknown variable: {var}")



def mean_nc_files(path:str, var:str, surface=True):
    """ Compute the mean, max, and min values of the files in the directory at
    path passed as argument for variable var, divided by season.
    Returns an array for each metric, with 4 values, one for each season.
    """

    seasons = (0, 1, 2, 3)
    arr_mean = np.zeros(len(seasons))
    arr_std = np.zeros(len(seasons))
    arr_min = np.zeros(len(seasons))
    arr_max = np.zeros(len(seasons))
    d_std = {season: [] for season in seasons}
    d_min = {season: [] for season in seasons}
    d_max = {season: [] for season in seasons}
    file_names = os.listdir(path)
    for file in file_names:
        period = int(file.split('.')[1][4:6]) -1  # ave naming: ave.%yyyy%mm%ddT...
        m = int(period / 3)  # divide the months inside the 4 seasons
        file_path = os.path.join(path, file)
        ds = nc.Dataset(file_path)
        if surface > 0:
            std = np.mean(ds[var][0,:,:])
            min_file = np.min(ds[var][0,:,:])
            max_file = np.max(ds[var][0,:,:])
        else:
            std = np.mean(ds[var][:])
            max_file = np.max(ds[var][:])
            min_file = np.min(ds[var][:])

        d_std[m].append(std)
        d_min[m].append(min_file)
        d_max[m].append(max_file)

    i = 0
    for key in d_std.keys():
        arr_std[i] = np.mean(np.array(d_std[key]))
        arr_min[i] = np.min(np.array(d_min[key]))
        arr_max[i] = np.max(np.array(d_max[key]))
        i += 1

    return arr_std, arr_min, arr_max


def unc_values_x_layer(path:str, var:str, layer:int):
    """Group numpy files at the specified path into seasons
    and averages the non-masked values at the specified layer.
    Returns an array with the means of the value, one for each season.
    """
    seasons = (0, 1, 2, 3)
    arr_means = np.zeros(len(seasons))
    arr_std = np.zeros(len(seasons))
    d_values = {season: [] for season in seasons}
    file_names = os.listdir(path)
    mask = np.load("mask.npy")
    for file in file_names:
        matrix = np.load(os.path.join(path, file))
        period = int(file.split('.')[0][-3:])  # name file with a sequential number before the name
        m = int(period / 18)   # 73 files divided in the 4 seasons
        avg, std = statistics(var)
        if surface:
            masked_matrix = matrix[0,layer,:,:][~mask[0,layer,:,:]] * std
        else:
            masked_matrix = matrix[~mask] * std
        mean = np.mean(masked_matrix)
        if m == 4:
            d_values[3].append(mean)
        else:
            d_values[m].append(mean)

    i = 0
    for key in d_means.keys():
        arr_means[i] = np.mean(np.array(d_values[key]))
        i += 1
    return arr_means



def unc_values(path, var, surface=True):
    seasons = (0, 1, 2, 3)
    arr_means = np.zeros(4)
    arr_std = np.zeros(4)
    d_means = {season: [] for season in seasons}
    d_std = {season: [] for season in seasons}
    file_names = os.listdir(path)
    mask = np.load("mask.npy")
    for file in file_names:
        if file.startswith(f'mean_matrix_{var}'):
                matrix = np.load(os.path.join(path, file))
                period = int(file.split('.')[0][-3:])
                m = int(period / 18)
                avg, std = statistics(var)
                if surface:
                    masked_matrix = (matrix[0,0,:,:][~mask[0,0,:,:]] * std) + avg
                else:
                    masked_matrix = (matrix[~mask] * std) + avg
                mean = np.mean(masked_matrix)

                if m == 4:
                    d_means[4].append(mean)
                else:
                    d_means[m].append(mean)
        elif file.startswith(f'std_matrix_{var}'):
                matrix = np.load(os.path.join(path, file))
                period = int(file.split('.')[0][-3:])
                m = int(period / 18)
                avg, std = statistics(var)
                if surface:
                    masked_matrix = matrix[0,0,:,:][~mask[0,0,:,:]] * std
                else:
                    masked_matrix = matrix[~mask] * std
                mean = np.mean(masked_matrix)
                if m == 4:
                    d_std[3].append(mean)
                else:
                    d_std[m].append(mean)

    i = 0
    for key in d_means.keys():
        arr_means[i] = np.mean(np.array(d_means[key]))
        arr_std[i] = np.mean(np.array(d_std[key]))
        i += 1
    return arr_means, arr_std


def plot_nparray(arr1, arr2, unc_arr, std_array, copernicus):
    """
    From a numpy array denoting the values taken by the variable that we want to represent,
    the mask, and the name of the variable, produce the map of that variable.
    """
    seasons = ['Win', 'Spr', 'Sum', 'Aut']
    label = set_label(var)

    barWidth = 0.25

    br1 = np.arange(len(seasons))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, abs(arr1-arr2), color='blue', width=barWidth)#, label='Prediction')
    #plt.bar(br2, arr2, color='red', width=barWidth)#, label='Target')
    #plt.plot(std_array, marker='x', color='green')#, label='Mean dropout')
    plt.bar(br2, abs(copernicus-arr1), color='red', width=barWidth)
    plt.bar(br3, abs(copernicus-arr2), color='purple', width=barWidth)
    #plt.fill_between(range(len(unc_arr)), unc_arr - std_array, unc_arr + std_array, color='green', alpha=0.2)  # Fill between array1 - std and array1 + std
    plt.title(label)
    plt.xticks(range(len(seasons)), seasons)  # Set x-axis labels to months
    plt.legend()
    #plt.show()

    #cMask = mcolors.ListedColormap(['#ffffff00', '#bfbfbfff'])
    #cMap = cmo.algae

    #sns.heatmap(matrix,  vmin=vmin, vmax=vmax)
    #plt.colorbar()
    #plt.imshow(mask, cmap=cMask, origin='lower')
    #plt.axis('off')

    # Save the plot to an image file

    #plt.xlabel('X-axis Label')
    #plt.ylabel('Y-axis Label')
    #plt.imshow()
    if surface:
        plt.savefig(f"seasonal_{var}_std_surface_cop", bbox_inches='tight', pad_inches = 0, dpi = 300)
    else:
        plt.savefig(f"seasonal_{var}_std_cop", bbox_inches='tight', pad_inches = 0, dpi = 300)
    plt.close()


def plot_unc(arr):
    seasons = ['winter', 'spring', 'summer', 'autumn']
    label = set_label(var)
    plt.plot(arr, marker='x', color='green')#, label='Mean dropout')
    plt.show()



def plot_lines(averages: np.array, uncertainties: np.array, targets: np.array, var: str, save_path: str = None):
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    # Define the positions on the x-axis
    x_positions = np.arange(len(seasons))

    # Create the figure and axis (adjusting figure size to make it taller)
    fig, ax = plt.subplots(figsize=(5, 7))  # 5 units wide, 7 units tall

    # Calculate the second standard deviation (2 * uncertainties)
    second_std = 2 * uncertainties

    # Plot the averages with error bars (dot and lines) for the first std (1σ)
    ax.errorbar(x_positions, averages, yerr=uncertainties, fmt='o', capsize=5, label='Average with 1σ uncertainty', color='blue')

    # Plot the second standard deviation (2σ) with wider error bars
    ax.errorbar(x_positions, averages, yerr=second_std, fmt='none', capsize=10, label='2σ uncertainty', color='gray', alpha=0.3)

    # Plot the actual targets as 'X'
    ax.scatter(x_positions, targets, marker='x', color='red', s=100, label='Actual Target')

    # Set the x-ticks to the seasons
    ax.set_xticks(x_positions)
    ax.set_xticklabels(seasons, rotation=45, ha='right')  # Rotate labels 45 degrees to save horizontal space

    # Add labels and title
    ax.set_ylabel('Value')
    names_var = {'chl' : 'Chlorophyll', 'no3' : 'Nitrate', 'po4' : 'Phosphate', 'so' : 'Salinity', 'thetao' : 'Temperature'}
    if surface == True:
        title = f'{var} (surface)'
        ax.set_title(title)
    else:
        title = f'{names_var[var]} (full 3D domain)'
        ax.set_title(title)

    title = f"{title.replace(' ', '_')}.png"
    # Add a legend
    ax.legend()

    # Adjust layout to reduce unnecessary space
    plt.tight_layout()

    save_path = os.path.join(save_path, title)

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=300)  # Save with high quality (300 dpi)

    # Show the plot
    plt.show()



var = "so"
surface = True

d_var = {"chl":"Chla","no3":"N3n", "thetao":"T", "so":"S", "po4":"N1p"}

arr_mean, arr_min, max_targ = mean_nc_files(f"/data/working_data/tests/OGS/{d_var[var]}", d_var[var], surface)
copernicus, min_targ, max_targ = mean_nc_files(f"/data/working_data/tests/iCMS/{var}", var, surface)
arr_pred, min_targ, max_targ = mean_nc_files(f"/data/test_unet/yesriv_1var/{var}", var, surface)


arr_depth = []

for i in range(0, MAX_LAYER):
    arr_layer_mean, arr_layer_unc = unc_values_x_layer(f"/data/test_unet/uncertainty/std/{var}", var, i)
    arr_depth.append(arr_layer_unc)

arr_depth = np.array(arr_depth)
print("depth", arr_depth)

saving_path = "/home/fadobbat/Documents/adriatic/basin_analysis/uncer_glob_dom"
print("target", arr_target)
print("pred", arr_pred)
print("drop", arr_std_drop)

#plot_nparray(arr_pred, arr_target, arr_mean_drop, arr_std_drop, copernicus)
plot_lines(arr_pred, arr_std_drop, arr_target,  var, saving_path)
#plot_unc(arr_std_drop)
