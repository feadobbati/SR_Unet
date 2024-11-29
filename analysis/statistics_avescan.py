"""This script uses the bit.sea OGS repo (https://github.com/inogs/bit.sea)
It assumes that the user has already produced a list of txt files with the average
mean and standard deviation of each season. This is possible with the avescan scripts
(which produce mean and std for each file) and the seasonal aggregation file, which
produces the average by season.
"""


import netCDF4 as nc
import numpy as np
import os
import re
import sys

import matplotlib.pyplot as plt
from collections import defaultdict

from commons.mask import Mask
from commons.submask import SubMask
#from basins.COASTAL12nm import NAd_coastal_basins

from basins.basin import SimpleBasin
from basins.cadeau.nad_V0 import nad, generate_basins, OUTSIDE_NORTH, OUTSIDE_WEST, OUTSIDE_EAST, DOMAIN_LIMIT_SOUTH

l = nad.basin_list
to_be_removed = {l[13], l[14], l[15], l[20], l[25], l[26],l[31],l[32]}

indexes_removed = sorted([13, 14, 15, 20, 25, 26, 31, 32])

ven = SimpleBasin('Ven', l[1].region + l[2].region)
marche = SimpleBasin('Marche', l[21].region + l[27].region)

#A1 = SimpleBasin('North Ven - coast', l[1].region+l[2].region+l[3].region)
#A2 = SimpleBasin('North Ven - mid', l[5].region+l[6].region+l[7].region)
A3 = SimpleBasin('Mar - coast', l[21].region + l[27].region)
A4 = SimpleBasin('Mar - mid', l[22].region + l[28].region)
A5 = SimpleBasin('Mar - deep', l[23].region+ l[29].region)
A6 = SimpleBasin('Mar - open', l[24].region+l[30].region)

SUBLIST_aggr = [A3, A4, A5, A6]
BASIN_LIST_PL = [item for item in l if item not in to_be_removed]

BASIN_LIST = l + SUBLIST_aggr

indexes_aggregated = sorted([21, 22, 23, 24, 27, 28, 29, 30])
indexes_to_remove = sorted([13, 14, 15, 20, 25, 26, 31, 32, 21, 27, 22, 28, 23, 29, 24, 30])
d_indexes = {24: len(BASIN_LIST) - 1, 30: len(BASIN_LIST) - 1, 23: len(BASIN_LIST) - 2, 29: len(BASIN_LIST) - 2,
22: len(BASIN_LIST) - 3, 28: len(BASIN_LIST) - 3, 21: len(BASIN_LIST) - 4, 27: len(BASIN_LIST) - 4,
5: len(BASIN_LIST) - 5, 6: len(BASIN_LIST) - 5, 7: len(BASIN_LIST) - 5, 1: len(BASIN_LIST) - 6,
2: len(BASIN_LIST) - 6, 3: len(BASIN_LIST) - 6}

fsz = 20


def find_values_on_surface(data_lines:list, levels:bool=False) -> list:
    """The functions take a list of values read from a file and returns
    a list where the valid values are conserved, and the nan are zeroed.
    (A nan represent a sub-basin where no data was present).
    We do not delete any value in order to not lose indices positions.
    We consider only surface, if the dataset includes also deeper layers
    (levels==True) we consider only the first.
    """
    sample_values = []
    if levels == False:
        for l in data_lines:
            if l != "nan":
                sample_values.append(eval(l))
            else:
                sample_values.append(0)
    else:
        for l in data_lines:
            if l.split(' ')[0] != "nan":
                sample_values.append(eval(l.split(' ')[0]))
            else:
                sample_values.append(0)
    return sample_values


def clean_data(sample_values:list) -> np.array:
    """In order to get only the basins of interest without replications,
    we consider the NAS indices as defined in the bit.sea. This implies to
    first add the basins in which we are not interested and then to remove
    them back, together with the individual basins that we aggregated
    """
    for _, pos in enumerate(indexes_removed):
        sample_values = np.insert(sample_values, pos, np.nan)
    return np.array([item for i, item in enumerate(sample_values) if i not in indexes_to_remove])


def read_txt_to_array(file_path:str, levels:bool = False) -> np.array:
    """Read a text file at path file_path with the metrics of a model,
    and clean it to get only the subset in which we are interested.
    Levels is a boolean which holds True if the dataset consider deeper
    layers, False otherwise.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sample_values = find_values_on_surface(lines, levels)
    return clean_data(sample_values)



def find_basins_with_measures(season:str, var:str, data_path:str) -> np.array:
    """ Return an array with 1 if the basin has measures, 0 otherwise.
    The measures should be at least 5 in the annual case, 1 if we are
    considering a specific season.
    """
    measured_basins = []
    with open(os.path.join(data_path, f'Nvals_{var_eionet[var]}.txt'), 'r') as file:
        lines = file.readlines()

    for l in lines:
        if season == "annual":
            if eval(l.split(' ')[0]) > 4:
                measured_basins.append(1)
            else:
                measured_basins.append(0)
        else:
            if eval(l.split(' ')[0]) > 0:
                measured_basins.append(1)
            else:
                measured_basins.append(0)

    for i, pos in enumerate(indexes_removed):
        measured_basins = np.insert(measured_basins, pos, 0)

    return np.array([item for i, item in enumerate(measured_basins) if i not in indexes_to_remove])

    return np.array(measured_basins)


def reorder_basins(arr:np.array) -> np.array:
    """Auxiliary function for the plot of basins; it insert values
    where they are expected given the initial basin list and it copy
    the values for each basin that has been aggregated.
    It returns an array of values respecting the order expected by the
    plot function
    """
    new_arr = arr.copy()
    for i, pos in enumerate(indexes_to_remove):
        new_arr = np.insert(new_arr, pos, np.nan)

    for i, pos in enumerate(indexes_aggregated):
        new_arr[pos] = new_arr[d_indexes[pos]]

    return new_arr


def bias(data1:np.array, data2:np.array) -> np.array:
    """ Compute the bias of data1 with respect to data2
    """
    return data1 - data2


def compute_seasonal_bias_sparse_observations(season:str, var:str, file_observations:str, file_model:str) -> np.array:
    """Compute the bias of file_model with respect to file_observation for a given
    variable and season. Returns the bias array, with a value for each sub-area.
    The observations are assumed to be sparse (i.e. there may be basins with empty obs)
    """
    dir_obs = os.path.dirname(file_observations)

    filled_basins = find_basins_with_measures(season, var, dir_obs)

    data_values = read_txt_to_array(file_observations, levels=True) * filled_basins
    ar_to_compare = read_txt_to_array(file_model) * filled_basins

    bias_values = bias(ar_to_compare, data_values)

    bias_values[filled_basins == 0] = np.nan

    return bias_values


def compute_seasonal_bias_full_observations(season:str, var:str, file_observations:str, file_model:str) -> np.array:
    """Compute the bias of file_model with respect to file_observation for a given
    variable and season. Returns the bias array, with a value for each sub-area.
    The observations are assumed to be full (e.g. L4 satellite data)
    """

    data_values = read_txt_to_array(file_observations)
    ar_to_compare = read_txt_to_array(file_model)

    bias_values = bias(ar_to_compare, data_values)

    return bias_values


def compute_seasonal_general_rmse(season:str, var:str,  obs_file:str, comp_file:str) -> float:
    """Compute the RMSE of the model array for a given variable and season, given the txt file with
    the model array and the txt file with the observations (truth).
    The obs_file is supposed to be full (i.e without any nan value inside)
    """
    data_values = read_txt_to_array(obs_file, levels=True)
    ar_comp = read_txt_to_array(os.path.join(comp_file))

    values_rmse = np.sqrt(np.sum((ar_comp - data_values)**2) / len(ar_comp))

    return values_rmse


def compute_rmse_from_bias(bias_arr: np.array) -> float:
    """Compute the RMSE given the array of biases. If any, it ignore nan values during the computation
    """
    valid_values = np.isfinite(bias_arr)  # Get a boolean array of non-NaN values
    return np.sqrt(np.nansum(bias_arr[valid_values]**2) / valid_values.sum())


def full_basins_values(file_path:str) -> np.array:
    """Read the values of the basins and fill the missing values
    Returns a numpy array with values
    """
    basins_val = read_txt_to_array(file_path)
    basins_val = reorder_basins(basins_val)
    return basins_val


def plot_basins_only():
    """Plot the basins that we are considering
    """
    import matplotlib.pyplot as plt
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    n_points = 500

    plt.figure(dpi=600, figsize=(16, 9))
    axes = plt.axes(projection=ccrs.PlateCarree())
    zones = generate_basins()
    for i, zone in enumerate(zones):
        if i in indexes_removed:
            color = 'white'
        elif i in d_indexes.keys() and i in indexes_aggregated:
            color = 'C{}'.format(d_indexes[i])
        else:
            color = 'C{}'.format(i)

        zone.plot(
            lon_window=(OUTSIDE_EAST, OUTSIDE_WEST),
            lat_window=(DOMAIN_LIMIT_SOUTH - 0.05, OUTSIDE_NORTH),
            lon_points=n_points,
            lat_points=2 * n_points,
            #lat_points=n_points,
            color=color, #'C{}'.format(i),
            axes=axes,
            alpha=1
        )

    land = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='10m',
        facecolor=(0.5, 0.5, 0.5, 1),
        edgecolor="black"
    )

    axes.add_feature(land, zorder=5)
    plt.savefig('basins.png', bbox_inches='tight')
    plt.close()


def plot_n_basins(values:tuple, var:str, season:str, labels:list, avg:bool=False, descr:str="values") -> None:
    """Produces n plots (where n is len(values)) each with a map divided into subbasins,
    and associates a value to each of the subareas.
        Input:
            values: a tuple of arrays, each array includes the values for all the subbasins
            that need to be represented
            var, season: the variable and season that will be represented in the plot
            labels: list of titles for the figures; it is supposed to be len(values) = len(labels)
            The title is added only in the annual case
            avg: a bool to define whether we are considering the average values of the basin (and
            in that case avg=True) or not (e.g. the bias). It affect the max and min values represented
    """
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib import cm
    from matplotlib.colors import Normalize, TwoSlopeNorm


    n_points = 500

    num_fig = len(values)

    fig, axes = plt.subplots(1, num_fig, figsize=(num_fig * 8, 8), dpi=600, subplot_kw={'projection': ccrs.PlateCarree()})

    if num_fig == 1:
        axes = [axes]

    zones = generate_basins()

    if len(values) > 1 :
        flat_values = np.concatenate(values)
    else:
        flat_values = values
    global_min = np.nanmin(flat_values)
    global_max = np.nanmax(flat_values)

    #if avg == True:
    #    global_min = 0
    #    global_max = 0.6

    for f in range(len(values)):
        ax = axes[f]
        if global_max > 0 and global_min < 0:
            abs_max = max(global_max, abs(global_min))
            # Normalize the values to the range [0, 1]
            norm = TwoSlopeNorm(vmin=-abs_max, vmax=abs_max, vcenter=0)
        else:
            #abs_max = max(abs(global_max), abs(global_min))
            norm = Normalize(vmin=global_min, vmax=global_max)

        # Choose a colormap
        if avg == False:
            colormap = cm.seismic
        else:
            colormap = cm.viridis


        for i, zone in enumerate(zones):

            # Get the color for this zone based on the normalized value
            if np.isnan(values[f][i]):
                color = 'lightgrey'
            else:
                color = colormap(norm(values[f][i]))

            #color = colormap(norm(values[f][i]))

            zone.plot(
                lon_window=(OUTSIDE_EAST, OUTSIDE_WEST),
                lat_window=(DOMAIN_LIMIT_SOUTH - 0.05, OUTSIDE_NORTH),
                lon_points=n_points,
                lat_points=2 * n_points,
                color=color,
                axes=ax,
                alpha=1
            )

        land = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='10m',
            facecolor=(0.5, 0.5, 0.5, 1),
            edgecolor="black"
        )

        ax.add_feature(land, zorder=5)
        #if season == "annual":
        ax.set_title(f'{labels[f]}', fontsize=fsz+4, pad=24)

        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        #print(values)
        sm.set_array(values)
    # Define a new axes for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.008, 0.39])  # Adjust the position and size as needed
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.yaxis.set_tick_params(labelsize=fsz)

    #axes.add_feature(land, zorder=5)
    if avg == False:
        plt.savefig(f'basins_{var}_{season}_{descr}.png',  bbox_inches='tight')
    else:
        plt.savefig(f'avg_basins_{var}_{season}_{descr}.png',  bbox_inches='tight')

    print(f"Plotted basins_{var}_{season}.png")
    plt.close()


def plot_rmse(list_of_dict:list, var_list:list, file_description:str) -> None:
    """Produce a barplot for each variable in var_list. In each barplot, CADEAU
    (if present), Copernicus Marine, and UnetR RMSE with respect to observations
    are compared, in the annual case and for each season.
    Input:
        list_of_dict: list of dictionaries, where each dictionary has the RMSE
        for a given variable and season, for all the models
        var_list: the list of variables for which we want a barplot
        file_description: part of the name of the file being saved
    """
    num_bars = len(list_of_dict[0])

    for var in var_list:
        rmse_array = np.zeros((5, 3))
        for el in list_of_dict:
            for key, rmse in el.items():
                if key[1] == var:
                    if key[2] == "annual":
                        if key[0] == "cadeau":
                            rmse_array[0][1] = rmse
                        elif key[0] == "cms":
                            rmse_array[0][2] = rmse
                        elif key[0] == "unet":
                            rmse_array[0][0] = rmse
                    elif key[2] == "winter":
                        if key[0] == "cadeau":
                            rmse_array[1][1] = rmse
                        elif key[0] == "cms":
                            rmse_array[1][2] = rmse
                        elif key[0] == "unet":
                            rmse_array[1][0] = rmse
                    if key[2] == "spring":
                        if key[0] == "cadeau":
                            rmse_array[2][1] = rmse
                        elif key[0] == "cms":
                            rmse_array[2][2] = rmse
                        elif key[0] == "unet":
                            rmse_array[2][0] = rmse
                    if key[2] == "summer":
                        if key[0] == "cadeau":
                            rmse_array[3][1] = rmse
                        elif key[0] == "cms":
                            rmse_array[3][2] = rmse
                        elif key[0] == "unet":
                            rmse_array[3][0] = rmse
                    if key[2] == "autumn":
                        if key[0] == "cadeau":
                            rmse_array[4][1] = rmse
                        elif key[0] == "cms":
                            rmse_array[4][2] = rmse
                        elif key[0] == "unet":
                            rmse_array[4][0] = rmse
        barWidth = 0.25
        br1 = np.arange(5)

        if num_bars == 3:
            br2 = [x + barWidth for x in br1]
            br3 = [x + barWidth for x in br2]
        else:
            br3 = [x + barWidth for x in br1]

        # Updated colors: Blue for Unet, Orange for Cadeau, Green for CMS
        plt.bar(br1, rmse_array[:, 0], color='#0072B2', label='UNetR RMSE', width=barWidth)
        if num_bars == 3:
            plt.bar(br2, rmse_array[:, 1], color='#E69F00', label='CADEAU RMSE', width=barWidth)
        plt.bar(br3, rmse_array[:, 2], color='#009E73', label='Copenicus Marine RMSE', width=barWidth)

        #if var == 'thetao':
        #    plt.legend(fontsize=fsz, loc='upper left', bbox_to_anchor=(1, 1))

        plt.yticks(fontsize=fsz-4)

        # Add group labels
        group_labels = ['Annual', 'Winter', 'Spring', 'Summer', 'Autumn']
        plt.xticks([r + barWidth for r in range(5)], group_labels, fontsize=fsz, rotation=45)


        plt.savefig(f"{file_description}_{var}.png",  bbox_inches='tight')
        plt.close()






if __name__ == "__main__":

    var_insitu = ['chl', 'thetao', 'po4', 'no3', 'so']
    var_sat = ['chl', 'thetao']
    var_eionet = {'chl' : 'CHL', 'po4' : 'N1p', 'no3' : 'N3n', 'so' : 'sal', 'thetao' : 'temp'}
    insert_positions = [13, 14, 15, 20, 25, 26, 31, 32]

    path = None

    if len(sys.argv) != 2:
        print(len(sys.argv))
        print("Usage: python statistics_avescan.py  <path_dir_to_the_txt>")
        sys.exit(1)

    path = sys.argv[1]
    sat_path = os.path.join(path, 'sat_2023')
    insitu_path = os.path.join(path, 'MedBGCins_2006-2023_PLsub_aggr')
    cadeau_path = os.path.join(path, 'cadeau')
    unet_path = os.path.join(path, 'unet')
    unc_path = os.path.join(path, 'unc_values')
    cms_path = os.path.join(path, 'iCMS')
    cms2023_path = os.path.join(path, 'iCMS2023')
    unet2023_path = os.path.join(path, 'unet2023')

    insitu_dict = []
    insitu_dict_2023 = []


    for var in var_insitu:
        seasonal_std = []
        seasonal_unc = []
        for season in ['annual', 'winter', 'spring', 'summer', 'autumn']:

            #define file names
            ave_name = f'ave.{season}_{var}_surface.txt'
            std_name = f'std.{season}_{var}_surface.txt'
            insitu_file = os.path.join(insitu_path, season, f'vertprof_{var_eionet[var]}.txt')
            cadeau_file = os.path.join(cadeau_path, ave_name)
            cms_file = os.path.join(cms_path, ave_name)
            unet_file = os.path.join(unet_path, ave_name)
            unc_file = os.path.join(unc_path, ave_name)
            unet_file_std = os.path.join(unet_path, std_name)
            cms2023_file = os.path.join(cms2023_path, ave_name)
            unet2023_file = os.path.join(unet2023_path, ave_name)

            basin_unet_std = full_basins_values(unet_file_std)
            basin_unc_avg = full_basins_values(unc_file)


            if season != "annual":
                seasonal_std.append(basin_unet_std)
                seasonal_unc.append(basin_unc_avg)

            #compute biases
            cadeau_bias  = compute_seasonal_bias_sparse_observations(season, var, insitu_file, cadeau_file)
            cms_bias  = compute_seasonal_bias_sparse_observations(season, var, insitu_file, cms_file)
            unet_bias  = compute_seasonal_bias_sparse_observations(season, var, insitu_file, unet_file)

            unet_bias_2023 = compute_seasonal_bias_sparse_observations(season, var, insitu_file, unet2023_file)
            cms_bias_2023 = compute_seasonal_bias_sparse_observations(season, var, insitu_file, cms2023_file)

            #compute RMSE given the bias arrays
            rmse_cadeau = compute_rmse_from_bias(cadeau_bias)
            rmse_cms = compute_rmse_from_bias(cms_bias)
            rmse_unet = compute_rmse_from_bias(unet_bias)
            d_rmse = {('cadeau', var, season) : rmse_cadeau, ('cms', var, season) : rmse_cms, ('unet', var, season) : rmse_unet}#, ('pred2023', var, season) : values_pred2023, ('cms2023', var, season) : values_cms2023}
            insitu_dict.append(d_rmse)

            rmse_unet2023 = compute_rmse_from_bias(unet_bias_2023)
            rmse_cms2023 = compute_rmse_from_bias(cms_bias_2023)
            d_rmse = {('cms', var, season) : rmse_cms2023, ('unet', var, season) : rmse_unet2023}
            insitu_dict_2023.append(d_rmse)

            bias_x_basin_cadeau = reorder_basins(cadeau_bias)
            bias_x_basin_cms = reorder_basins(cms_bias)
            bias_x_basin_unet = reorder_basins(unet_bias)

            #plot_n_basins((bias_x_basin_cms, bias_x_basin_cadeau, bias_x_basin_unet), var, season, ['Copernicus Marine', 'CADEAU', 'UNetR'], False, "bias_test_set")
            plot_n_basins(([basin_unc_avg]), var, season, [''], True, "basin_unc_test_set")
        #plot_n_basins(seasonal_std[:2], var, season, ['', ''], True, "basin_std_win_spr_set_part1")
        #plot_n_basins(seasonal_std[2:], var, season, ['', ''], True, "basin_std_sum_aut_set_part2")
        #plot_n_basins(seasonal_unc[:2], var, season, ['Winter', 'Spring'], True, "basin_unc_win_spr_set_part1")
        #plot_n_basins(seasonal_unc[2:], var, season, ['Summer', 'Autumn'], True, "basin_unc_sum_aut_set_part2")
    plot_rmse(insitu_dict, var_insitu, "avg_barpl_test_set")
    plot_rmse(insitu_dict_2023, var_insitu, "avg_barpl_2023_insitu")

    sat_dict = []
    list_of_dict_std = []

    for var in var_sat:
        seasonal_std_2023 = []
        for season in ['annual', 'winter', 'spring', 'summer', 'autumn']:

            #define file names
            ave_name = f'ave.{season}_{var}_surface.txt'
            std_name = f'std.{season}_{var}_surface.txt'
            cms2023_file = os.path.join(cms2023_path, ave_name)
            unet2023_file = os.path.join(unet2023_path, ave_name)
            sat_file = os.path.join(sat_path, ave_name)
            std_file_cms2023 = os.path.join(cms2023_path, std_name)
            std_file_unet2023 = os.path.join(unet2023_path, std_name)
            std_file_sat2023 = os.path.join(sat_path, std_name)
            ave_file_cms2023 = os.path.join(cms2023_path, ave_name)
            ave_file_unet2023 = os.path.join(unet2023_path, ave_name)
            ave_file_sat2023 = os.path.join(sat_path, ave_name)

            #compute biases
            unet_bias_sat = compute_seasonal_bias_full_observations(season, var, sat_file, unet2023_file)
            cms_bias_sat = compute_seasonal_bias_full_observations(season, var, sat_file, cms2023_file)

            #compute RMSE given the bias arrays
            rmse_unet2023 = compute_rmse_from_bias(unet_bias_sat)
            rmse_cms2023 = compute_rmse_from_bias(cms_bias_sat)
            d_rmse = {('cms', var, season) : rmse_cms2023, ('unet', var, season) : rmse_unet2023}
            sat_dict.append(d_rmse)

            basin_cms_avg = full_basins_values(ave_file_cms2023)
            basin_unet_avg = full_basins_values(ave_file_unet2023)
            basin_sat_avg = full_basins_values(ave_file_sat2023)

            #compute RMSE on standard deviations
            rmse_std_cms = compute_seasonal_general_rmse(season, var,  std_file_sat2023, std_file_cms2023)
            rmse_std_unet = compute_seasonal_general_rmse(season, var,  std_file_sat2023, std_file_unet2023)
            d_rmse_std = {('cms', var, season) : rmse_std_cms, ('unet', var, season) : rmse_std_unet}
            list_of_dict_std.append(d_rmse_std)

            basin_cms_std = full_basins_values(std_file_cms2023)
            basin_unet_std = full_basins_values(std_file_unet2023)
            basin_sat_std = full_basins_values(std_file_sat2023)

            bias_x_basin_cms = reorder_basins(cms_bias_sat)
            bias_x_basin_unet = reorder_basins(unet_bias_sat)

            if season != "annual":
                seasonal_std_2023.append(basin_unet_std)

            #plot_n_basins((bias_x_basin_cms, bias_x_basin_unet), var, season, ['Copernicus Marine', 'UNetR'], False, "bias_sat")

            #plot_n_basins((basin_cms_avg, basin_unet_avg, basin_sat_avg), var, season, ['Copernicus Marine', 'UNetR', 'Satellite'], True, "avg_values")
            #plot_n_basins((basin_cms_std, basin_unet_std), var, season, ['Copernicus Marine', 'UNetR', 'Satellite'], True, "std_values_2im")
            #plot_n_basins(([basin_sat_std]), var, season, ['Satellite'], True, "std_values_1im")
        #plot_n_basins(seasonal_std_2023, var, season, ['', '', '', ''], True, "basin_std_seasons_2023")

    plot_rmse(sat_dict, var_sat, "avg_barpl_sat")
    plot_rmse(list_of_dict_std, var_sat,  "std_barpl_sat")
