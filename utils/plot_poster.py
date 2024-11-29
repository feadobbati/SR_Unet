import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean as cmo
import netCDF4 as nc
import sys

pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
sqrt = np.sqrt

fsz = 18

#from mapperFunc import *
#sourceTensor.clone().detach() 133, 104, 181
##
#2014,009 ÷ 2011,023 ÷ 2008,038

#timeGroup = '2008', '038'
#TEST_PATH = "../data/NAsea/working_data/3Ddata/tests/"
TEST_PATH = "/data/working_data/tests/"
CMS2OGS_MAP = {"chl":"Chla", "dissic":"DIC", "nh4":"N4n", "no3":"N3n", "o2":"O2o", "phyc":"PhyC", "po4":"N1p", "so":"S", "talk":"Ac", "thetao":"T"}
DESTINATION_PATH = "."

def variables_data(var):
    if var == "chl":
        return 0, 1.6, "Chlorophyll-α concentration [mg m⁻³]"

    elif var == "so":
        return 35, 39, "Salinity"

    elif var == "thetao":
        return 10, 15, "Temperature [°C]"

    elif var == "dissic":
        return 2250, 2350, "dissic"

    elif var == "nh4":
        return 0, 1, "nh4"

    elif var == "no3":
        return 0, 15, "Nitrate [mmol m⁻³]"

    elif var == "po4":
        return 0, 0.4, "po4"

    elif var == "phyc":
        return 0, 40, "phyc"

    elif var == "talk":
        return  2640, 2690, "talk"

    elif var == "o2":
        return 250, 280, "o2"

def name_components(filename):
    components = filename.split("_")
    var = components[0]
    time = components[1:]
    return var, time

def find_month(num:int):
    if num < 7:
        return "January"
    elif num < 13:
        return "February"
    elif num < 19:
        return "March"
    elif num < 25:
        return "April"
    elif num < 31:
        return "May"
    elif num < 37:
        return "June"
    elif num < 43:
        return "July"
    elif num < 49:
        return "August"
    elif num < 55:
        return "September"
    elif num < 61:
        return "October"
    elif num < "67":
        return "November"
    else:
        return "December"



def plot_reanalysis(filename):
    var, time = name_components(filename)
    print("time", time)
    time = time[0].split('-')
    timeGroup = [time[0], time[1].split(".")[0]]
    print(timeGroup)
    #timeGroup[1] = timeGroup[1].split(".")[0]
    # Specify the paths and filenames of the NetCDF files
    file_path_1 = TEST_PATH + "iCMS/" + var + "/" + var + "_" + timeGroup[0] + "-" + timeGroup[1] + ".nc"
    file_path_2 = TEST_PATH + "OGS/" + CMS2OGS_MAP[var] + "/" + CMS2OGS_MAP[var] + "_" + timeGroup[0] + "_" + timeGroup[1] + ".nc"
    file_path_3 = "/data/test_unet/noriv_1var/" + var + "/" + var + "_" + timeGroup[0] + "_" + timeGroup[1] + ".nc"
    file_path_4 = "/data/test_unet/yesriv_1var/" + var + "/" + var + "_" + timeGroup[0] + "_" + timeGroup[1] + ".nc"
    file_path_5 = "/data/test_unet/yesriv_Multivar/" + var + "/" + var + "_" + timeGroup[0] + "_" + timeGroup[1] + ".nc"

    # Open the NetCDF files
    nc_file_1 = nc.Dataset(file_path_1, "r")
    nc_file_2 = nc.Dataset(file_path_2, "r")
    nc_file_3 = nc.Dataset(file_path_3, "r")
    nc_file_4 = nc.Dataset(file_path_4, "r")
    nc_file_5 = nc.Dataset(file_path_5, "r")
    print(nc_file_3)

    # Read the data variables from each NetCDF file
    data_var_1 = nc_file_1.variables[var][0, :, :]
    data_var_2 = nc_file_2.variables[CMS2OGS_MAP[var]][0, :, :]
    data_var_3 = nc_file_3.variables[var][0, :, :]
    data_var_4 = nc_file_4.variables[var][0, :, :]
    data_var_5 = nc_file_5.variables[var][0, :, :]

    data_var_101 = nc_file_1.variables[var][4, :, :]
    data_var_102 = nc_file_2.variables[CMS2OGS_MAP[var]][4, :, :]
    data_var_103 = nc_file_3.variables[var][4, :, :]
    data_var_104 = nc_file_4.variables[var][4, :, :]
    data_var_105 = nc_file_5.variables[var][4, :, :]

    lon = nc_file_1['longitude'][:]
    lat = nc_file_1['latitude'][:]
    dep = nc_file_1['depth'][:]

    print(data_var_1.shape, data_var_2.shape, data_var_3.shape)
    print('\n')
    print(data_var_1.max(), data_var_1.min())

    #'''
    # Determine the common color scale limits for all maps
    #vmin = min(data_var_1.min(), data_var_2.min(), data_var_3.min())
    #vmax = max(data_var_1.max(), data_var_2.max(), data_var_3.max())
    vmin, vmax, label = variables_data(var)

    # Close the NetCDF files
    nc_file_1.close()
    nc_file_2.close()
    nc_file_3.close()

    # Plot the maps
    fig, axis = plt.subplots(1, 2, layout = 'constrained', sharey = True)
    #plt.figure(figsize=(12, 4))
    axis = np.atleast_2d(axis)
    cMask = mcolors.ListedColormap(['#ffffff00', '#bfbfbfff'])
    cMap = cmo.cm.algae

    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    month = find_month(int(timeGroup[1]))

    #plt.subplot(1, 3, 2)

    #plt.colorbar()

    #plt.subplot(1, 3, 1)

    # Access the subplot at position (0, 0)
    #cMap = 'viridis'  # Example colormap
    vmin, vmax = 0, 2.5  # Example min and max values for the color scale

    # Plotting on the (0, 0) subplot
    pcolormesh_plot = axis[0, 0].pcolormesh(lon, lat, data_var_1, cmap=cMap, vmin=vmin, vmax=vmax)
    axis[0, 0].set_title(f'Input: Low resolution reanalysis', fontsize = fsz-2)
    #axis[0, 0].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 0].set_ylabel("Latitude [°N]", fontsize = fsz-2)
    axis[0, 0].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 0].tick_params(axis = 'both', labelsize = fsz-4)
    axis[0, 0].set_aspect(2**0.5)
    #plt.colorbar()

    #axis[0, 1].pcolormesh(lon, lat, ma.getmask(data_var_2), cmap = cMask)
    axis[0, 1].pcolormesh(lon, lat, data_var_2, cmap = cMap, vmin=vmin, vmax=vmax)
    axis[0, 1].set_title("Target: High resolution reanalysis", fontsize = fsz-2)
    axis[0, 1].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 1].tick_params(axis = 'both', labelsize = fsz-4)
    axis[0, 1].set_aspect(2**0.5)

    fig.canvas.draw()  # Force a draw to ensure the transformations are updated

    # Coordinates of the arrow in axes fraction for the start and end points
    #start = axis[0, 0].transAxes.transform((0.95, 0.5))  # Near the right edge of the left plot
    #end = axis[0, 1].transAxes.transform((0.05, 0.5))   # Near the left edge of the right plot

    #inv = fig.transFigure.inverted()
    #start_fig = inv.transform(start)
    #print(start_fig)

    #end_fig = inv.transform(end)
    #print(end_fig)
    #axis[0,0].annotate('', xy=end_fig - 0.5, xytext=start_fig, xycoords='figure fraction', textcoords='figure fraction',
    #            arrowprops=dict(arrowstyle="->", color='blue', lw=2))


    # Add colorbar for better visualization
    cbar = fig.colorbar(pcolormesh_plot, ax=axis[0, 1])
    cbar.set_label("Chlorophyll-α concentration [mg m⁻³]", fontsize = fsz-2)

    # Show plot
    #plt.show()
    #axis[0, 0].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[0, 0].pcolormesh(lon, lat, data_var_1, cmap = cMap, vmin=vmin, vmax=vmax)
    #axis[0, 0].set_title("Interpolated CMS reanalysis", fontsize = fsz-2)
    #axis[0, 0].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    #axis[0, 0].tick_params(axis = 'both', labelsize = fsz-4)
    #axis[0, 0].set_aspect(2**0.5)
    #plt.colorbar()

    #fig.tight_layout()
    fig.set_size_inches([20,8], forward = True)

    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    month = find_month(int(timeGroup[1]))
     #'''
    fig.suptitle(f'Year {timeGroup[0]}\n5-day average {month} ({seasons[int(timeGroup[1])//18]})', fontsize=fsz+1)
    fig.savefig(f'{var}_report_map_{timeGroup[0]}-{timeGroup[1]}_2poster.jpg', pad_inches = 0, dpi = 300)
    #plt.show()


    ####
    #•••
    ####


if __name__ == "__main__":
    i = 1
    file_list = []
    cms_type = None

    while i < len(sys.argv): # unnecessary so far, but in the future we may have more arguments...
        if sys.argv[i] == "-f":
            if file_list != []: raise ValueError("Repeated input for variable")
            while i < len(sys.argv)-1:
                print(sys.argv[i+1])
                if not sys.argv[i+1].startswith('-'):
                    #print(sys.argv[i+1])
                    file_list.append(sys.argv[i+1]) ; i+= 1
                else:
                    break
            i+= 1
            print(file_list)
        #if sys.argv[i] == "-t":
        #    if cms_type != None: raise ValueError("Repeated input for CMS dataset type")
        #    cms_type = sys.argv[i+1]; i+= 2
        #    if cms_type not in ["raw", "interpolated"]: raise ValueError("Data type must be either raw or interpolated")
        #else:
        #    i+=1
    if file_list == []: raise TypeError("Missing value for variable")
    #if cms_type is None: raise TypeError("Missing value for CMS dataset type")

    var_names= ""
    for file in file_list:
        plot_reanalysis(file)
