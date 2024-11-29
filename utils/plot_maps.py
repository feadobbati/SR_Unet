import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean as cmo
import netCDF4 as nc
import sys
from datetime import datetime

pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
sqrt = np.sqrt

fsz = 20

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

def name_components_ave(filename):
    comp = filename.split(".")
    var = comp[-2]
    time = comp[1]
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


def define_colors(var):
    print("var")
    if var == "thetao":
        color_map = cmo.cm.thermal
    elif var == "chl":
        color_map = cmo.cm.algae
    elif var == "so":
        color_map = cmo.cm.haline
    elif var == "no3":
        color_map = cmo.cm.matter
    elif var == "po4":
        color_map = cmo.cm.dense
    return color_map


def plot_six_figures(filename):
    var, time = name_components_ave(filename)
    #timeGroup = [time[0], time[1].split(".")[0]]
    print(time)
    #timeGroup[1] = timeGroup[1].split(".")[0]
    # Specify the paths and filenames of the NetCDF files
    file_path_1 = f"{TEST_PATH}iCMS/{var}/ave.{time}.{var}.nc"   #+ var + "_" + timeGroup[0] + "-" + timeGroup[1] + ".nc"
    file_path_2 = f"{TEST_PATH}OGS/{CMS2OGS_MAP[var]}/ave.{time}.{CMS2OGS_MAP[var]}.nc"
    file_path_3 = f"/data/test_unet/noriv_1var/{var}/ave.{time}.{var}.nc" #+ var + "/" + var + "_" + timeGroup[0] + "_" + timeGroup[1] + ".nc"
    file_path_4 = f"/data/test_unet/yesriv_1var/{var}/ave.{time}.{var}.nc" #+ var + "/" + var + "_" + timeGroup[0] + "_" + timeGroup[1] + ".nc"
    file_path_5 = f"/data/test_unet/yesriv_Multivar/{var}/ave.{time}.{var}.nc" #+ var + "/" + var + "_" + timeGroup[0] + "_" + timeGroup[1] + ".nc"

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
    fig, axis = plt.subplots(2, 5, layout = 'constrained', sharey = True)
    #plt.figure(figsize=(12, 4))

    cMask = mcolors.ListedColormap(['#ffffff00', '#bfbfbfff'])
    cMap = define_colors(var)

    #plt.subplot(1, 3, 2)
    axis[0, 0].pcolormesh(lon, lat, ma.getmask(data_var_2), cmap = cMask)
    axis[0, 0].pcolormesh(lon, lat, data_var_2, cmap = cMap, vmin=vmin, vmax=vmax)
    axis[0, 0].set_title("CADEAU", fontsize = fsz+4)
    #axis[0, 1].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 0].set_ylabel("Latitude [°N]", fontsize = fsz+2)
    axis[0, 0].tick_params(axis = 'both', labelsize = fsz)
    axis[0, 0].set_aspect(2**0.5)
    #plt.colorbar()

    #plt.subplot(1, 3, 1)
    axis[0, 1].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    axis[0, 1].pcolormesh(lon, lat, data_var_1, cmap = cMap, vmin=vmin, vmax=vmax)
    axis[0, 1].set_title("Interp", fontsize = fsz+4)
    #axis[0, 0].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 1].tick_params(axis = 'both', labelsize = fsz)
    axis[0, 1].set_aspect(2**0.5)
    #plt.colorbar()

    #plt.subplot(1, 3, 3)
    axis[0, 2].pcolormesh(lon, lat, ma.getmask(data_var_3), cmap = cMask)
    pc = axis[0, 2].pcolormesh(lon, lat, data_var_3, cmap = cMap, vmin=vmin, vmax=vmax)
    axis[0, 2].set_title("UNetNR", fontsize = fsz+4)
    #axis[0, 2].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 2].tick_params(axis = 'both', labelsize = fsz)
    axis[0, 2].set_aspect(2**0.5)
    #cb = fig.colorbar(pc, ax=axis[0, 2], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)

    axis[0, 3].pcolormesh(lon, lat, ma.getmask(data_var_4), cmap = cMask)
    pc = axis[0, 3].pcolormesh(lon, lat, data_var_4, cmap = cMap, vmin=vmin, vmax=vmax)
    axis[0, 3].set_title("UNetR", fontsize = fsz+4)
    #axis[0, 3].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 3].tick_params(axis = 'both', labelsize = fsz)
    axis[0, 3].set_aspect(2**0.5)
    #cb = fig.colorbar(pc, ax=axis[0, 3], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)

    axis[0, 4].pcolormesh(lon, lat, ma.getmask(data_var_5), cmap = cMask)
    pc = axis[0, 4].pcolormesh(lon, lat, data_var_5, cmap = cMap, vmin=vmin, vmax=vmax)
    axis[0, 4].set_title("MUNetR", fontsize = fsz+4)
    #axis[0, 4].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 4].tick_params(axis = 'both', labelsize = fsz)
    axis[0, 4].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[:, 4], shrink = 0.80)
    cb.set_label(label, fontsize = fsz+4)
    cb.ax.tick_params(labelsize = fsz)


    #plt.subplot(2, 3, 2)
    axis[1, 0].pcolormesh(lon, lat, ma.getmask(data_var_102), cmap = cMask)
    axis[1, 0].pcolormesh(lon, lat, data_var_102, cmap = cMap, vmin=vmin, vmax=vmax)
    #axis[1, 1].set_title("Target: OGS reanalysis", fontsize = fsz-2)
    axis[1, 0].set_ylabel("Latitude [°N]", fontsize = fsz+2)
    axis[1, 0].set_xlabel("Longitude [°E]", fontsize = fsz+2)
    axis[1, 0].tick_params(axis = 'both', labelsize = fsz)
    axis[1, 0].set_aspect(2**0.5)
    #plt.colorbar()

    #plt.subplot(2, 3, 1)
    axis[1, 1].pcolormesh(lon, lat, ma.getmask(data_var_101), cmap = cMask)
    axis[1, 1].pcolormesh(lon, lat, data_var_101, cmap = cMap, vmin=vmin, vmax=vmax)
    #axis[1, 0].set_title("Input: interpolated CMS reanalysis", fontsize = fsz-2)
    axis[1, 1].set_xlabel("Longitude [°E]", fontsize = fsz+2)

    axis[1, 1].tick_params(axis = 'both', labelsize = fsz)
    axis[1, 1].set_aspect(2**0.5)
    #plt.colorbar()

    #plt.subplot(2, 3, 3)
    axis[1, 2].pcolormesh(lon, lat, ma.getmask(data_var_103), cmap = cMask)
    pc = axis[1, 2].pcolormesh(lon, lat, data_var_103, cmap = cMap, vmin=vmin, vmax=vmax)
    #axis[1, 2].set_title("Output: CNN map", fontsize = fsz-2)
    axis[1, 2].set_xlabel("Longitude [°E]", fontsize = fsz+2)
    axis[1, 2].tick_params(axis = 'both', labelsize = fsz)
    axis[1, 2].set_aspect(2**0.5)
    #cb = fig.colorbar(pc, ax=axis[1, 2], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)

    axis[1, 3].pcolormesh(lon, lat, ma.getmask(data_var_104), cmap = cMask)
    pc = axis[1, 3].pcolormesh(lon, lat, data_var_104, cmap = cMap, vmin=vmin, vmax=vmax)
    #axis[1, 3].set_title("Output: CNN map", fontsize = fsz-2)
    axis[1, 3].set_xlabel("Longitude [°E]", fontsize = fsz+2)
    axis[1, 3].tick_params(axis = 'both', labelsize = fsz)
    axis[1, 3].set_aspect(2**0.5)
    #cb = fig.colorbar(pc, ax=axis[1, 3], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)

    axis[1, 4].pcolormesh(lon, lat, ma.getmask(data_var_104), cmap = cMask)
    pc = axis[1, 4].pcolormesh(lon, lat, data_var_104, cmap = cMap, vmin=vmin, vmax=vmax)
    #axis[1, 4].set_title("Output: CNN map", fontsize = fsz-2)
    axis[1, 4].set_xlabel("Longitude [°E]", fontsize = fsz+2)
    axis[1, 4].tick_params(axis = 'both', labelsize = fsz)
    axis[1, 4].set_aspect(2**0.5)
    #cb = fig.colorbar(pc, ax=axis[1, 4], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)


    #fig.tight_layout()
    fig.set_size_inches([20,8], forward = True)

    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    date_obj = datetime.strptime(time, "%Y%m%d-%H:%M:%S")

    # Extract the year and month
    year = date_obj.year
    month = date_obj.month
    #month = find_month(int(timeGroup[1]))
     #'''
    #fig.suptitle(f'Year {timeGroup[0]}\n5-day average {month} ({seasons[int(timeGroup[1])//18]})', fontsize=fsz+1)
    fig.savefig(f'{var}_report_map_{year}-{month}.jpg', pad_inches = 0, dpi = 300)
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
        plot_six_figures(file)
