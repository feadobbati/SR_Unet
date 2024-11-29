import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean as cmo
import netCDF4 as nc
import sys
import os

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


def statistics(var:str):
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

def plot_six_figures(year, days):

    #timeGroup[1] = timeGroup[1].split(".")[0]
    # Specify the paths and filenames of the NetCDF files
    file_path_1 = f"{TEST_PATH}OGS/Chla/Chla_{year}_{days}.nc"
    file_path_2 = f"{TEST_PATH}OGS/N3n/N3n_{year}_{days}.nc"
    file_path_3 = f"{TEST_PATH}OGS/N1p/N1p_{year}_{days}.nc"
    file_path_4 = f"{TEST_PATH}OGS/S/S_{year}_{days}.nc"
    file_path_5 = f"{TEST_PATH}OGS/T/T_{year}_{days}.nc"

    var, time = name_components(os.path.basename(file_path_1))
    timeGroup = [time[0], time[1].split(".")[0]]

    # Open the NetCDF files
    nc_file_1 = nc.Dataset(file_path_1, "r")
    nc_file_2 = nc.Dataset(file_path_2, "r")
    nc_file_3 = nc.Dataset(file_path_3, "r")
    nc_file_4 = nc.Dataset(file_path_4, "r")
    nc_file_5 = nc.Dataset(file_path_5, "r")
    print(nc_file_3)

    # Read the data variables from each NetCDF file
    data_var_1 = nc_file_1.variables["Chla"][0, :, :]
    data_var_2 = nc_file_2.variables["N3n"][0, :, :]
    data_var_3 = nc_file_3.variables["N1p"][0, :, :]
    data_var_4 = nc_file_4.variables["S"][0, :, :]
    data_var_5 = nc_file_5.variables["T"][0, :, :]

    lon = nc_file_1['longitude'][:]
    lat = nc_file_1['latitude'][:]

    print(data_var_1.shape, data_var_2.shape, data_var_3.shape)
    print('\n')
    print(data_var_1.max(), data_var_1.min())

    #'''
    # Determine the common color scale limits for all maps
    #vmin = min(data_var_1.min(), data_var_2.min(), data_var_3.min())
    #vmax = max(data_var_1.max(), data_var_2.max(), data_var_3.max())
    #vmin, vmax, label = variables_data(var)

    # Close the NetCDF files
    nc_file_1.close()
    nc_file_2.close()
    nc_file_3.close()
    nc_file_4.close()
    nc_file_5.close()

    m1, s1 = statistics("chl")
    m2, s2 = statistics("no3")
    m3, s3 = statistics("po4")
    m4, s4 = statistics("so")
    m5, s5 = statistics("thetao")

    unc_path = "/data/test_unet/uncertainty"
    matrix_mean1 = np.load(os.path.join(unc_path, f"mean_matrix_chl_{year}-{days}.npy"))
    matrix_mean2 = np.load(os.path.join(unc_path, f"mean_matrix_no3_{year}-{days}.npy"))
    matrix_mean3 = np.load(os.path.join(unc_path, f"mean_matrix_po4_{year}-{days}.npy"))
    matrix_mean4 = np.load(os.path.join(unc_path, f"mean_matrix_so_{year}-{days}.npy"))
    matrix_mean5 = np.load(os.path.join(unc_path, f"mean_matrix_thetao_{year}-{days}.npy"))
    mask = np.load("mask.npy")

    matrix_mean1 = (matrix_mean1[0, 0, :, :] * s1) + m1
    matrix_mean2 = (matrix_mean2[0, 0, :, :] * s2) + m2
    matrix_mean3 = (matrix_mean3[0, 0, :, :]* s3) + m3
    matrix_mean4 = (matrix_mean4[0, 0, :, :]* s4) + m4
    matrix_mean5 = (matrix_mean5[0, 0, :, :]* s5) + m5

    matrix_std1 = np.load(os.path.join(unc_path, f"std_matrix_chl_{year}-{days}.npy"))
    matrix_std2 = np.load(os.path.join(unc_path, f"std_matrix_no3_{year}-{days}.npy"))
    matrix_std3 = np.load(os.path.join(unc_path, f"std_matrix_po4_{year}-{days}.npy"))
    matrix_std4 = np.load(os.path.join(unc_path, f"std_matrix_so_{year}-{days}.npy"))
    matrix_std5 = np.load(os.path.join(unc_path, f"std_matrix_thetao_{year}-{days}.npy"))
    mask = np.load("mask.npy")

    matrix_std1 = matrix_std1[0, 0, :, :] * s1
    matrix_std2 = matrix_std2[0, 0, :, :] * s2
    matrix_std3 = matrix_std3[0, 0, :, :] * s3
    matrix_std4 = matrix_std4[0, 0, :, :] * s4
    matrix_std5 = matrix_std5[0, 0, :, :] * s5

    mask = mask[0,0,:,:]

    #vmin = np.min(matrix[matrix != 0])
    #vmax = np.max(matrix[matrix != 0])

    # Plot the maps
    fig, axis = plt.subplots(3, 5, layout = 'constrained', sharey = True, gridspec_kw={'hspace': -0.5, 'wspace': 0})
    #plt.figure(figsize=(12, 4))

    cMask = mcolors.ListedColormap(['#ffffff00', '#bfbfbfff'])
    cMap = cmo.cm.algae
    cVar = cmo.cm.thermal

    #plt.subplot(1, 3, 2)
    axis[0, 0].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    pc = axis[0, 0].pcolormesh(lon, lat, data_var_1, cmap = cMap, vmin=0, vmax=3)
    axis[0, 0].set_title("Chlorophyll", fontsize = fsz-2)
    #axis[0, 1].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 0].set_ylabel("Latitude [°N]", fontsize = fsz-2)
    axis[0, 0].tick_params(axis = 'both', labelsize = fsz-4)
    axis[0, 0].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[0, 0], shrink = 0.80)
    #cb.set_label("label", fontsize = fsz-4)
    #cb.set_label("label", fontsize = fsz-4)
    #plt.colorbar()

    #plt.subplot(1, 3, 1)
    axis[0, 1].pcolormesh(lon, lat, ma.getmask(data_var_2), cmap = cMask)
    pc=axis[0, 1].pcolormesh(lon, lat, data_var_2, cmap = cMap, vmin=0, vmax=10) #, vmin=vmin, vmax=vmax)
    axis[0, 1].set_title("Nitrate", fontsize = fsz-2)
    #axis[0, 0].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 1].tick_params(axis = 'both', labelsize = fsz-4)
    axis[0, 1].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[0, 1], shrink = 0.80)
    #cb.set_label("label", fontsize = fsz-4)
    #plt.colorbar()

    #plt.subplot(1, 3, 3)
    axis[0, 2].pcolormesh(lon, lat, ma.getmask(data_var_3), cmap = cMask)
    pc = axis[0, 2].pcolormesh(lon, lat, data_var_3, cmap = cMap, vmin=0, vmax=0.4)#, vmin=vmin, vmax=vmax)
    axis[0, 2].set_title("Phosphate", fontsize = fsz-2)
    #axis[0, 2].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 2].tick_params(axis = 'both', labelsize = fsz-4)
    axis[0, 2].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[0, 2], shrink = 0.80)
    #cb.set_label(fontsize = fsz-4)
    #cb = fig.colorbar(pc, ax=axis[0, 2], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)

    axis[0, 3].pcolormesh(lon, lat, ma.getmask(data_var_4), cmap = cMask)
    pc = axis[0, 3].pcolormesh(lon, lat, data_var_4, cmap = cMap, vmin=35, vmax=39)
    axis[0, 3].set_title("Salinity", fontsize = fsz-2)
    #axis[0, 3].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 3].tick_params(axis = 'both', labelsize = fsz-4)
    axis[0, 3].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[0, 3], shrink = 0.80)
    #cb.set_label(fontsize = fsz-4)
    #cb = fig.colorbar(pc, ax=axis[0, 3], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)

    axis[0, 4].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    pc = axis[0, 4].pcolormesh(lon, lat, data_var_5, cmap = cMap, vmin=20, vmax=25)
    axis[0, 4].set_title("Temperature", fontsize = fsz-2)
    #axis[0, 4].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[0, 4].tick_params(axis = 'both', labelsize = fsz-4)
    axis[0, 4].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[0, 4], shrink = 0.80)
    #cb.set_label("label", fontsize = fsz-4)
    cb.ax.tick_params(labelsize = fsz-5)



    #plt.subplot(2, 3, 2)

    pc = axis[1, 0].pcolormesh(lon, lat, matrix_mean1, cmap = cMap, vmin=0, vmax=3)
    axis[1, 0].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 1].set_title("Target: OGS reanalysis", fontsize = fsz-2)
    axis[1, 0].set_ylabel("Latitude [°N]", fontsize = fsz-2)
    axis[1, 0].tick_params(axis = 'both', labelsize = fsz-4)
    axis[1, 0].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[1, 0], shrink = 0.80)
    #plt.colorbar()

    #plt.subplot(2, 3, 1)

    pc = axis[1, 1].pcolormesh(lon, lat, matrix_mean2, cmap = cMap, vmin=0, vmax=10)
    axis[1, 1].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 0].set_title("Input: interpolated CMS reanalysis", fontsize = fsz-2)

    axis[1, 1].tick_params(axis = 'both', labelsize = fsz-4)
    axis[1, 1].set_aspect(2**0.5)
    #plt.colorbar()
    cb = fig.colorbar(pc, ax=axis[1, 1], shrink = 0.80)
    #plt.subplot(2, 3, 3)

    pc = axis[1, 2].pcolormesh(lon, lat, matrix_mean3, cmap = cMap, vmin=0, vmax=0.4)
    axis[1, 2].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 2].set_title("Output: CNN map", fontsize = fsz-2)
    axis[1, 2].tick_params(axis = 'both', labelsize = fsz-4)
    axis[1, 2].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[1, 2], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)


    pc = axis[1, 3].pcolormesh(lon, lat, matrix_mean4, cmap = cMap, vmin=35, vmax=39)
    axis[1, 3].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 3].set_title("Output: CNN map", fontsize = fsz-2)
    axis[1, 3].tick_params(axis = 'both', labelsize = fsz-4)
    axis[1, 3].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[1, 3], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)


    pc = axis[1, 4].pcolormesh(lon, lat, matrix_mean5, cmap = cMap, vmin=20, vmax=25)
    axis[1, 4].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 4].set_title("Output: CNN map", fontsize = fsz-2)
    axis[1, 4].tick_params(axis = 'both', labelsize = fsz-4)
    axis[1, 4].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[1, 4], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)


    #plt.subplot(2, 3, 2)

    pc = axis[2, 0].pcolormesh(lon, lat, matrix_std1, cmap = cVar, vmax=0.2)#, vmin=vmin, vmax=vmax)
    axis[2, 0].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 1].set_title("Target: OGS reanalysis", fontsize = fsz-2)
    axis[2, 0].set_ylabel("Latitude [°N]", fontsize = fsz-2)
    axis[2, 0].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[2, 0].tick_params(axis = 'both', labelsize = fsz-4)
    axis[2, 0].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[2, 0], shrink = 0.80)
    #plt.colorbar()

    #plt.subplot(2, 3, 1)

    pc = axis[2, 1].pcolormesh(lon, lat, matrix_std2, cmap = cVar, vmax=2.5)#, vmin=vmin, vmax=vmax)
    axis[2, 1].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 0].set_title("Input: interpolated CMS reanalysis", fontsize = fsz-2)
    axis[2, 1].set_xlabel("Longitude [°E]", fontsize = fsz-2)

    axis[2, 1].tick_params(axis = 'both', labelsize = fsz-4)
    axis[2, 1].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[2, 1], shrink = 0.80)
    #plt.colorbar()

    #plt.subplot(2, 3, 3)

    pc = axis[2, 2].pcolormesh(lon, lat, matrix_std3, cmap = cVar, vmax=0.06)#, vmin=vmin, vmax=vmax)
    axis[2, 2].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 2].set_title("Output: CNN map", fontsize = fsz-2)
    axis[2, 2].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[2, 2].tick_params(axis = 'both', labelsize = fsz-4)
    axis[2, 2].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[2, 2], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)


    pc = axis[2, 3].pcolormesh(lon, lat, matrix_std4, cmap = cVar, vmax=0.75)#, vmin=vmin, vmax=vmax)
    axis[2, 3].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 3].set_title("Output: CNN map", fontsize = fsz-2)
    axis[2, 3].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[2, 3].tick_params(axis = 'both', labelsize = fsz-4)
    axis[2, 3].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[2, 3], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    cb.ax.tick_params(labelsize = fsz-5)


    pc = axis[2, 4].pcolormesh(lon, lat, matrix_std5, cmap = cVar, vmax=0.3)#, vmin=vmin, vmax=vmax)
    axis[2, 4].pcolormesh(lon, lat, ma.getmask(data_var_1), cmap = cMask)
    #axis[1, 4].set_title("Output: CNN map", fontsize = fsz-2)
    axis[2, 4].set_xlabel("Longitude [°E]", fontsize = fsz-2)
    axis[2, 4].tick_params(axis = 'both', labelsize = fsz-4)
    axis[2, 4].set_aspect(2**0.5)
    cb = fig.colorbar(pc, ax=axis[2, 4], shrink = 0.80)
    #cb.set_label(label, fontsize = fsz-4)
    #cb.ax.tick_params(labelsize = fsz-5)

    #fig.tight_layout()
    fig.set_size_inches([20,8], forward = True)

    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    #month = find_month(int(timeGroup[1]))
     #'''
    fig.suptitle(f'Year {year}\n5-day average September ({seasons[int(days)//18]})', fontsize=fsz+1)
    fig.savefig(f'all_report_map_{year}-{days}.jpg', pad_inches = 0, dpi = 300)
    #plt.show()


    ####
    #•••
    ####


if __name__ == "__main__":
    i = 1
    file_list = []
    cms_type = None
    year = "2013"
    days = "052"
    #while i < len(sys.argv): # unnecessary so far, but in the future we may have more arguments...
    #    if sys.argv[i] == "-f":
    #        if file_list != []: raise ValueError("Repeated input for variable")
    #        while i < len(sys.argv)-1:
    #            print(sys.argv[i+1])
    #            if not sys.argv[i+1].startswith('-'):
    #                #print(sys.argv[i+1])
    #                file_list.append(sys.argv[i+1]) ; i+= 1
    #            else:
    #                break
    #        i+= 1
    #        print(file_list)
        #if sys.argv[i] == "-t":
        #    if cms_type != None: raise ValueError("Repeated input for CMS dataset type")
        #    cms_type = sys.argv[i+1]; i+= 2
        #    if cms_type not in ["raw", "interpolated"]: raise ValueError("Data type must be either raw or interpolated")
        #else:
        #    i+=1
    #if file_list == []: raise TypeError("Missing value for variable")
    #if cms_type is None: raise TypeError("Missing value for CMS dataset type")

    #var_names= ""
    #for file in file_list:
    plot_six_figures(year, days)
