"""Create txt files with the seasonal averages and standard deviations
for each basin and each variable.
Takes as input the path of the mask, the input dir with the results of
the avescan, the output dir to save the files and the model that we want
to consider.

Dependency: bit.sea (public OGS repo)
"""

import netCDF4 as nc
import numpy as np
import os
import re
import sys

from commons.mask import Mask
from commons.submask import SubMask
#from basins.COASTAL12nm import NAd_coastal_basins

from basins.basin import SimpleBasin
from basins.cadeau.nad_V0 import nad


l = nad.basin_list

to_be_removed = {l[13], l[14], l[15], l[20], l[25], l[26],l[31],l[32]}

ven = SimpleBasin('Ven', l[1].region + l[2].region)
marche = SimpleBasin('Marche', l[21].region + l[27].region)

A1 = SimpleBasin('North Ven - coast', l[1].region+l[2].region+l[3].region)
A2 = SimpleBasin('North Ven - mid', l[5].region+l[6].region+l[7].region)
A3 = SimpleBasin('Mar - coast', l[21].region + l[27].region)
A4 = SimpleBasin('Mar - mid', l[22].region + l[28].region)
A5 = SimpleBasin('Mar - deep', l[23].region+ l[29].region)
A6 = SimpleBasin('Mar - open', l[24].region+l[30].region)

SUBLIST_aggr = [A1, A2, A3, A4, A5, A6]
BASIN_LIST_PL = [item for item in l if item not in to_be_removed]

BASIN_LIST = BASIN_LIST_PL + SUBLIST_aggr






def create_txt(season:str, output_dir:str, avg_array:np.array, var:str, metric:str) -> None:
    """ Write a txt file in output_dir with the mean averages or standard deviations
    (depending on 'metric' parameter and on avg_array) for variable var in season season
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if metric == "std":
        stat = 1
    else:
        stat = 0

    # Construct the filename
    filename = os.path.join(output_dir, f"{metric}.{season}_{var}_surface.txt")

    # Open the file for writing
    with open(filename, 'w') as file:
        # Write the data to the file
        for i in range(len(BASIN_LIST)):
            # Convert the array slice to a string and write to the file
            file.write(f"{avg_array[i, :, 0, stat][0]}\n")



def divide_by_season(f):
    """Returns the season based on the month specified in the
    file name (form 01 to 12)
    """
    name_parts = re.split('[.-]', f)
    day_of_the_year = name_parts[1]
    month = int(day_of_the_year[4:6])
    if month <= 3:
        return 'winter'
    elif month <= 6:
        return 'spring'
    elif month <= 9:
        return 'summer'
    else:
        return 'autumn'


def compute_seasonal_avg(list_of_files:list, path:str, var:str) -> np.array:
    """For each subbasins, compute the array of the means of the files given
    in the list_of_files
    """

    list_of_values = []

    for f in list_of_files:
        if f.endswith('.nc'):
            values_array = nc.Dataset(os.path.join(path, f))
            list_of_values.append(values_array[var][:])

    seasonal_avg = np.mean(np.array(list_of_values), axis=0)

    return seasonal_avg


 def help():
    print("Usage: python seasonal_aggragation -m <model> -ip <input_path> -op <output_path>")


if __name__ == "__main__":

    var_list = ['chl', 'thetao', 'po4', 'no3', 'so']

    model = None
    input_path = None
    output_path = None
    mesh_mask_path = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--model':
            model = sys.argv[i+1]
            i = i + 2
        elif sys.argv[i] == '--mask':
            mesh_mask_path = sys.argv[i+1]
            i = i + 2
        elif sys.argv[i] == '-ip':
            input_path = sys.argv[i+1]
            i = i + 2
        elif sys.argv[i] == '-op':
            output_path = sys.argv[i+1]
            i = i + 2
        else:
            i += 1

    if (model == None or input_path == None or output_path == None or mesh_mask_path == None):
        help()
        exit(1)

    #model = "unc_values"
    #path = f"/home/fadobbat/Documents/adriatic/basin_analysis/ensemble_ave_scan/output_avescan/bacini_aggr/{model}/standard/main"
    #mesh_mask_path = 'ensemble_ave_scan/meshmask_cadeau.nc'
    files = os.listdir(input_path)
    mask = Mask(mesh_mask_path, maskvarname="tmask_noRiver", ylevelsmatvar="gphit", xlevelsmatvar="glamt")
    #output_dir = f"/home/fadobbat/Documents/adriatic/basin_analysis/climatologies/basins_aggr/{model}"
    d_season = {'winter' : [], 'spring' : [], 'summer' : [], 'autumn' : []}

    for var in var_list:
        annual = compute_seasonal_avg(files, input_path, var)
        create_txt("annual", output_path, annual, var, "avg")
        create_txt_std("annual", output_path, annual, var, "std")

    ####################### AVERAGE BY SEASON ########################
    for f in files:
        if f.endswith('.nc'):
            season = divide_by_season(f)
            d_season[season].append(f)


    for season in d_season.keys():
        for var in var_list:
            avg_array = compute_seasonal_avg(d_season[season], path, var)
            print(f'avg computed for var {var} and season {season}')
            create_txt(season, output_dir, avg_array, var, "std")
            create_txt(season, output_dir, avg_array, var, "avg")
