import os
import glob
import sys
import json
import netCDF4 as nc
from typing import Dict
from alive_progress import alive_bar

from utils.interpolator2D import ncOpener, interpolate

def convert_file_name(file_name:str, cms2ogs_map:Dict[str, str]) -> str:
    '''
        Converts the file name to the format of OGS file.
    
        Args:
            file_name (str): original file name.
        Returns:
            converted file name (str).
    '''
    # Split the file name into prefix and date sections
    prefix, date = file_name.split('_')
    ogs_var = cms2ogs_map[prefix]
    # Split the date into year and day sections
    year, day = date.split('-')
    # Create the new file name with the desired format
    new_file_name = f"{ogs_var}_{year}_{day.split('.')[0]}.nc"

    return new_file_name


def interpolate_data(cms_name:str, data_path:str, cms2ogs_map:Dict[str, str]) -> None:
    '''
        Performs the interpolation of a given variable and saves the results into a different folder.
    
        Args:
            cms_name (str): variable name in the CMS format.
            data_path (str): directory containing both CMS and OGS data.
            cms2ogs_map (Dict[str, str]): dictionary where keys are cms names and values are the corresponding ogs names.
    '''
    ogs_name = cms2ogs_map[cms_name]
    source_dir =  os.path.join(data_path, "nc_CMS", cms_name)
    target_dir =  os.path.join(data_path, "nc_OGS", ogs_name)
    destination_dir =  os.path.join(data_path, "nc_iCMS", cms_name)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    with alive_bar(0, title=f"Interpolating raw CMS data...") as bar:
        # Iterate over the netCDF files in the source directory
        for file_path in glob.glob(os.path.join(source_dir, "*.nc")):
            # Open the original netCDF file for reading
            with nc.Dataset(file_path, "r") as source_nc:

                source_file_name = os.path.basename(file_path)
                source_ds = nc.Dataset(os.path.join(source_dir, source_file_name))
                target_file_name = convert_file_name(source_file_name, cms2ogs_map)
                target_ds = nc.Dataset(os.path.join(target_dir, target_file_name))
                
                new_longitudes, new_latitudes = ncOpener(target_ds) # New latitude and logitudes values
                new_depth = target_ds['depth'][:]

                # Define the path for the modified version in the destination directory
                new_file_path = os.path.join(destination_dir, source_file_name)

                # Create a new netCDF file for writing
                with nc.Dataset(new_file_path, "w") as dest_nc:
                    # Create dimensions in the new file based on the interpolated grid
                    dest_nc.createDimension("depth", len(new_depth))
                    dest_nc.createDimension("longitude", len(new_longitudes))
                    dest_nc.createDimension("latitude", len(new_latitudes))
                    
                    # Create latitude and longitude variables in the new file
                    dest_depth = dest_nc.createVariable("depth", new_depth.dtype, ("depth",))
                    dest_longitudes = dest_nc.createVariable("longitude", new_longitudes.dtype, ("longitude",))
                    dest_latitudes = dest_nc.createVariable("latitude", new_latitudes.dtype, ("latitude",))
                    

                    # Write the new latitude and longitude values
                    dest_latitudes[:] = new_latitudes
                    dest_longitudes[:] = new_longitudes
                    dest_depth[:] = new_depth
                    
                    interpolated_array = interpolate(source_ds, target_ds, cms_name)

                    # Create a variable in the new file and write the interpolated array
                    dest_array = dest_nc.createVariable(cms_name, interpolated_array.dtype, ("depth", "latitude", "longitude"))
                    dest_array[:] = interpolated_array

                    # Copy global attributes from the original file
                    dest_nc.setncatts(source_nc.__dict__)

                bar()


if __name__== "__main__":
    '''
        Interpolates the files present in the source directory (nc_CMS) to match the dimensions with those of the files in the target directory (nc_OGS).
        
        Parameters:
        -dp data path
        -v set of variables to interpolate ("all" for all of them)
    '''
    i = 1
    var_list = []
    data_path = None

    while i < len(sys.argv):
        if sys.argv[i] == "-dp":
            if data_path != None: raise ValueError("Repeated input for data path")
            data_path = sys.argv[i+1]; i+= 2      
        elif sys.argv[i] == "-v":
            if var_list != []: raise ValueError("Repeated input for variable")
            if sys.argv[i+1] == "all":
                var_list = "all" ; i+= 2
            else:
                while i < len(sys.argv)-1:
                    if not sys.argv[i+1].startswith('-'):
                        var_list.append(sys.argv[i+1]) ; i+= 1
                    else:
                        break
                i+= 1
        else:
            i+= 1

    if data_path is None: raise TypeError("Missing value for data path")
    map_path =  os.path.join(data_path, "cms2ogs.json")
    with open(map_path, 'r') as f:
        cms2ogs_map = json.load(f)

    if var_list == []: raise TypeError("Missing value for variable")
    elif var_list == "all":
        var_list = list(cms2ogs_map.keys())
    
    for var in var_list:
        print(f"[interpolate_cms for variable '{var}'] Starting execution")
        interpolate_data(var,  os.path.join(data_path, "original"), cms2ogs_map)
        print(f"[interpolate_cms for variable '{var}'] Ending execution")
