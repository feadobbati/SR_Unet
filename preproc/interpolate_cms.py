import os
import glob
import sys
import json
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from typing import Dict
from alive_progress import alive_bar

import scipy.interpolate as intrp



def interpolate_2d(values2interp: np.array, old_lon:np.array, old_lat:np.array, new_grid:nc.Dataset, var_grid:str):
    """
    Interpolates 2D data (lat, lon) from an old grid to a new grid using nearest neighbor interpolation.

    Args:
        values2interp: 2D array (lon, lat, dep) to be interpolated.
        old_lon: 1D array of old longitudes.
        old_lat: 1D array of old latitudes.
        new_grid: xarray containing 'longitude' and 'latitude' arrays for the new grid.
        var_grid: Variable name to extract from the new grid.

    Returns:
        Interpolated 2D array on the new grid.
    """
    # Get the new longitudes, latitudes, and depths
    new_lon = new_grid['longitude'][:]
    new_lat = new_grid['latitude'][:]

    # Create 2D meshgrid for old and new grids
    old_lon, old_lat  = np.meshgrid(old_lat, old_lon,  indexing='ij')
    new_lon, new_lat = np.meshgrid(new_lat, new_lon, indexing='ij')

    # Extract the data to be interpolated
    new_data = new_grid[vargrid][:]

    # Apply mask if the data is masked (land points or invalid values)
    masked_data = np.ma.masked_invalid(values2interp)  # Mask invalid values (NaNs, etc.)
    new_mask = np.ma.masked_invalid(new_data)
    new_mask = new_mask.mask  # Extract the mask

    # Create a mask of valid (non-masked) points
    valid_mask = ~masked_data.mask

    # Apply the mask to both the coordinates and values
    points = np.vstack((old_lon[valid_mask], old_lat[valid_mask])).T  # Only valid (lon, lat, dep) points
    values = masked_data[valid_mask]  # Only valid data values

    # Create new points on the target grid for interpolation
    grid_points = np.vstack((new_lon.ravel(), new_lat.ravel())).T

    # Perform nearest neighbor interpolation
    interp_data = intrp.griddata(points, values, grid_points, method='nearest')

    # Reshape back to the grid shape of the target
    interp_data = interp_data.reshape(new_lon.shape)
    interp_data = ma.masked_array(interp_data, mask=new_mask, fill_value=1e+20, dtype=np.float32)

    return interp_data


def interpolate_3d(values2interp: np.array, old_lon:np.array, old_lat:np.array, old_dep:np.array, new_grid:nc.Dataset, var_grid:str):
    """
    Interpolates 3D data (lat, lon, dep) from an old grid to a new grid using nearest neighbor interpolation.

    Args:
        values2interp: 3D array (lon, lat, dep) to be interpolated.
        old_lon: 1D array of old longitudes.
        old_lat: 1D array of old latitudes.
        old_dep: 1D array of old depths.
        new_grid: xarray containing 'longitude', 'latitude', and 'depth' arrays for the new grid.
        var_grid: Variable name to extract from the new grid.

    Returns:
        Interpolated 3D array on the new grid.
    """
    # Get the new longitudes, latitudes, and depths
    new_lon = new_grid['longitude'][:]
    new_lat = new_grid['latitude'][:]
    new_dep = new_grid['depth'][:]

    # Create 3D meshgrid for old and new grids
    old_dep, old_lon, old_lat  = np.meshgrid(old_dep, old_lat, old_lon,  indexing='ij')
    new_dep, new_lon, new_lat = np.meshgrid(new_dep, new_lat, new_lon, indexing='ij')

    # Extract the data to be interpolated
    new_data = new_grid[var_grid][:]

    # Apply mask if the data is masked (land points or invalid values)
    masked_data = np.ma.masked_invalid(values2interp)  # Mask invalid values (NaNs, etc.)
    new_mask = np.ma.masked_invalid(new_data)
    new_mask = new_mask.mask  # Extract the mask

    # Create a mask of valid (non-masked) points
    valid_mask = ~masked_data.mask

    # Apply the mask to both the coordinates and values
    points = np.vstack((old_lon[valid_mask], old_lat[valid_mask], old_dep[valid_mask])).T  # Only valid (lon, lat, dep) points
    values = masked_data[valid_mask]  # Only valid data values

    # Create new points on the target grid for interpolation
    grid_points = np.vstack((new_lon.ravel(), new_lat.ravel(), new_dep.ravel())).T

    # Perform nearest neighbor interpolation
    interp_data = intrp.griddata(points, values, grid_points, method='nearest')

    # Reshape back to the grid shape of the target
    interp_data = interp_data.reshape(new_lon.shape)
    interp_data = ma.masked_array(interp_data, mask=new_mask, fill_value=1e+20, dtype=np.float32)

    return interp_data


def interpolate_data(cms_name: str, input_path: str, output_path: str, grid_file: str, var_grid:str, n_dim: int = 3) -> None:
    '''
        Performs the interpolation of a given variable and saves the results into a different folder.

        Args:
            cms_name (str): variable name in the Copernicus Marine format.
            input_path (str): directory containing Copernicus Marine data.
            output_path (str): directory where the interpolated files are saved
            grid_file (str): path to thethe NetCDF file that contains the grid for interpolation.
            var_grid (str): variable in the grid file of which we want to copy the shape
            n_dim (int): number of dimensions of the data; def 3 (dep, lat, lon)
    '''


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the grid file to get the new dimensions
    with nc.Dataset(grid_file, "r") as grid_nc:
        new_longitudes = grid_nc['longitude'][:]
        new_latitudes = grid_nc['latitude'][:]  # New latitude and longitude values
        if n_dim == 3:
            new_depth = grid_nc['depth'][:]

        with alive_bar(0, title=f"Interpolating raw CMS data...") as bar:
            # Iterate over the NetCDF files in the source directory
            for file_path in glob.glob(os.path.join(input_path, "*.nc")):
                # Open the original NetCDF file for reading
                with nc.Dataset(file_path, "r") as source_nc:
                    source_file_name = os.path.basename(file_path)
                    source_ds = nc.Dataset(file_path)
                    old_lon = source_ds['longitude'][:]
                    old_lat = source_ds['latitude'][:]
                    old_dep = source_ds['depth'][:]
                    old_data = source_ds[cms_name][:]
                    # Define the path for the modified version in the destination directory
                    new_file_path = os.path.join(output_path, source_file_name)
                    if not os.path.exists(new_file_path):
                    # Create a new NetCDF file for writing
                        with nc.Dataset(new_file_path, "w") as dest_nc:
                            # Create dimensions in the new file based on the interpolated grid
                            if n_dim == 3:
                                dest_nc.createDimension("depth", len(new_depth))
                            dest_nc.createDimension("longitude", len(new_longitudes))
                            dest_nc.createDimension("latitude", len(new_latitudes))

                            # Create depth, latitude, and longitude variables in the new file
                            if n_dim == 3:
                                dest_depth = dest_nc.createVariable("depth", new_depth.dtype, ("depth",))
                            dest_longitudes = dest_nc.createVariable("longitude", new_longitudes.dtype, ("longitude",))
                            dest_latitudes = dest_nc.createVariable("latitude", new_latitudes.dtype, ("latitude",))

                            # Write the new depth, latitude, and longitude values
                            if n_dim == 3:
                                dest_depth[:] = new_depth
                            dest_longitudes[:] = new_longitudes
                            dest_latitudes[:] = new_latitudes

                            # Perform interpolation
                            if n_dim == 3:
                                interpolated_array = interpolate_3d(old_data, old_lon, old_lat, old_dep, grid_nc, var_grid)
                            else:
                                interpolated_array = interpolate_2d(old_data, old_lon, old_lat, grid_nc, var_grid)
                            # Create a variable in the new file and write the interpolated array
                            if n_dim == 3:
                                spatial_dim = ("depth", "latitude", "longitude")
                            else:
                                spatial_dim = ("latitude", "longitude")
                            dest_array = dest_nc.createVariable(cms_name, interpolated_array.dtype, spatial_dim)
                            dest_array[:] = interpolated_array

                            # Copy global attributes from the original file
                            dest_nc.setncatts(source_nc.__dict__)
                        print(f"Created {new_file_path}")
                    else:
                        print(f"{new_file_path} have already been created")
                    bar()






if __name__ == "__main__":
    '''
        Interpolates the files present in the source directory (with Copernicus Marine files)
        to match the dimensions with those of the files in the target directory (with CADEAU files).
        All the files referring to given variable are assumed to be in a directory named as the variable.

        Parameters:
        -ip string with the input path (excluded vairable specific dir)
        -op string with the output path (excluded variable specific dir)
        -gp string with the path to a file with the target grid of interpolation
        -vgp string with the var in the gp file
        -v set of variables to interpolate
    '''
    i = 1
    var_list = []
    input_path = None
    output_path = None
    grid_file_path = None
    var_grid_path = None

    while i < len(sys.argv):
        if sys.argv[i] == "-ip":
            if input_path is not None: raise ValueError("Repeated input for input path")
            input_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-op":
            if output_path is not None: raise ValueError("Repeated input for output path")
            output_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-gp":
            if grid_file_path is not None: raise ValueError("Repeated input for output path")
            grid_file_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-vgp":
            if var_grid_path is not None: raise ValueError("Repeated input for output path")
            var_grid_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-v":
            if var_list != []: raise ValueError("Repeated input for variable")
            while i < len(sys.argv) - 1:
                if not sys.argv[i+1].startswith('-'):
                    var_list.append(sys.argv[i+1])
                    i += 1
                else:
                    break
            i += 1
        else:
            i += 1

    if input_path is None: raise TypeError("Missing value for input path")
    if output_path is None: raise TypeError("Missing value for output path")
    if grid_file_path is None: raise TypeError("Missing value for grid file path")

    if var_list == []: raise TypeError("Missing value for variable")
    if var_grid_path == None: raise TypeError("Missing value for grid variable")

    # Path to the file containing the grid to be used for interpolation

    for var in var_list:
        print(f"[interpolate_cms for variable '{var}'] Starting execution")
        interpolate_data(var, os.path.join(input_path, var), os.path.join(output_path, var), grid_file_path, var_grid_path, n_dim=3)
        print(f"[interpolate_cms for variable '{var}'] Ending execution")
