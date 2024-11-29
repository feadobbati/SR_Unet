import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
import matplotlib.cm as cmo
import seaborn as sns
import cmocean

def plot_netcdf(file, var_name):
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)

    # Open the NetCDF file
    dataset = nc.Dataset(file, 'r')

    # Extract the variable data
    var_data = dataset.variables[var_name][:]

    # Extract the coordinate data
    lon = dataset.variables['longitude'][:]
    lat = dataset.variables['latitude'][:]
    vmin = 0
    vmax = 0.3

    # Plot the data
    im = ax.pcolormesh(lon, lat, var_data[0, :, :], cmap=cmocean.cm.dense, vmin=vmin, vmax=vmax)

    # Set the labels and title
    ax.set_xlabel('Longitudine', fontsize=40)
    ax.set_ylabel('Latitudine', fontsize=40)
    #ax.set_title(f'{var_name} Distribution', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=30)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Fosfato [mmol m⁻³]", fontsize=40)
    cbar.ax.tick_params(labelsize=30)

    # Close the dataset
    dataset.close()

    # Save the plot
    plt.savefig("single_plot.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <variable_name> <file>")
        sys.exit(1)

    var_name = sys.argv[1]
    file = sys.argv[2]

    plot_netcdf(file, var_name)
