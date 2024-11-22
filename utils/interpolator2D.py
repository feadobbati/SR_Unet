# importing packages
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
#from time import time
import scipy.interpolate as intrp
import matplotlib as mpl
import matplotlib.pyplot as plt

# some useful definitions
pi = np.pi
sin = np.sin
cos = np.cos
fsz = 20

CMS2OGS_MAP = {"chl":"Chla", "dissic":"DIC", "nh4":"N4n", "no3":"N3n", "o2":"O2o", "phyc":"PhyC", "po4":"N1p", "so":"S", "talk":"Ac", "thetao":"T"}

###################################################
###################################################

#INTERPOLATION FUNCTIONS
####################################################

"""Run the function produce interpolation in order
to get the CMS dataset interpolated on the OGS grid.
The function needs as INPUT:
   - the CMS and OGS 3D (lon, lat, dep) dataset
   - the variable whose values need to be interpolated
"""

def locator(x0, xarr):
    '''Function to locate position of value x0 in array xarr'''
    i = np.argmin(np.abs(xarr - x0)) #the values do not coincides precisely
    return i

def ncOpener(DS):
    '''Function to open and extract spatial reference (lon, lat) from a netCDF4 product'''
    lon = DS['longitude'][:]
    lat = DS['latitude'][:]
    return lon, lat


def points_n_values(DS, zl, namevar, npnts):
    '''Pre-processing for the interpolation:
    gives input and output grids, makes mask and finds non-land
    points and corresponding values
    '''
    var0 = DS[namevar][zl, :, :]
    X0, Y0 = outGrid(DS)

    mask0 = ~ma.getmask(var0)
    pnts = np.empty((npnts, 2))
    pnts[:, 1] = X0[mask0].flatten()
    pnts[:, 0] = Y0[mask0].flatten()
    vals = var0[mask0]

    return pnts, vals

def check_water_presence(DS, zl, namevar):
    '''Check whether a certain depth has at least a water point,
    if no, it returns 0, otherwise, it returns the number of water points'''
    var = DS[namevar][zl, :, :]
    mask = ~ma.getmask(var)
    return np.sum(mask)

def mask_interp(DS0, DS1, zl, namevar0):
    '''Function to interpolate the CMS mask on the OGS grid'''
    var0 = DS0[namevar0][zl, :, :]
    X0, Y0 = outGrid(DS0)

    X1, Y1 = outGrid(DS1)

    maskL = ma.getmask(var0)
    npnts = maskL.shape[0] * maskL.shape[1]
    pnts = np.empty((npnts, 2))
    pnts[:, 1] = X0.flatten()
    pnts[:, 0] = Y0.flatten()

    maskOut = intrp.griddata(pnts, maskL.flatten(), (Y1, X1), method = 'nearest', fill_value = np.nan)
    return maskOut

def outGrid(DS):
	'''Function to get output mesh (from lat and lon)'''
	lon, lat = ncOpener(DS)
	return np.meshgrid(lat, lon, indexing = 'ij')

def interpolator(DS0, DS1, zl, namevar0, npnts):
    #to understand vout
	'''Function to interpolate horizontally;
    nearest neighbours method (copies values from big to small grid)
    and then covers land with output mask
    '''
	points, values = points_n_values(DS0, zl, namevar0, npnts)
	X, Y = outGrid(DS1)
	vout = ma.array(intrp.griddata(points, values, (Y, X), method = 'nearest',
    fill_value = np.nan), mask = mask_interp(DS0, DS1, zl, namevar0)) # mask during vert. interpol.
	return vout

def vert_interpolation(DS0, DS1, namevar0):
    """Function producing horizontal interpolation"""
    lon0, lat0 = ncOpener(DS0)
    lon1, lat1 = ncOpener(DS1)

    dep0 = DS0['depth'][:]
    dep1 = DS1['depth'][:]
#
    vint = ma.empty((len(dep0), len(lat1), len(lon1)), np.float32)
    for iz in range(len(dep0)):
        npnts = check_water_presence(DS0, iz, namevar0)
        if npnts != 0:
            vint[iz, :, :] = interpolator(DS0, DS1, iz, namevar0, npnts)
    maskL = ma.concatenate(
    [mask_interp(DS0, DS1, iz, namevar0) for iz in range(vint.shape[0])],
    axis = 0
    )
    vint = ma.array(vint, mask = maskL)
    return vint

def interpolate(DS0, DS1, namevar0):
    """Function interpolating the values related to the variable namevar0
    in DS0 on the grid in DS1"""
    vint = vert_interpolation(DS0, DS1, namevar0)
    dep0 = DS0['depth'][:]
    dep1 = DS1['depth'][:]

    # interpolate vertically
    vfin = ma.empty_like(vint) * np.nan # define output 3D

    # define function object with scipy.interpolate method; linear interpolation, when outside range extrapolate
    vint1d = intrp.interp1d(dep0, vint, axis = 0, kind = 'linear', fill_value = 'extrapolate')
    # interpolate along dep1 and mask according to bathymetry
    vfin = vint1d(dep1.filled(np.nan))
    vfin = ma.array(vfin, mask = ma.getmask(DS1[CMS2OGS_MAP[namevar0]][:]))
    # print(type(vfin))
    return vfin

###########################################
###########################################

#PLOT FUNCTIONS
###########################################

def plot_masks(DS0, DS1, zl, namevar0):
    fig, axs = plt.subplots(1,2)
    lon0, lat0 = ncOpener(DS0)
    lon1, lat1 = ncOpener(DS1)
    axs[0].pcolormesh(lon0, lat0, ma.getmask(DS0[namevar0][zl, :, :]))
    axs[1].pcolormesh(lon1, lat1, mask_interp(DS0, DS1, 0, namevar0))
    axs[0].set_aspect(2**0.5)
    axs[1].set_aspect(2**0.5)
    plt.show()
    return

def plot_interpolated_grids(DS0, DS1, zl, namevar0):
    coasts = np.loadtxt('new_Adriatic_coastline.txt')
    lon0, lat0 = ncOpener(DS0)
    lon1, lat1 = ncOpener(DS1)
    var0 = DS0[namevar0][zl, :, :]
    var1 = DS1[CMS2OGS_MAP[namevar0]][zl, :, :]
    print('\t', coasts.shape)
    # plotting map of interpolated variable at vertical level zl
    fig, axs = plt.subplots(2,2)
    # max value for map
    vmx = np.extract(var0 == var0, var0).max()
    vint = vert_interpolation(DS0, DS1, namevar0)
    vfin = interpolate(DS0, DS1, namevar0)
    P = axs[0,0].plot(coasts[:,0], coasts[:,1], color = 'k')
    P = axs[0,0].pcolormesh(lon0, lat0, var0, vmin = 0., vmax = vmx, edgecolors = '#0000001f', lw = 0.005)
    axs[1,1].plot(coasts[:,0], coasts[:,1], color = 'k')
    axs[1,1].pcolormesh(lon1, lat1, var1, vmin = 0., vmax = vmx, edgecolors = '#0000001f', lw = 0.005)
    axs[0,0].set_aspect(2**0.5)
    axs[1,1].set_aspect(2**0.5)
    axs[0,1].plot(coasts[:,0], coasts[:,1], color = 'k')
    axs[0,1].pcolormesh(lon1, lat1, vint[zl, :, :], vmin = 0., vmax = vmx, edgecolors = '#0000001f', lw = 0.005)
    axs[1,0].plot(coasts[:,0], coasts[:,1], color = 'k')
    axs[1,0].pcolormesh(lon1, lat1, vfin[zl, :, :], vmin = 0., vmax = vmx, edgecolors = '#0000001f', lw = 0.005)
    axs[0,1].set_aspect(2**0.5)
    axs[1,0].set_aspect(2**0.5)
    #fig.colorbar(P, ax = axs[2])
    axs[0,0].set_xlim([lon0[0], lon0[-1]]); axs[0,0].set_ylim([lat0[0], lat0[-1]])
    for axi in axs.flatten()[1:]:
        axi.set_xlim([lon1[0], lon1[-1]]); axi.set_ylim([lat1[0], lat1[-1]])
    fig.set_size_inches((20,11), forward = True)
    plt.show()
    return

def plot_vertical_profile(DS0, DS1, zl, namevar0):
    lon0, lat0 = ncOpener(DS0)
    lon1, lat1 = ncOpener(DS1)
    var0 = DS0[namevar0][zl, :, :]
    var1 = DS1[CMS2OGS_MAP[namevar0]][zl, :, :]
    dep0 = DS0['depth'][:]
    dep1 = DS1['depth'][:]
    listll = [[45.445, 13.299], [45.636, 13.598], [45.694, 13.695], [45.741, 13.600]]
    lbls = ['Out of Gulf', 'Middle of Gulf', 'Near Miramare', 'Near Isonzo']
    clrs = ['#df0000', '#ffbf00', '#008f1f', '#000f8f']
    vint = vert_interpolation(DS0, DS1, namevar0)
    vfin = interpolate(DS0, DS1, namevar0)

    ii = 0
    fig, ax = plt.subplots()
    ax.grid(True)
    for (lats, lons) in listll:
        profile1 = profiler(lats, lons, lat1, lon1, vfin)
        profile0 = profiler(lats, lons, lat1, lon1, vint)
        Pl = ax.plot(profile1, dep1, lw = 3, marker = 'o', markersize = 10, label = lbls[ii]+' W/ VERT INTERP', color = clrs[ii], alpha = 0.6)
        Pl = ax.plot(profile0, dep0, lw = 3, marker = 'X', ls = ':', markersize = 10, label = lbls[ii]+' NO VERT INTERP', color = clrs[ii])
        ii += 1
    ax.invert_yaxis()
    ax.set_ylim(top = 0.)
    ax.set_yticks(dep1)
    ax.set_xlabel(f'Chlorophyll concentration [mg m⁻³]', fontsize = fsz-2)
    #ax.set_xlabel(f'Temperature [°C]', fontsize = fsz-2)
    ax.set_ylabel(f'Depth [m]', fontsize = fsz-2)
    ax.tick_params(axis = 'both', labelsize = fsz-4)
    ax.legend(fontsize = fsz-4)
    ax.set_title('2017 Chlorophyll', fontsize = fsz)
    #ax.set_title('2017 Temperature', fontsize = fsz)
    fig.set_size_inches((20, 11), forward = True)

    plt.show()
    ## control for depths
    print(f'Depths:')
    print(f'\tInput\t||\tOutput')
    for iz in range(len(dep1)):
        print(f'\t{dep0[iz]:.2f} m\t||\t{dep1[iz]:.2f} m')
    return

def profiler(lats, lons, lat1, lon1, var):
	idx0, idx1 = locator(lats, lat1), locator(lons, lon1)
	return var[:, idx0, idx1]

def plot_all(DS0, DS1, zl, namevar0):
    plot_masks(DS0, DS1, zl, namevar0)
    plot_interpolated_grids(DS0, DS1, zl, namevar0)
    plot_vertical_profile(DS0, DS1, zl, namevar0)
    return
