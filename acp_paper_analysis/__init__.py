'''
This is the hard working code in order to create publication plots
'''
# import gamma_limits_sensitivity as gls
#
# import matplotlib.pyplot as plt
import numpy as np
import acp_paper_analysis as acp
from scipy.interpolate import interpolate


def get_resources_paths():
    '''
    This function returns a dict including relative paths
    to resource files with fluxes (protons, leptons)
    '''
    fluxes_relative_paths = {
        'electron_plus_positron': acp.__path__[0]+'/resources/e_plus_e_minus_spec.dat',
        'proton': acp.__path__[0]+'/resources/proton_spec.dat',
    }

    return fluxes_relative_paths


def get_cosmic_ray_flux_interpol(
        flux_path,
        base_energy_in_TeV,
        powerlaw_slope,
        base_area_in_cm_2,
        base_time_in_sec=1.
        ):
    '''
    Function to get the interpolated cr fluxes
    from a file path

    my own units are: log10(E/TeV), sec, sr, cm^2.
    AMS02 flues are usually given in E/GeV, m^2, sec, sr
    '''
    flux_data = np.loadtxt(flux_path)

    # first, convert to normal flux, then convert to cm^2, convert to per TeV
    flux_data[:, 1] = flux_data[:, 1]/(flux_data[:, 0]**powerlaw_slope)
    flux_data[:, 1] = flux_data[:, 1]/base_area_in_cm_2
    flux_data[:, 1] = flux_data[:, 1]/base_energy_in_TeV
    flux_data[:, 1] = flux_data[:, 1]/base_time_in_sec

    # then, convert to TeV scale, then take log10
    flux_data[:, 0] = flux_data[:, 0]*base_energy_in_TeV
    flux_data[:, 0] = np.log10(flux_data[:, 0])

    # interpolate the data points, every energy outside definition range
    # from the data file is assumed to have 0 effective area
    flux_data_interpol = interpolate.interp1d(
        flux_data[:, 0],
        flux_data[:, 1],
        bounds_error=False,
        fill_value=0.
    )

    return flux_data_interpol
