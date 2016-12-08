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
    resource_paths_dict = {
        'fluxes': {
            'electron_positron': acp.__path__[0] +
            '/resources/e_plus_e_minus_spec.dat',
            'proton': acp.__path__[0]+'/resources/proton_spec.dat',
        },
        'Aeff': {
            'magic': acp.__path__[0] + '/resources/MAGIC_lowZd_Aeff.dat',
            'fermi_lat': acp.__path__[0] +
            '/resources/FermiLAT_P8R2_OnAxis_Total_Aeff.dat',
        },
        'isez': {
            'fermi_lat': acp.__path__[0] + '/resources/' +
            'FermiLAT_isez_p8r2_source_v6_10yr_gal_north.txt'
        }
    }
    return resource_paths_dict


def get_cosmic_ray_flux_interpol(
        file_path,
        base_energy_in_TeV,
        plot_power_slope,
        base_area_in_cm_2,
        base_time_in_sec=1.
        ):
    '''
    Function to get the interpolated cr fluxes
    from a file path

    my own units are: log10(E/TeV), sec, sr, cm^2.
    AMS02 flues are usually given in E/GeV, m^2, sec, sr
    '''
    # prevent errors from happening
    if plot_power_slope < 0:
        raise ValueError('plot_power_slope is supposed to be positive. ' +
                         'It is the power of energy by which the flux points' +
                         ' were multiplied'
                         )

    flux_data = np.loadtxt(file_path)

    # first, convert to normal flux, then convert to cm^2, convert to per TeV
    flux_data[:, 1] = flux_data[:, 1]/(flux_data[:, 0]**plot_power_slope)
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


def get_fermi_lat_isez(file_path):
    '''
    This reads in a file containing the isez from
    FermiLAT (they call it broadband sensitivity)

    Transforms it into an interpolated function,
    in usual coordinates: TeV, cm^2, s
    '''
    b_energy_x_in_tev = 1e-6  # was given in MeV
    b_energy_y_in_tev = 0.62415091  # was given in erg
    base_area_in_cm_2 = 1.
    base_time_in_sec = 1.
    plot_power_slope = 2.

    isez_data = np.loadtxt(file_path)

    # convert energy axis to energy used on y axis
    isez_data[:, 0] = isez_data[:, 0]*b_energy_x_in_tev  # -> TeV
    isez_data[:, 0] = isez_data[:, 0]/b_energy_y_in_tev  # -> erg

    # convert to normal flux, then convert to cm^2, convert to per TeV
    isez_data[:, 1] = isez_data[:, 1]/(isez_data[:, 0]**plot_power_slope)  # -> 1/(erg cm^2 s)
    isez_data[:, 1] = isez_data[:, 1]/base_area_in_cm_2
    isez_data[:, 1] = isez_data[:, 1]/b_energy_y_in_tev  # -> 1/(TeV cm^2 s)
    isez_data[:, 1] = isez_data[:, 1]/base_time_in_sec

    # convert energy axis to TeV scale, then take log10
    isez_data[:, 0] = isez_data[:, 0]*b_energy_y_in_tev  # -> TeV
    isez_data[:, 0] = np.log10(isez_data[:, 0])
    
    # interpolate the data points, every energy outside definition range
    # from the data file is assumed to be 0
    isez_data_interpol = interpolate.interp1d(
        isez_data[:, 0],
        isez_data[:, 1],
        bounds_error=False,
        fill_value=0.
    )

    return isez_data_interpol
