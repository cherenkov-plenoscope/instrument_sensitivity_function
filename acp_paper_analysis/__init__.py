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
            'magic': acp.__path__[0] + '/resources/MAGIC_lowZd_Aeff.dat'
        },
        'isez': {
            'fermi_lat': acp.__path__[0] + '/resources/' +
            'FermiLAT_isez_p8r2_source_v6_10yr_gal_north.txt'
        },
        'crab': {
            'broad_sed': acp.__path__[0] +
            '/resources/crab_nebula_sed_fermi_magic.dat'
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
    b_energy_x_in_tev = base_energy_in_TeV
    b_energy_y_in_tev = base_energy_in_TeV

    return get_spectrum_from_linear_file(
        file_path,
        b_energy_x_in_tev=b_energy_x_in_tev,
        b_energy_y_in_tev=b_energy_y_in_tev,
        base_area_in_cm_2=base_area_in_cm_2,
        base_time_in_sec=base_time_in_sec,
        plot_power_slope=plot_power_slope
        )


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

    return get_spectrum_from_linear_file(
        file_path,
        b_energy_x_in_tev=b_energy_x_in_tev,
        b_energy_y_in_tev=b_energy_y_in_tev,
        base_area_in_cm_2=base_area_in_cm_2,
        base_time_in_sec=base_time_in_sec,
        plot_power_slope=plot_power_slope
        )


def get_crab_spectrum(file_path):
    '''
    This reads in a file containing the Crab nebula SED

    Transforms it into an interpolated spectrum,
    in usual coordinates: TeV, cm^2, s
    '''
    b_energy_x_in_tev = 1e-3  # was given in GeV
    b_energy_y_in_tev = 1.  # was given in TeV
    base_area_in_cm_2 = 1.
    base_time_in_sec = 1.
    plot_power_slope = 2.

    return get_spectrum_from_linear_file(
        file_path,
        b_energy_x_in_tev=b_energy_x_in_tev,
        b_energy_y_in_tev=b_energy_y_in_tev,
        base_area_in_cm_2=base_area_in_cm_2,
        base_time_in_sec=base_time_in_sec,
        plot_power_slope=plot_power_slope
        )


def get_spectrum_from_linear_file(
        file_path,
        b_energy_x_in_tev=1.,
        b_energy_y_in_tev=1.,
        base_area_in_cm_2=1.,
        base_time_in_sec=1.,
        plot_power_slope=0.
        ):
    '''
    Method to read any file containing data
    first column is energy (not logarithmic)
    and the second column some sort of flux,
    also linear.
    '''
    if plot_power_slope < 0:
        raise ValueError('plot_power_slope is supposed to be positive. ' +
                         'It is the power of energy by which the flux points' +
                         ' were multiplied'
                         )

    data = np.loadtxt(file_path)

    # convert energy axis to energy used on y axis
    data[:, 0] = data[:, 0]*b_energy_x_in_tev
    data[:, 0] = data[:, 0]/b_energy_y_in_tev

    # convert to normal flux, then convert to cm^2, convert to per TeV
    data[:, 1] = data[:, 1]/(data[:, 0]**plot_power_slope)
    data[:, 1] = data[:, 1]/base_area_in_cm_2
    data[:, 1] = data[:, 1]/b_energy_y_in_tev
    data[:, 1] = data[:, 1]/base_time_in_sec

    # convert energy axis to TeV scale, then take log10
    data[:, 0] = data[:, 0]*b_energy_y_in_tev
    data[:, 0] = np.log10(data[:, 0])

    # interpolate the data points, every energy outside definition range
    # from the data file is assumed to be 0
    data_interpol = interpolate.interp1d(
        data[:, 0],
        data[:, 1],
        bounds_error=False,
        fill_value=0.
    )

    return data_interpol
