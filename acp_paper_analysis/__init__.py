'''
This is the hard working code in order to create publication plots
'''
# import gamma_limits_sensitivity as gls
#
# import matplotlib.pyplot as plt
import os
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate

import acp_paper_analysis as acp
import gamma_limits_sensitivity as gls


def analysis(
        in_folder,
        is_test=False
        ):
    '''
    This method contains the main logic behind the analysis.
    The options passed in 'arguments' include inpath and potentially
    an outpath. If 'is_test' is set, make the plotting run on
    lower resolution in order to speed things up.
    '''
    effective_area_dict = get_interpolated_effective_areas(in_folder)
    effective_area_figure = get_effective_area_figure(effective_area_dict)

    one_data = np.array([0.])

    figures = {
        'effective_area_figure': effective_area_figure
        }

    data = {
        'one_data': one_data
    }

    dictionary = {
        'plots': figures,
        'data': data
        }

    return dictionary


def get_interpolated_effective_areas(in_folder):
    aeff_file_paths = acp.generate_absolute_filepaths(in_folder)

    effective_areas_dict = {
        'gamma': {
            'trigger': gls.get_effective_area(
                aeff_file_paths['gamma']
                ),
            'cut': gls.get_effective_area(
                aeff_file_paths['gamma_cut']
                )
        },
        'electron_positron': {
            'trigger': gls.get_effective_area(
                aeff_file_paths['electron_positron']
                ),
            'cut': gls.get_effective_area(
                aeff_file_paths['electron_positron_cut']
                )
        },
        'proton': {
            'trigger': gls.get_effective_area(
                aeff_file_paths['proton']
                ),
            'cut': gls.get_effective_area(
                aeff_file_paths['proton_cut']
                )
        },

    }

    return effective_areas_dict


def generate_absolute_filepaths(in_folder):
    '''
    This function looks into the provided in folder
    and scans for the six necessary effective areas.
    '''
    aeff_dict = {
        'gamma': in_folder + '/gamma_aeff.dat',
        'gamma_cut': in_folder + '/gamma_cut_aeff.dat',
        'electron_positron': in_folder + '/electron_positron_aeff.dat',
        'electron_positron_cut': in_folder +
        '/electron_positron_cut_aeff.dat',
        'proton': in_folder + '/proton_aeff.dat',
        'proton_cut': in_folder + '/proton_cut_aeff.dat'
        }

    for aeff_name in aeff_dict:
        if os.path.isfile(aeff_dict[aeff_name]) is False:
            raise ValueError(
                aeff_dict[aeff_name] +
                ' was not found. please' +
                ' provide a correct path to the effective area for ' +
                aeff_name
                )
    return aeff_dict


def get_resources_paths():
    '''
    This function returns a dict including absolute paths
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
        },
        'fermi_lat': {
            '3fgl': acp.__path__[0] +
            '/resources/FermiLAT_3FGL_gll_psc_v16.fit'
        }
    }
    return resource_paths_dict


def get_3fgl_catalog(file_path):
    '''
    Function to get the relevant information from the
    3FGL FITS file. These are:

    source name
    Ra / deg
    Dec / deg
    GalLong / deg
    GalLat / deg
    Spectrum_Type (in the range [100MeV .. 100GeV])
    Pivot_Energy / MeV
    Spectral_Index
    Flux_Density / 1/(cm^2 s MeV)

    and transform them into my standard units: TeV, cm^2, s
    '''
    hdu_list = pyfits.open(file_path)

    # make a list of dcts for each source and return it
    name_index = 0
    ra_index = 1
    dec_index = 2
    gal_long_index = 3
    gal_lat_index = 4
    spec_type_index = 21
    pivot_energy_index = 13
    spectral_index_index = 22
    flux_density_index = 14

    source_dict_list = []
    for source in hdu_list[1].data:
        source_dict = {
            'name': source[name_index],
            'ra': source[ra_index],
            'dec': source[dec_index],
            'gal_long': source[gal_long_index],
            'gal_lat': source[gal_lat_index],
            'spec_type': source[spec_type_index],
            'pivot_energy': source[pivot_energy_index]*1e-6,
            'spectral_index': -1*source[spectral_index_index],
            'flux_density': source[flux_density_index]*1e6
        }
        source_dict_list.append(source_dict)

    return source_dict_list


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


def time_to_detection(
        f_0,
        gamma,
        e_0,
        a_eff_interpol,
        sigma_bg,
        alpha,
        threshold=5.
        ):
    '''
    This function calls gls functions in order to calculate time to detections
    '''
    return gls.t_obs_li_ma_criterion(
        f_0 * gls.effective_area_averaged_flux(
            gamma,
            e_0,
            a_eff_interpol
            ),
        sigma_bg,
        alpha,
        threshold
        )


def get_effective_area_figure(
        effective_area_dict
        ):
    '''
    Get a plot showing the effective areas
    referenced by a_eff_interpol
    '''
    figure = plt.figure()

    colors = ['k', 'c', 'm', 'b', 'r', 'g']
    linestyles = ['-', '--', '-.', ':']
    last_cut = ''

    for i, particle in enumerate(effective_area_dict):
        for j, cut in enumerate(effective_area_dict[particle]):
            label = particle+' '+cut
            style = colors[j]+linestyles[i]

            gls.plot_effective_area(
                effective_area_dict[particle][cut],
                style=style,
                label=label)

    return figure
