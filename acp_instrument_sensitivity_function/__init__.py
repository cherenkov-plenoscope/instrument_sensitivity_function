'''
This is the hard working code in order to create publication plots
'''
# import gamma_limits_sensitivity as gls
#
# import matplotlib.pyplot as plt
import os
from os.path import join
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interpolate
from scipy import integrate
from astropy.table import Table
import acp_instrument_sensitivity_function as acp
import gamma_limits_sensitivity as gls
from pkg_resources import resource_filename


def analysis(
        gamma_collection_area_path,
        electron_collection_acceptance_path, 
        proton_collection_acceptance_path,
        rigidity_cutoff_in_tev=10e-3,
        relative_flux_below_cutoff=0.1,
        fov_in_deg=6.5,
        source='3FGL J2254.0+1608',
        gamma_eff=0.67,
        is_test=False,
        plot_isez_all=False,
        out_path=None
        ):
    '''
    This method contains the main logic behind the analysis.
    The options passed in 'arguments' include inpath and potentially
    an outpath. If 'is_test' is set, make the plotting run on
    lower resolution in order to speed things up.
    '''
    # prepare the data
    effective_area_dict = {
        'gamma': gls.get_effective_area(gamma_collection_area_path),
        'electron_positron': gls.get_effective_area(electron_collection_acceptance_path),
        'proton': gls.get_effective_area(proton_collection_acceptance_path)
    }
    resource_dict = get_resources_paths()

    fermi_lat_3fgl_catalog = acp.get_3fgl_catalog(
        resource_dict['fermi_lat']['3fgl']
        )
    gamma_spec = get_gamma_spect(fermi_lat_3fgl_catalog, source=source)

    electron_positron_flux = get_cosmic_ray_flux_interpol(
        resource_dict['fluxes']['electron_positron'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=3.,
        base_area_in_cm_2=1e4
        )
    proton_spec = get_cosmic_ray_flux_interpol(
        resource_dict['fluxes']['proton'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=2.7,
        base_area_in_cm_2=1e4
        )

    # start producing plots and data products
    gamma_effective_area_figure = get_gamma_effective_area_figure(
        effective_area_dict)

    charged_acceptance_figure = get_charged_acceptance_figure(
        effective_area_dict)

    rates_figure, rates_data = get_rates_over_energy_figure(
        effective_area_dict,
        proton_spec=proton_spec,
        electron_positron_spec=electron_positron_flux,
        gamma_spec=gamma_spec,
        source=source,
        rigidity_cutoff_in_tev=rigidity_cutoff_in_tev,
        relative_flux_below_cutoff=relative_flux_below_cutoff,
        fov_in_deg=fov_in_deg,
        gamma_eff=gamma_eff
        )

    # get the integral bg rate in on region (roi region)
    acp_sigma_bg = (
        rates_data['electron_positron_roi_rate'] +
        rates_data['proton_roi_rate']
        )
    acp_alpha = 1./3.

    # make a coparison of the Fermi-LAT, MAGIC,
    # and ACP integral spectral exclusion zone
    plotting_energy_range = [0.1e-3, 10.]  # in TeV
    # get efficiency scaled acp aeff
    acp_aeff_scaled = get_interpol_func_scaled(
        effective_area_dict['gamma'],
        gamma_eff=gamma_eff)

    isez_figure, isez_data = get_isez_figure(
        resource_dict,
        acp_aeff=acp_aeff_scaled,
        acp_sigma_bg=acp_sigma_bg,
        energy_range=plotting_energy_range,
        is_test=is_test,
        plot_isez_all=plot_isez_all
        )

    grb_130427A_dict = {
        'grb_130427A_seconds_to_det': np.array([acp.time_to_detection(
            f_0=1.e-7,
            gamma=-2.,
            e_0=1.,
            a_eff_interpol=effective_area_dict['gamma'],
            sigma_bg=acp_sigma_bg,
            alpha=acp_alpha)]),

        'grb_130427A_gamma_rate': np.array([
            1.e-7*gls.effective_area_averaged_flux(
                gamma=-2.,
                e_0=1.,
                a_eff_interpol=effective_area_dict['gamma'])
            ])
        }

    sorted_times_to_detection_map, reduced_catalog = acp.get_time_to_detections(
        fermi_lat_3fgl_catalog,
        a_eff=effective_area_dict['gamma'],
        sigma_bg=acp_sigma_bg,
        alpha=acp_alpha,
        out_path=out_path)

    t_est_histogram, t_est_data = acp.get_t_est_histogram(
        np.array(sorted_times_to_detection_map)[:, 1])

    figures = {
        'gamma_effective_area_figure': gamma_effective_area_figure,
        'charged_acceptance_figure': charged_acceptance_figure,
        'rates_figure': rates_figure,
        'isez_figure': isez_figure,
        't_est_figure': t_est_histogram
        }
    data = merge_dicts(rates_data, isez_data, t_est_data, grb_130427A_dict)

    dictionary = {
        'plots': figures,
        'data': data
        }

    return dictionary


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def get_resources_paths():
    '''
    This function returns a dict including absolute paths
    to resource files with fluxes (protons, leptons)
    '''
    resource_paths_dict = {
        'fluxes': {
            'electron_positron': resource_filename(
                'acp_instrument_sensitivity_function', 
                'resources/e_plus_e_minus_spec.dat'),
            'proton': resource_filename(
                'acp_instrument_sensitivity_function', 
                'resources/proton_spec.dat'),
        },
        'Aeff': {
            'magic': resource_filename(
                'acp_instrument_sensitivity_function', 
                'resources/MAGIC_lowZd_Aeff.dat')
        },
        'isez': {
            'fermi_lat': resource_filename(
                'acp_instrument_sensitivity_function', 
                'resources/FermiLAT_isez_p8r2_source_v6_10yr_gal_north.txt')
        },
        'crab': {
            'broad_sed': resource_filename(
                'acp_instrument_sensitivity_function', 
                'resources/crab_nebula_sed_fermi_magic.dat')
        },
        'fermi_lat': {
            '3fgl': resource_filename(
                'acp_instrument_sensitivity_function', 
                'resources/FermiLAT_3FGL_gll_psc_v16.fit')
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
    name_map = {
        'Source_Name': 'name',
        'RAJ2000': 'ra',
        'DEJ2000': 'dec',
        'GLON': 'gal_long',
        'GLAT': 'gal_lat',
        'SpectrumType': 'spec_type',
        'Pivot_Energy': 'pivot_energy',
        'Spectral_Index': 'spectral_index',
        'Flux_Density': 'flux_density',
        'beta': 'beta',
        'Cutoff': 'cutoff',
        'Exp_Index': 'exp_index',
    }

    t = Table.read(file_path)
    t.remove_columns(set(t.dtype.names).difference(name_map.keys()))
    df = t.to_pandas()
    df.rename(columns=name_map, inplace=True)

    df.spectral_index *= -1
    df.beta *= -1
    df.pivot_energy *= 1e-6
    df.flux_density *= 1e6
    df.cutoff *= 1e-6

    df['spec_type'] = df.spec_type.str.strip()
    df['name'] = df.name.str.strip()

    return list(df.T.to_dict().values())

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
        beta=False,
        cutoff=False,
        exp_index=False,
        spec_type='PowerLaw',
        threshold=5.
        ):
    '''
    This function calls gls functions in order to calculate time to detections.
    spectrum types:

    PowerLaw
    LogParabola
    PLExpCutoff / PLSuperExpCutoff
    '''
    lambda_lim_val = lambda_lim(
        f_0=f_0,
        gamma=gamma,
        e_0=e_0,
        a_eff_interpol=a_eff_interpol,
        beta=beta,
        cutoff=cutoff,
        exp_index=exp_index,
        spec_type=spec_type)

    return gls.t_obs_li_ma_criterion(
        lambda_lim_val,
        sigma_bg,
        alpha,
        threshold
        )


def lambda_lim(
        f_0,
        gamma,
        e_0,
        a_eff_interpol,
        beta,
        cutoff,
        exp_index,
        spec_type):
    '''
    calculate the expected number of events, given a spectrum
    '''
    energy_range = gls.get_energy_range(a_eff_interpol)

    if spec_type == 'PowerLaw':
        return f_0 * gls.effective_area_averaged_flux(
            gamma,
            e_0,
            a_eff_interpol
            )

    elif spec_type == 'LogParabola':
        integrand = lambda x: log_parabola_3fgl(
            x,
            f_0=f_0,
            alpha=gamma,
            e_0=e_0,
            beta=beta
            )*a_eff_interpol(np.log10(x))

    elif spec_type == 'PLExpCutoff' or spec_type == 'PLSuperExpCutoff':
        integrand = lambda x: pl_super_exp_cutoff_3fgl(
            x,
            f_0=f_0,
            gamma=gamma,
            e_0=e_0,
            cutoff=cutoff,
            exp_index=exp_index
            )*a_eff_interpol(np.log10(x))

    return integrate.quad(
        integrand,
        energy_range[0],
        energy_range[1],
        limit=10000,
        full_output=1,
        points=[energy_range[0], energy_range[0]*10]
        )[0]


def log_parabola_3fgl(energy, f_0, alpha, e_0, beta):
    '''
    log parabola as defined in 3fgl cat
    but with already negative alpha and beta
    '''
    return f_0*(energy/e_0)**(+alpha+beta*np.log10(energy/e_0))


def pl_super_exp_cutoff_3fgl(energy, f_0, gamma, e_0, cutoff, exp_index):
    '''
    pl super exponential cutoff as defined in 3FGL cat,
    but with already negative gamma
    '''
    return f_0*(energy/e_0)**(gamma)*np.exp(
        (e_0/cutoff)**exp_index - (energy/cutoff)**exp_index
        )


def solid_angle_of_cone(apex_angle_in_deg):
    '''
    WIKI:
    solid angle of cone with apex angle 2phi =
    area of a spherical cap on a unit sphere

    input: deg
    returns: steradian
    '''
    return 2*np.pi*(1-np.cos(apex_angle_in_deg/180.*np.pi))


def rigidity_to_energy(rigidity, charge, mass):
    '''
    Transform rigidity -> energy
    input units:

    rigidity / teravolt (TV)
    unit charge [1, 2, ..]
    mass / (TeV/c^2)

    output:
    TeV
    '''
    return np.sqrt((rigidity*charge)**2 + mass**2) - mass


def linestyle(particle):
    buf_dict = {
        'gamma': 'k-',
        'electron_positron': 'k--',
        'proton': 'k:'
    }
    return buf_dict[particle]


def get_charged_acceptance_figure(
        effective_area_dict
        ):
    figure = plt.figure()

    for particle in effective_area_dict:
        if 'electron' in particle or 'proton' in particle:
            plot_effective_area(
                effective_area_dict[particle],
                style=linestyle(particle),
                label=particle,
                diffuse=True
                )

    plt.title('Instrument Acceptance')
    plt.legend(loc='best', fontsize=10)
    return figure


def get_gamma_effective_area_figure(
        effective_area_dict
        ):
    figure = plt.figure()

    for particle in effective_area_dict:
        if 'gamma' in particle:
            plot_effective_area(
                effective_area_dict[particle],
                style=linestyle(particle),
                label=particle,
                diffuse=False
                )

    plt.title('Effective Area')
    plt.legend(loc='best', fontsize=10)
    return figure


def plot_effective_area(
        a_eff_interpol, style='k', label='', diffuse=False):
    '''
    fill a plot with the effective energy from the supplied
    interpolated data
    '''
    start = a_eff_interpol.x.min()
    stop = a_eff_interpol.x.max()
    samples = 1000

    energy_samples = np.linspace(start, stop, samples)
    area_samples = np.array([
        a_eff_interpol(energy)
        for energy
        in energy_samples
        ])

    plt.plot(np.power(10, energy_samples), area_samples/10000.,
             style,
             label=label)

    plt.loglog()

    plt.xlabel('Energy / TeV')
    plt.ylabel('A$_{eff}$ / m$^2$')
    if diffuse:
        plt.ylabel('Acceptance / (m$^2$ sr)')

    return


def get_rates_over_energy_figure(
        effective_area_dict,
        proton_spec,
        electron_positron_spec,
        gamma_spec,
        source,
        rigidity_cutoff_in_tev=10e-3,
        relative_flux_below_cutoff=0.1,
        fov_in_deg=6.5,
        gamma_eff=1.0
        ):
    '''
    Get a plot with rates over energy for the stated:
    spectra, aeff, rigidity cutoff, relative second spectrum intensity,
    roi radius (half apex angle of the viewcone), and gamma ray source
    parameters.
    '''
    # prepare constants
    e_mass = 0.511e-6  # in TeV/c^2
    p_mass = 0.938272e-3
    e_energy_cutoff = rigidity_to_energy(
        rigidity_cutoff_in_tev,
        charge=1,
        mass=e_mass)
    p_energy_cutoff = rigidity_to_energy(
        rigidity_cutoff_in_tev,
        charge=1,
        mass=p_mass)

    figure = plt.figure()
    rates_data = {}

    # Electron Positron
    ep_plot_data, ep_roi_rate = plot_rate_over_energy_charged_diffuse(
        effective_area_dict['electron_positron'],
        style=linestyle('electron_positron'),
        label='electron_positron',
        charged_spec=electron_positron_spec,
        cutoff=e_energy_cutoff,
        relative_flux_below_cutoff=relative_flux_below_cutoff,
        fov_in_deg=fov_in_deg)
    ep_fov_rate = get_rate_charged_diffuse(
        effective_area_dict['electron_positron'],
        charged_spec=electron_positron_spec,
        cutoff=e_energy_cutoff,
        relative_flux_below_cutoff=relative_flux_below_cutoff,
        roi_radius_in_deg=fov_in_deg/2.,
        fov_in_deg=fov_in_deg)
    rates_data['electron_positron'+'_rate_plot_data'] = ep_plot_data
    rates_data['electron_positron'+'_roi_rate'] = ep_roi_rate
    rates_data['electron_positron'+'_fov_rate'] = ep_fov_rate

    # Proton
    p_plot_data, p_roi_rate = plot_rate_over_energy_charged_diffuse(
        effective_area_dict['proton'],
        style=linestyle('proton'),
        label='proton',
        charged_spec=proton_spec,
        cutoff=p_energy_cutoff,
        relative_flux_below_cutoff=relative_flux_below_cutoff,
        fov_in_deg=fov_in_deg)
    p_fov_rate = get_rate_charged_diffuse(
        effective_area_dict['proton'],
        charged_spec=proton_spec,
        cutoff=e_energy_cutoff,
        relative_flux_below_cutoff=relative_flux_below_cutoff,
        roi_radius_in_deg=fov_in_deg/2.,
        fov_in_deg=fov_in_deg)
    rates_data['proton'+'_rate_plot_data'] = p_plot_data
    rates_data['proton'+'_roi_rate'] = p_roi_rate
    rates_data['proton'+'_fov_rate'] = p_fov_rate

    # Gamma
    g_plot_data, g_roi_rate = plot_rate_over_energy_power_law_source(
        effective_area_dict['gamma'],
        gamma_spec=gamma_spec,
        style=linestyle('gamma'),
        label=source,
        efficiency=gamma_eff)
    rates_data['gamma'+'_rate_plot_data'] = g_plot_data
    rates_data['gamma'+'_roi_rate'] = g_roi_rate

    plt.title('Diff. Rate')
    plt.legend(loc='best', fontsize=10)
    return figure, rates_data


def cutoff_spec(
        charged_spec,
        cutoff,
        relative_flux_below_cutoff
        ):
    '''
    this is a function in order to check the cutoff
    is implemented

    cutoff / TeV
    '''
    return lambda x: (
        charged_spec(x) *
        (0.5*(np.sign(10**x-cutoff)+1) *
            (1-relative_flux_below_cutoff) +
            relative_flux_below_cutoff)     # heaviside function
        )


def psf_electromagnetic_in_deg(energy_in_tev):
    '''
    This function returns the half angle / deg of the
    psf cone which contains 0.67% of gamma events
    according to Aharonian et al. 5@5 paper
    '''
    return 0.8*(energy_in_tev*1000.)**(-0.4)

def plot_over_energy_log_log(
        function,
        energy_range,
        scale_factor=1.,
        style='k',
        label='',
        alpha=1.,
        ylabel='dN/dE / [(cm$^2$ s TeV)$^{-1}$]',
        log_resolution=0.05):
    '''
    This function plots anything (in log log) over energy

    The function input is expected to get energies as

        function( log10(E/TeV) )
    '''
    e_x = 10**np.arange(
        np.log10(energy_range[0]),
        np.log10(energy_range[1])+0.05,
        log_resolution)

    e_y = np.array([function(np.log10(x)) for x in e_x])
    e_y = e_y*scale_factor

    plt.plot(e_x, e_y, style, label=label, alpha=alpha)
    plt.loglog()

    plt.xlabel("E / TeV")
    plt.ylabel(ylabel)

    return e_x, e_y


def plot_rate_over_energy_charged_diffuse(
        effective_area,
        style,
        label,
        charged_spec,
        cutoff,
        relative_flux_below_cutoff,
        fov_in_deg
        ):

    energy_range = gls.get_energy_range(effective_area)

    solid_angle_ratio = lambda x: (
        acp.solid_angle_of_cone(
            psf_electromagnetic_in_deg(x)
            ) /
        acp.solid_angle_of_cone(fov_in_deg/2.)
        )

    charged_spec_cutoff = cutoff_spec(
        charged_spec, cutoff, relative_flux_below_cutoff)

    diff_rate = lambda x: (
        charged_spec_cutoff(x) *
        effective_area(x) *
        solid_angle_ratio(10**x))

    plot_data_x, plot_data_y = plot_over_energy_log_log(
        diff_rate,
        energy_range=energy_range,
        style=style,
        label=label,
        ylabel='Rate / (s TeV)$^{-1}$'
        )

    rate = get_rate_charged_diffuse(
        effective_area=effective_area,
        charged_spec=charged_spec,
        cutoff=cutoff,
        relative_flux_below_cutoff=relative_flux_below_cutoff,
        fov_in_deg=fov_in_deg
        )

    plot_data = np.vstack((plot_data_x, plot_data_y)).T
    return plot_data, rate


def get_rate_charged_diffuse(
        effective_area,
        charged_spec,
        cutoff,
        relative_flux_below_cutoff,
        fov_in_deg,
        roi_radius_in_deg=None,
        ):

    energy_range = gls.get_energy_range(effective_area)

    charged_spec_cutoff = cutoff_spec(
        charged_spec, cutoff, relative_flux_below_cutoff)

    if roi_radius_in_deg is not None:
        solid_angle_ratio = (
            acp.solid_angle_of_cone(roi_radius_in_deg) /
            acp.solid_angle_of_cone(fov_in_deg/2.)
            )

        integrand = lambda x: (
            charged_spec_cutoff(np.log10(x)) *
            effective_area(np.log10(x)) *
            solid_angle_ratio)
    else:
        solid_angle_ratio = lambda x: (
            acp.solid_angle_of_cone(
                psf_electromagnetic_in_deg(x)
                ) /
            acp.solid_angle_of_cone(fov_in_deg/2.)
            )

        integrand = lambda x: (
            charged_spec_cutoff(np.log10(x)) *
            effective_area(np.log10(x)) *
            solid_angle_ratio(x))

    points_to_watch_out = [energy_range[0], energy_range[0]*10]
    if cutoff > energy_range[0] and cutoff < energy_range[1]:
        points_to_watch_out.append(cutoff)

    rate = np.array([integrate.quad(
        integrand,
        energy_range[0],
        energy_range[1],
        limit=10000,
        full_output=1,
        points=points_to_watch_out
        )[0]])

    return rate


def plot_rate_over_energy_power_law_source(
        effective_area,
        gamma_spec,
        style,
        label,
        efficiency
        ):
    energy_range = gls.get_energy_range(effective_area)
    energy_range_plot = energy_range.copy()
    energy_range_plot[1] = 0.1  # make the plot go only until 100GeV

    diff_rate = lambda x: (
        # gls.power_law(10**x, f_0=f_0, gamma=gamma, e_0=e_0) *
        gamma_spec(x) *
        effective_area(x)*efficiency)

    integrand = lambda x: (
        # gls.power_law(x, f_0=f_0, gamma=gamma, e_0=e_0) *
        gamma_spec(np.log10(x)) *
        effective_area(np.log10(x))*efficiency)

    plot_data_x, plot_data_y = plot_over_energy_log_log(
        diff_rate,
        energy_range=energy_range_plot,
        style=style,
        label=label,
        ylabel='Rate / (s TeV)$^{-1}$')

    points_to_watch_out = [energy_range[0], energy_range[0]*10]

    rate = np.array([integrate.quad(
        integrand,
        energy_range[0],
        energy_range[1],
        limit=10000,
        full_output=1,
        points=points_to_watch_out
        )[0]])

    plot_data = np.vstack((plot_data_x, plot_data_y)).T
    return plot_data, rate


def get_isez_figure(
        resource_dict,
        acp_sigma_bg,
        energy_range,
        acp_aeff,
        acp_alpha=1./3.,
        t_obs=50.*3600.,
        is_test=False,
        plot_isez_all=False
        ):
    '''
    This function shall return a set of isze curves, in a figure and as data
    Furthermore it shall compare everything to te crab nebula emission.
    '''
    crab_broad_spectrum = get_crab_spectrum(resource_dict['crab']['broad_sed'])

    # get magic sensitivity parameters as stated in ul paper
    magic_aeff = gls.get_effective_area(resource_dict['Aeff']['magic'])
    magic_sigma_bg = 0.0020472222222222224  # bg per second in the on region
    magic_alpha = 0.2  # five off regions
    n_points_to_plot = 21
    if is_test:
        n_points_to_plot = 1
    magic_energy_range = gls.get_energy_range(magic_aeff)

    fermi_lat_isez = acp.get_fermi_lat_isez(resource_dict['isez']['fermi_lat'])

    figure = plt.figure()
    data = np.array([1.])

    for i in range(4):
        plot_over_energy_log_log(
            crab_broad_spectrum,
            energy_range,
            scale_factor=np.power(10., (-1)*i),
            style='k--',
            label='%.3f C.U.' % np.power(10., (-1)*i),
            alpha=1./(1.+i),
            ylabel='dN/dE / [(cm$^2$ s TeV)$^{-1}$]',
            log_resolution=0.2)

    plot_over_energy_log_log(
        fermi_lat_isez,
        gls.get_energy_range(fermi_lat_isez),
        style='k',
        label='Fermi-LAT 10y gal. north',
        ylabel='dN/dE / (cm$^2$ s TeV)$^{-1}$',
        log_resolution=0.05)

    # magic_energy_x, magic_dn_de_y = gls.plot_sens_spectrum_figure(
    gls.plot_sens_spectrum_figure(
        sigma_bg=magic_sigma_bg,
        alpha=magic_alpha,
        t_obs=t_obs,
        a_eff_interpol=magic_aeff,
        e_0=magic_energy_range[0]*5.,
        n_points_to_plot=n_points_to_plot,
        fmt='b',
        label='MAGIC %2.1fh'%(t_obs/3600.)
        )

    # plot the acp sensitivity
    acp_energy_range = gls.get_energy_range(acp_aeff)

    energy_x, dn_de_y = gls.plot_sens_spectrum_figure(
        sigma_bg=acp_sigma_bg,
        alpha=acp_alpha,
        t_obs=t_obs,
        a_eff_interpol=acp_aeff,
        e_0=acp_energy_range[0]*5.,
        n_points_to_plot=n_points_to_plot,
        fmt='r',
        label='ACP %2.1fh' % (t_obs/3600.)
        )

    if plot_isez_all:
        gls.plot_sens_spectrum_figure(
            sigma_bg=acp_sigma_bg,
            alpha=acp_alpha,
            t_obs=3600,
            a_eff_interpol=acp_aeff,
            e_0=acp_energy_range[0]*5.,
            n_points_to_plot=n_points_to_plot,
            fmt='r--',
            label='ACP 1h'
            )

        gls.plot_sens_spectrum_figure(
            sigma_bg=acp_sigma_bg,
            alpha=acp_alpha,
            t_obs=60,
            a_eff_interpol=acp_aeff,
            e_0=acp_energy_range[0]*5.,
            n_points_to_plot=n_points_to_plot,
            fmt='r:',
            label='ACP 1min'
            )

        gls.plot_sens_spectrum_figure(
            sigma_bg=acp_sigma_bg,
            alpha=acp_alpha,
            t_obs=3600*1000,
            a_eff_interpol=acp_aeff,
            e_0=acp_energy_range[0]*5.,
            n_points_to_plot=n_points_to_plot,
            fmt='r-.',
            label='ACP 1y (1000h)'
            )

        gls.plot_sens_spectrum_figure(
            sigma_bg=acp_sigma_bg,
            alpha=acp_alpha,
            t_obs=1,
            a_eff_interpol=acp_aeff,
            e_0=acp_energy_range[0]*5.,
            n_points_to_plot=n_points_to_plot,
            fmt='c-',
            label='ACP 1s'
            )

    plt.title('Integral Spectral Exclusion Zones')
    plt.xlim(energy_range)
    plt.legend(loc='best', fontsize=10)

    plot_data = np.vstack((energy_x, dn_de_y)).T
    plot_data_dict = {
        'acp_isez_plot_data': plot_data
    }
    return figure, plot_data_dict


def get_interpol_func_scaled(func, gamma_eff):
    '''
    Function to return efficiency scaled
    effective area of the instrument
    '''
    x_s = np.array(func.x)
    y_s = np.array([func(x)*gamma_eff for x in x_s])

    scaled_interpol = interpolate.interp1d(
        x_s,
        y_s,
        bounds_error=False,
        fill_value=0.
    )

    return scaled_interpol


def get_time_to_detections(
        fermi_lat_3fgl_catalog,
        a_eff,
        sigma_bg,
        alpha,
        out_path=None,
        is_test=False
        ):
    '''
    This function maps methods for getting time
    to detections onto the 3FGL and returns a sorted list
    of times to detection and indices where to find them
    in the 3fgl
    '''

    # 'name': source[name_index],
    # 'ra': source[ra_index],
    # 'dec': source[dec_index],
    # 'gal_long': source[gal_long_index],
    # 'gal_lat': source[gal_lat_index],
    # 'spec_type': source[spec_type_index],
    # 'pivot_energy': source[pivot_energy_index]*1e-6,
    # 'spectral_index': -1*source[spectral_index_index],
    # 'flux_density': source[flux_density_index]*1e6
    detection_times = []
    gal_lat_cut = 15  # only src with |gal lat| > 15
    total = len(fermi_lat_3fgl_catalog)


    for i, source in tqdm(enumerate(fermi_lat_3fgl_catalog), total=total):
        # check that it is pl and far off the gal. plane
        if np.abs(source['gal_lat']) > gal_lat_cut:
            e_0 = source['pivot_energy']
            f_0 = source['flux_density']
            gamma = source['spectral_index']
            beta = source['beta']
            cutoff = source['cutoff']
            exp_index = source['exp_index']
            time_to_det = acp.time_to_detection(
                f_0=f_0,
                gamma=gamma,
                e_0=e_0,
                a_eff_interpol=a_eff,
                sigma_bg=sigma_bg,
                alpha=alpha,
                beta=beta,
                cutoff=cutoff,
                exp_index=exp_index,
                spec_type=source['spec_type'],
                )

            list_buf = [i, time_to_det]
            detection_times.append(list_buf)
        if is_test and i > 10:
            break

    # sort on the times to detection
    detection_times.sort(key=lambda x: x[1])

    reduced_sorted_catalog = [
        [
            fermi_lat_3fgl_catalog[i[0]]['name'],
            fermi_lat_3fgl_catalog[i[0]]['ra'],
            fermi_lat_3fgl_catalog[i[0]]['dec'],
            fermi_lat_3fgl_catalog[i[0]]['gal_long'],
            fermi_lat_3fgl_catalog[i[0]]['gal_lat'],
            fermi_lat_3fgl_catalog[i[0]]['spec_type'],
            fermi_lat_3fgl_catalog[i[0]]['pivot_energy'],
            fermi_lat_3fgl_catalog[i[0]]['spectral_index'],
            fermi_lat_3fgl_catalog[i[0]]['flux_density'],
            i[1]
        ]
        for i in detection_times
        ]

    if out_path is not None:
        with open(join(out_path, 'time_to_detections.csv'), 'w') as myfile:
            writer = csv.writer(myfile)
            myfile.write(
                '# time_to_detections, written: ' +
                datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S\n"
                    )
                )
            myfile.write(
                '# name, ra, dec, gal_long, gal_lat, spec_type, ' +
                'pivot_energy, spectral_index, flux_density, time_est' +
                '\n'
                )
            for row in reduced_sorted_catalog:
                writer.writerow(row)

    return detection_times, reduced_sorted_catalog


def get_t_est_histogram(sorted_t_est_list, style='k', label=''):
    '''
    this function produces a figure
    containing the times to detection for the
    given list
    '''
    figure = plt.figure()

    data = plot_t_est_histogram(sorted_t_est_list, style, label)
    data_dict = {
        't_est_cdf': data
    }
    if label is not '':
        plt.legend(loc='best')

    return figure, data_dict


def plot_t_est_histogram(sorted_t_est_list, style, label):
    '''
    This function plots the t_est_histogram into the current figure
    '''
    yvals = np.arange(len(sorted_t_est_list))
    plt.step(sorted_t_est_list, yvals, style, label=label)
    plt.loglog()
    plt.xlabel('time / s')
    plt.ylabel('number of src')
    plt.title('Fermi-LAT 3FGL sources detected, |b|>15$^{\circ}$ ')

    plot_data = np.vstack((sorted_t_est_list, yvals)).T
    return plot_data


def get_gamma_spect(fermi_cat, source):
    '''
    This function produces a function for
    calculating the spectum of a named source
    in the fermi-lat 3fgl
    '''

    source_dict = get_gamma_dict(fermi_cat, source)

    if source_dict['spec_type'] == 'PowerLaw':
        return_func = lambda x: gls.power_law(
            energy=10**x,
            f_0=source_dict['f_0'],
            gamma=source_dict['gamma'],
            e_0=source_dict['e_0']
            )

    elif source_dict['spec_type'] == 'LogParabola':
        return_func = lambda x: log_parabola_3fgl(
            10**x,
            f_0=source_dict['f_0'],
            alpha=source_dict['gamma'],
            e_0=source_dict['e_0'],
            beta=source_dict['beta']
            )

    elif source_dict['spec_type'] == 'PLExpCutoff' or source_dict['spec_type'] == 'PLSuperExpCutoff':
        return_func = lambda x: pl_super_exp_cutoff_3fgl(
            10**x,
            f_0=source_dict['f_0'],
            gamma=source_dict['gamma'],
            e_0=source_dict['e_0'],
            cutoff=source_dict['cutoff'],
            exp_index=source_dict['exp_index']
            )

    return return_func


def get_gamma_dict(fermi_cat, source):
    '''
    search the dict for a source and produce its properties
    '''
    for cat_entry in fermi_cat:
        if cat_entry['name'] == source:
            return_dict = {
                'name': source,
                'e_0': cat_entry['pivot_energy'],
                'f_0': cat_entry['flux_density'],
                'gamma': cat_entry['spectral_index'],
                'beta': cat_entry['beta'],
                'cutoff': cat_entry['cutoff'],
                'exp_index': cat_entry['exp_index'],
                'spec_type': cat_entry['spec_type']
            }
            break

    return return_dict
