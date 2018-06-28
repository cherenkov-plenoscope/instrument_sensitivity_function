import os
from os.path import join
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from astropy.table import Table
import acp_instrument_sensitivity_function as acp
import gamma_limits_sensitivity as gls
from pkg_resources import resource_filename



def get_resources_paths():
    '''
    Absolute paths to resource files with fluxes (protons, leptons),
    Crab, FermiLAT
    '''
    resource_paths = {
        'fluxes': {
            'electron_positron': resource_filename(
                'acp_instrument_sensitivity_function',
                'resources/e_plus_e_minus_spec.dat'
            ),
            'proton': resource_filename(
                'acp_instrument_sensitivity_function',
                'resources/proton_spec.dat'
            )
        },
        'Aeff': {
            'magic': resource_filename(
                'acp_instrument_sensitivity_function',
                'resources/MAGIC_lowZd_Aeff.dat'
            )
        },
        'isez': {
            'fermi_lat': resource_filename(
                'acp_instrument_sensitivity_function',
                'resources/FermiLAT_isez_p8r2_source_v6_10yr_gal_north.txt'
            )
        },
        'crab': {
            'broad_sed': resource_filename(
                'acp_instrument_sensitivity_function',
                'resources/crab_nebula_sed_fermi_magic.dat'
            )
        },
        'fermi_lat': {
            '3fgl': resource_filename(
                'acp_instrument_sensitivity_function',
                'resources/FermiLAT_3FGL_gll_psc_v16.fit'
            )
        }
    }
    return resource_paths


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


def get_cosmic_ray_flux_interpolated(
    path,
    base_energy_in_TeV,
    plot_power_slope,
    base_area_in_cm_2,
    base_time_in_sec=1.
):
    '''
    Get the interpolated cosmic-ray fluxes from a file in path.

    my own units are: log10(E/TeV), sec, sr, cm^2.
    AMS02 fluxes are usually given in E/GeV, m^2, sec, sr
    '''
    b_energy_x_in_tev = base_energy_in_TeV
    b_energy_y_in_tev = base_energy_in_TeV

    return get_spectrum_from_linear_file(
        path,
        b_energy_x_in_tev=b_energy_x_in_tev,
        b_energy_y_in_tev=b_energy_y_in_tev,
        base_area_in_cm_2=base_area_in_cm_2,
        base_time_in_sec=base_time_in_sec,
        plot_power_slope=plot_power_slope)


def get_fermi_lat_integral_spectral_exclusion_zone(path):
    '''
    This reads in a file containing the isez from
    FermiLAT (they call it broadband sensitivity)

    Transforms it into an interpolated function,
    in usual coordinates: TeV, cm^2, s
    '''
    return get_spectrum_from_linear_file(
        path,
        b_energy_x_in_tev=1e-6,  # was given in MeV
        b_energy_y_in_tev=0.62415091,  # was given in erg
        base_area_in_cm_2=1.,
        base_time_in_sec=1.,
        plot_power_slope=2.)


def get_crab_spectrum(path):
    '''
    This reads in a file containing the Crab nebula SED

    Transforms it into an interpolated spectrum,
    in usual coordinates: TeV, cm^2, s
    '''
    return get_spectrum_from_linear_file(
        path,
        b_energy_x_in_tev=1e-3,  # was given in GeV
        b_energy_y_in_tev=1.,  # was given in TeV
        base_area_in_cm_2=1.,
        base_time_in_sec=1.,
        plot_power_slope=2.)


def get_spectrum_from_linear_file(
    path,
    b_energy_x_in_tev=1.,
    b_energy_y_in_tev=1.,
    base_area_in_cm_2=1.,
    base_time_in_sec=1.,
    plot_power_slope=0.
):
    '''
    Read file, first column is energy (not logarithmic)
    and the second column some sort of flux, also linear.
    '''
    if plot_power_slope < 0:
        raise ValueError('plot_power_slope is supposed to be positive. ' +
                         'It is the power of energy by which the flux points' +
                         ' were multiplied'
                         )

    data = np.loadtxt(path)

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
    data_interpol = scipy.interpolate.interpolate.interp1d(
        data[:, 0],
        data[:, 1],
        bounds_error=False,
        fill_value=0.)

    return data_interpol


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

    elif (
        source_dict['spec_type'] == 'PLExpCutoff' or
        source_dict['spec_type'] == 'PLSuperExpCutoff'
    ):
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


def pl_super_exp_cutoff_3fgl(energy, f_0, gamma, e_0, cutoff, exp_index):
    '''
    pl super exponential cutoff as defined in 3FGL cat,
    but with already negative gamma
    '''
    return f_0*(energy/e_0)**(gamma)*np.exp(
        (e_0/cutoff)**exp_index - (energy/cutoff)**exp_index
        )


def psf_electromagnetic_in_deg(energy_in_tev):
    '''
    This function returns the half angle / deg of the
    psf cone which contains 0.67% of gamma events
    according to Aharonian et al. 5@5 paper
    '''
    return 0.8*(energy_in_tev*1000.)**(-0.4)


def solid_angle_of_cone(apex_angle_in_deg):
    '''
    WIKI:
    solid angle of cone with apex angle 2phi =
    area of a spherical cap on a unit sphere

    input: deg
    returns: steradian
    '''
    return 2*np.pi*(1-np.cos(apex_angle_in_deg/180.*np.pi))


def cutoff_spec(
    charged_spec,
    cutoff_energy_TeV,
    relative_flux_below_cutoff
):
    '''
    this is a function in order to check the cutoff
    is implemented

    cutoff / TeV
    '''
    return lambda x: (
        charged_spec(x) *
        (0.5*(np.sign(10**x - cutoff_energy_TeV) + 1) *
            (1 - relative_flux_below_cutoff) +
            relative_flux_below_cutoff)     # heaviside function
        )

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


def get_interpol_func_scaled(func, gamma_eff):
    '''
    Function to return efficiency scaled
    effective area of the instrument
    '''
    x_s = np.array(func.x)
    y_s = np.array([func(x)*gamma_eff for x in x_s])

    scaled_interpol = scipy.interpolate.interpolate.interp1d(
        x_s,
        y_s,
        bounds_error=False,
        fill_value=0.
    )

    return scaled_interpol


def lambda_lim(
    f_0,
    gamma,
    e_0,
    a_eff_interpol,
    beta,
    cutoff,
    exp_index,
    spec_type
):
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
        threshold)


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


# Collect all resources
# ---------------------

gamma_response_path = 'run/irf/gamma/results/irf.csv'
electron_response_path = 'run/irf/electron/results/irf.csv'
proton_response_path = 'run/irf/proton/results/irf.csv'
source = '3FGL J2254.0+1608'
out_dir = 'run/isf'
fov_in_deg = 6.5
rigidity_cutoff_in_TeV=10e-3
psf_containment = 0.67
relative_flux_below_cutoff=0.05

gamma_response = gls.get_effective_area(gamma_response_path)
electron_response = gls.get_effective_area(electron_response_path)
proton_response = gls.get_effective_area(proton_response_path)

resource_paths = get_resources_paths()

fermi_lat_3fgl_catalog = get_3fgl_catalog(
    resource_paths['fermi_lat']['3fgl'])

gamma_source_flux = get_gamma_spect(fermi_lat_3fgl_catalog, source=source)

electron_positron_flux = get_cosmic_ray_flux_interpolated(
    resource_paths['fluxes']['electron_positron'],
    base_energy_in_TeV=1e-3,
    plot_power_slope=3.,
    base_area_in_cm_2=1e4
)

proton_flux = get_cosmic_ray_flux_interpolated(
    resource_paths['fluxes']['proton'],
    base_energy_in_TeV=1e-3,
    plot_power_slope=2.7,
    base_area_in_cm_2=1e4
)


# plot
# ----

number_points = 40
dpi=250
pixel_rows = 1920
pixel_columns = 1920


# Effective-Area gamma-rays
# -------------------------
figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
lmar = 0.1
bmar = 0.06
tmar = 0.02
rmar = 0.02
axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

log_E_TeV = np.linspace(np.log10(0.0001), np.log10(1), number_points)
gamma_A_cm2 = gamma_response(log_E_TeV)

axes.plot(
    np.power(10, log_E_TeV)*1e3,
    gamma_A_cm2/(1e2*1e2),
    linestyle='-',
    color='k',
    label='gamma-rays')
axes.loglog()
axes.legend(loc='best', fontsize=10)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_xlabel('Energy / GeV')
axes.set_ylabel('Area / m$^2$')
axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
figure.savefig(
    join(out_dir, 'response_to_gamma_rays.png'),
    dpi=dpi)


# Effective-Acceptance charged particles
# --------------------------------------
figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

log_E_TeV = np.linspace(np.log10(0.0001), np.log10(1), number_points)
electron_A_cm2_sr = electron_response(log_E_TeV)
proton_A_cm2_sr = proton_response(log_E_TeV)

axes.plot(
    np.power(10, log_E_TeV)*1e3,
    electron_A_cm2_sr/(1e2*1e2),
    linestyle='--',
    color='k',
    label='electrons and positrons')

axes.plot(
    np.power(10, log_E_TeV)*1e3,
    proton_A_cm2_sr/(1e2*1e2),
    linestyle=':',
    color='k',
    label='protons')

axes.loglog()
axes.legend(loc='best', fontsize=10)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_xlabel('Energy / GeV')
axes.set_ylabel('Acceptance / (m$^2$ sr)')
axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
figure.savefig(
    join(out_dir, 'response_to_charged_particles.png'),
    dpi=dpi)


# Expected rates
# --------------
energy_range = [0.0001, 1]

# Gamma

gamma_diff_rate_roi = lambda energy: (
    gamma_source_flux(energy) * gamma_response(energy) * psf_containment)

gamma_integrand_roi = lambda energy: (
    gamma_source_flux(np.log10(energy)) *
    gamma_response(np.log10(energy)) *
    psf_containment)

points_to_watch_out = [energy_range[0], energy_range[0]*10]
gamma_rate_roi = scipy.integrate.quad(
    gamma_integrand_roi,
    energy_range[0],
    energy_range[1],
    limit=10000,
    full_output=1,
    points=points_to_watch_out
    )[0]

# Electron and Positron

electron_mass = 0.511e-6  # in TeV/c^2

electron_cutoff_energy = rigidity_to_energy(
    rigidity_cutoff_in_TeV,
    charge=1,
    mass=electron_mass)

electron_positron_flux_cutoff = cutoff_spec(
    charged_spec=electron_positron_flux,
    cutoff_energy_TeV=electron_cutoff_energy,
    relative_flux_below_cutoff=relative_flux_below_cutoff)

solid_angle_ratio = lambda energy: (
    solid_angle_of_cone(psf_electromagnetic_in_deg(energy)) /
    solid_angle_of_cone(fov_in_deg/2.)
)

electron_positron_diff_rate_roi = lambda energy: (
    electron_positron_flux_cutoff(energy) *
    electron_response(energy) *
    solid_angle_ratio(10**energy))

electron_positron_integrand_roi = lambda energy: (
    electron_positron_flux_cutoff(np.log10(energy)) *
    electron_response(np.log10(energy)) *
    solid_angle_ratio(energy))

points_to_watch_out = [energy_range[0], energy_range[0]*10]
if (
    electron_cutoff_energy > energy_range[0] and
    electron_cutoff_energy < energy_range[1]
):
    points_to_watch_out.append(electron_cutoff_energy)

electron_positron_rate_roi = scipy.integrate.quad(
    electron_positron_integrand_roi,
    energy_range[0],
    energy_range[1],
    limit=10000,
    full_output=1,
    points=points_to_watch_out
    )[0]


# Proton

proton_mass = 0.938272e-3 # in TeV/c^2

proton_cutoff_energy = rigidity_to_energy(
    rigidity_cutoff_in_TeV,
    charge=1,
    mass=proton_mass)

proton_flux_cutoff = cutoff_spec(
    charged_spec=proton_flux,
    cutoff_energy_TeV=proton_cutoff_energy,
    relative_flux_below_cutoff=relative_flux_below_cutoff)

proton_diff_rate_roi = lambda energy: (
    proton_flux_cutoff(energy) *
    proton_response(energy) *
    solid_angle_ratio(10**energy))

proton_integrand_roi = lambda energy: (
    proton_flux_cutoff(np.log10(energy)) *
    proton_response(np.log10(energy)) *
    solid_angle_ratio(energy))

points_to_watch_out = [energy_range[0], energy_range[0]*10]
if (
    electron_cutoff_energy > energy_range[0] and
    electron_cutoff_energy < energy_range[1]
):
    points_to_watch_out.append(electron_cutoff_energy)

proton_rate_roi = scipy.integrate.quad(
    proton_integrand_roi,
    energy_range[0],
    energy_range[1],
    limit=10000,
    full_output=1,
    points=points_to_watch_out
    )[0]

# Figure

figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

log_E_TeV = np.linspace(np.log10(0.0001), np.log10(1), number_points)

gamma_diff_rate_roi_sTeV = gamma_diff_rate_roi(log_E_TeV)
axes.plot(
    np.power(10, log_E_TeV)*1e3,
    gamma_diff_rate_roi_sTeV/1e3,
    linestyle='-',
    color='k',
    label='gamma-rays from '+source)

electron_positron_rate_roi_sTeV = electron_positron_diff_rate_roi(log_E_TeV)
axes.plot(
    np.power(10, log_E_TeV)*1e3,
    electron_positron_rate_roi_sTeV/1e3,
    linestyle='--',
    color='k',
    label='electrons and positrons')

proton_rate_roi_sTeV = proton_diff_rate_roi(log_E_TeV)
axes.plot(
    np.power(10, log_E_TeV)*1e3,
    proton_rate_roi_sTeV/1e3,
    linestyle=':',
    color='k',
    label='protons')

axes.loglog()
axes.legend(loc='best', fontsize=10)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_xlabel('Energy / GeV')
axes.set_ylabel('Rates / (s GeV)$^{-1}$')
axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
figure.savefig(
    join(out_dir, 'expected_trigger_rates.png'),
    dpi=dpi)


# Integral-Spectral-Exclusion-Zone
# --------------------------------




# get the integral bg rate in on-region (roi)
acp_sigma_bg = electron_positron_rate_roi + proton_rate_roi
acp_alpha = 1./3.

# make a coparison of the Fermi-LAT, MAGIC,
# and ACP integral spectral exclusion zone
energy_range = [0.1e-3, 10.]  # in TeV
# get efficiency scaled acp aeff
acp_aeff_scaled = get_interpol_func_scaled(
    gamma_response,
    gamma_eff=psf_containment)


'''
This function shall return a set of isze curves, in a figure and as data
Furthermore it shall compare everything to te crab nebula emission.
'''
crab_broad_spectrum = get_crab_spectrum(resource_paths['crab']['broad_sed'])

# get magic sensitivity parameters as stated in ul paper
magic_aeff = gls.get_effective_area(resource_paths['Aeff']['magic'])
magic_sigma_bg = 0.0020472222222222224  # bg per second in the on region
magic_alpha = 0.2  # five off regions
n_points_to_plot = 21
magic_energy_range = gls.get_energy_range(magic_aeff)
t_obs=50.*3600.

fermi_lat_isez = get_fermi_lat_integral_spectral_exclusion_zone(
    resource_paths['isez']['fermi_lat'])



figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

log_resolution=0.05

# Crab reference fluxes

for i in range(4):
    scale_factor = np.power(10., (-1)*i),
    log_resolution = 0.2

    e_x = 10**np.arange(
        np.log10(energy_range[0]),
        np.log10(energy_range[1])+0.05,
        log_resolution)

    e_y = np.array([crab_broad_spectrum(np.log10(x)) for x in e_x])
    e_y = e_y*scale_factor

    axes.plot(
        e_x*1e3,
        e_y*1e-3*1e4,
        color='k',
        linestyle='--',
        label='%.3f Crab' % np.power(10., (-1)*i),
        alpha=1./(1.+i))


# Fermi-LAT

log_resolution = 0.05

e_x = 10**np.arange(
    np.log10(energy_range[0]),
    np.log10(energy_range[1])+0.05,
    log_resolution)

e_y = np.array([fermi_lat_isez(np.log10(x)) for x in e_x])

axes.plot(
    e_x*1e3,
    e_y*1e-3*1e4,
    color='k',
    linestyle='-',
    label='Fermi-LAT 10y galactic north',)

# MAGIC
n_points_to_plot = 21
waste_figure = plt.figure()
magic_energy_x, magic_dn_de_y = gls.plot_sens_spectrum_figure(
    sigma_bg=magic_sigma_bg,
    alpha=magic_alpha,
    t_obs=t_obs,
    a_eff_interpol=magic_aeff,
    e_0=magic_energy_range[0]*5.,
    n_points_to_plot=n_points_to_plot,
    fmt='b',
    label='')
axes.plot(
    magic_energy_x*1e3,
    magic_dn_de_y*1e-3*1e4,
    'b',
    label='MAGIC %2.0fh' % (t_obs/3600.))


# ACP
acp_energy_range = gls.get_energy_range(acp_aeff_scaled)
acp_energy_x, acp_dn_de_y = gls.plot_sens_spectrum_figure(
    sigma_bg=acp_sigma_bg,
    alpha=acp_alpha,
    t_obs=t_obs,
    a_eff_interpol=acp_aeff_scaled,
    e_0=acp_energy_range[0]*5.,
    n_points_to_plot=n_points_to_plot,
    fmt='r',
    label='')
axes.plot(
    acp_energy_x*1e3,
    acp_dn_de_y*1e-3*1e4,
    'r',
    label='Portal %2.0fh' % (t_obs/3600.))

axes.loglog()
axes.legend(loc='best', fontsize=10)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_xlabel('Energy / GeV')
axes.set_ylabel('d Flux / d Energy / (m$^2$ s GeV)$^{-1}$')
axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
figure.savefig(
    join(out_dir, 'integral_spectral_exclusion_zone.png'),
    dpi=dpi)

# Times to detection
# ------------------

sorted_times_to_detection_map, reduced_catalog = get_time_to_detections(
    fermi_lat_3fgl_catalog,
    a_eff=gamma_response,
    sigma_bg=acp_sigma_bg,
    alpha=acp_alpha,
    out_path=out_dir)

sorted_t_est_list = np.array(sorted_times_to_detection_map)[:, 1]

figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

yvals = np.arange(len(sorted_t_est_list))
axes.step(
    sorted_t_est_list,
    yvals,
    color='k')
axes.axvline(x=3600*50, color='grey', linestyle=':')
axes.text(x=3600*50*1.1, y=8e-1, s='50h')

axes.loglog()
axes.legend(loc='best', fontsize=10)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.set_xlabel('time-to-detection / s')
axes.set_ylabel('number of sources')
axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
figure.savefig(
    join(out_dir, 'times_to_detection.png'),
    dpi=dpi)


# Time to detection of Gamma-ray-burst GBR-130427A

grb_f0 = 1.e-7
grb_gamma = -2.
grb_e0 = 1.

grb_130427A_time_to_detection = acp.time_to_detection(
    f_0=grb_f0,
    gamma=grb_gamma,
    e_0=grb_e0,
    a_eff_interpol=gamma_response,
    sigma_bg=acp_sigma_bg,
    alpha=acp_alpha)

grb_130427A_gamma_rate = grb_f0*gls.effective_area_averaged_flux(
    gamma=grb_gamma,
    e_0=grb_e0,
    a_eff_interpol=gamma_response)