from os.path import join
import csv
import datetime
import numpy as np
from tqdm import tqdm
import scipy
from astropy.table import Table
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
            ),
            'cta-south': resource_filename(
                'acp_instrument_sensitivity_function',
                'resources/cta_south_aeff_50h.dat'
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


def get_3fgl_catalog(path):
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
        'Flux1000': 'flux1000'
    }

    t = Table.read(path)
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


def get_cosmic_ray_spectrum_interpolated(
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


def get_gamma_ray_spectrum_of_source(fermi_catalog, source_name):
    '''
    This function produces a function for
    calculating the spectum of a named source
    in the fermi-lat 3fgl
    '''

    source_dict = get_gamma_ray_source_properties(fermi_catalog, source_name)

    if source_dict['spec_type'] == 'PowerLaw':
        return_func = lambda x: gls.power_law(
            energy=10**x,
            f_0=source_dict['f_0'],
            gamma=source_dict['gamma'],
            e_0=source_dict['e_0'])

    elif source_dict['spec_type'] == 'LogParabola':
        return_func = lambda x: power_law_log_parabola_according_to_3fgl(
            10**x,
            f_0=source_dict['f_0'],
            alpha=source_dict['gamma'],
            e_0=source_dict['e_0'],
            beta=source_dict['beta'])

    elif (
        source_dict['spec_type'] == 'PLExpCutoff' or
        source_dict['spec_type'] == 'PLSuperExpCutoff'
    ):
        return_func = lambda x: power_law_super_exp_cutoff_according_to_3fgl(
            10**x,
            f_0=source_dict['f_0'],
            gamma=source_dict['gamma'],
            e_0=source_dict['e_0'],
            cutoff=source_dict['cutoff'],
            exp_index=source_dict['exp_index'])

    return return_func


def get_gamma_ray_source_properties(fermi_catalog, source_name):
    '''
    search the dict for a source and produce its properties
    '''
    for cat_entry in fermi_catalog:
        if cat_entry['name'] == source_name:
            return_dict = {
                'name': source_name,
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


def power_law_super_exp_cutoff_according_to_3fgl(
    energy,
    f_0,
    gamma,
    e_0,
    cutoff,
    exp_index
):
    '''
    pl super exponential cutoff as defined in 3FGL cat,
    but with already negative gamma
    '''
    return f_0*(energy/e_0)**(gamma)*np.exp(
        (e_0/cutoff)**exp_index - (energy/cutoff)**exp_index)


def power_law_log_parabola_according_to_3fgl(energy, f_0, alpha, e_0, beta):
    '''
    log parabola as defined in 3fgl cat
    but with already negative alpha and beta
    '''
    return f_0*(energy/e_0)**(+alpha+beta*np.log10(energy/e_0))


psf_electromagnetic_containment = 0.67


def psf_electromagnetic_in_deg(energy_in_tev):
    '''
    This function returns the half angle / deg of the
    psf cone which contains 0.67% of gamma events
    according to Aharonian et al. 5@5 paper

    @article{aharonian2001,
        Author = {
            Aharonian, FA and Konopelko, AK and V{\"o}lk, HJ and Quintana, H},
        Journal = {Astroparticle Physics},
        Number = {4},
        Pages = {335--356},
        Publisher = {Elsevier},
        Title = {
            5@ 5--a 5 GeV energy threshold array of imaging atmospheric
            Cherenkov telescopes at 5 km altitude},
        Volume = {15},
        Year = {2001}}
    '''
    return 0.8*(energy_in_tev*1000.)**(-0.4)


def psf_electromagnetic_low_energy_acp_in_deg(energy_in_tev):
    if type(energy_in_tev) is np.ndarray:
        out = np.zeros_like(energy_in_tev)
        for i in range(energy_in_tev.shape[0]):
            out[i] = psf_electromagnetic_low_energy_acp_in_deg_float(
                energy_in_tev[i])
        return out
    else:
        return psf_electromagnetic_low_energy_acp_in_deg_float(energy_in_tev)


def psf_electromagnetic_low_energy_acp_in_deg_float(energy_in_tev):
    """
    The angular resolution for low energy gamma-rays as it was found in the
    early (first) studies for the 71m Portal-ACP.

    For energies above 2.5GeV, we use the psf estimated for 5@5, for energies
    below 2.5GeV we use the psf found in simulations on the ACP.
    """
    """
    energy_bin_centers = 1e-3*np.array([
        0.9,
        1.26,
        1.764,
        2.4696])
    one_sigma_resolutions = np.array([
        0.34297829882763775,
        0.34597156398104262,
        0.3778997256173609,
        0.48665502619107004])

    if energy_in_tev > 0.0025:
        return psf_electromagnetic_in_deg(energy_in_tev)
    else:
        return np.interp(
            x=energy_in_tev,
            xp=energy_bin_centers,
            fp=one_sigma_resolutions)
    """
    return 0.34297829882763775


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
        fill_value=0.)

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
        integrand = lambda x: power_law_log_parabola_according_to_3fgl(
            x,
            f_0=f_0,
            alpha=gamma,
            e_0=e_0,
            beta=beta
            )*a_eff_interpol(np.log10(x))

    elif spec_type == 'PLExpCutoff' or spec_type == 'PLSuperExpCutoff':
        integrand = lambda x: power_law_super_exp_cutoff_according_to_3fgl(
            x,
            f_0=f_0,
            gamma=gamma,
            e_0=e_0,
            cutoff=cutoff,
            exp_index=exp_index
            )*a_eff_interpol(np.log10(x))

    return scipy.integrate.quad(
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
            time_to_det = time_to_detection(
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
            i[1],
            fermi_lat_3fgl_catalog[i[0]]['flux1000'],
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
                'pivot_energy, spectral_index, flux_density, time_est, ' +
                'flux1000' +
                '\n'
                )
            for row in reduced_sorted_catalog:
                writer.writerow(row)

    return detection_times, reduced_sorted_catalog
