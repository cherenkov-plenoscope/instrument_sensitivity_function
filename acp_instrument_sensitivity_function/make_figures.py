from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import scipy
import acp_instrument_sensitivity_function as isf
import gamma_limits_sensitivity as gls
import json


def analysis(
    gamma_collection_area_path='run/irf/gamma/results/irf.csv',
    electron_collection_acceptance_path='run/irf/electron/results/irf.csv',
    proton_collection_acceptance_path='run/irf/proton/results/irf.csv',
    source_name='3FGL J2254.0+1608',
    out_dir='run/isf',
    fov_in_deg=6.5,
    rigidity_cutoff_in_tev=10*1e-3,
    relative_flux_below_cutoff=0.05,
    number_points=40,
    dpi=250,
    pixel_rows=1920,
    pixel_columns=1920,
    lmar=0.1,
    bmar=0.06,
    tmar=0.02,
    rmar=0.02,
):
    # Collect all resources
    # ---------------------
    gamma_response = gls.get_effective_area(
        gamma_collection_area_path)
    electron_response = gls.get_effective_area(
        electron_collection_acceptance_path)
    proton_response = gls.get_effective_area(
        proton_collection_acceptance_path)

    resource_paths = isf.utils.get_resources_paths()

    fermi_lat_3fgl_catalog = isf.utils.get_3fgl_catalog(
        resource_paths['fermi_lat']['3fgl'])

    gamma_source_flux = isf.utils.get_gamma_ray_spectrum_of_source(
        fermi_lat_3fgl_catalog, source_name=source_name)

    electron_positron_flux = isf.utils.get_cosmic_ray_spectrum_interpolated(
        resource_paths['fluxes']['electron_positron'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=3.,
        base_area_in_cm_2=1e4)

    proton_flux = isf.utils.get_cosmic_ray_spectrum_interpolated(
        resource_paths['fluxes']['proton'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=2.7,
        base_area_in_cm_2=1e4)

    # Assumed-angular-resolution
    # --------------------------

    # @article{hofmann2006performance,
    #   title={Performance limits for Cherenkov instruments},
    #   author={Hofmann, Werner},
    #   journal={arXiv preprint astro-ph/0603076},
    #   year={2006}
    # }

    hoffmann_E_vs_res = np.array([  # GeV and deg
        # [9749.2, 0.002188],
        [966.70, 0.005758],
        [95.855, 0.019627],
        [29.055, 0.043256],
        [9.7492, 0.113806],
    ])

    hoffmann_E_vs_res_with_magnetic_field = np.array([  # GeV and deg
        # [9749.2, 0.002188],
        [991.57, 0.007771],
        [98.321, 0.030353],
        [29.803, 0.073590],
    ])

    # @article{fermi2010fermi,
    #   title={Fermi gamma-ray imaging of a radio galaxy},
    #   author={Fermi-LAT Collaboration and others},
    #   journal={Science},
    #   volume={328},
    #   number={5979},
    #   pages={725--729},
    #   year={2010},
    #   publisher={American Association for the Advancement of Science}
    # }

    def fermi_resolution_deg(Energy_TeV):
        return 0.8 * (Energy_TeV*1e3)**(-0.8)

    E_TeV_to_10 = np.linspace(1e-4, 1e-2, number_points)
    E_TeV_1G = np.linspace(0.001, 1, number_points)

    out = {
        'Hofman_limits': {
            'energy_GeV': hoffmann_E_vs_res[:, 0].tolist(),
            'resolution_deg': hoffmann_E_vs_res[:, 1].tolist()},
        'Hofman_limits_with_earth_magnetic_field': {
            'energy_GeV': hoffmann_E_vs_res_with_magnetic_field[:, 0].tolist(),
            'resolution_deg':
                hoffmann_E_vs_res_with_magnetic_field[:, 1].tolist()},
        'Fermi_LAT': {
            'energy_GeV': (E_TeV_to_10*1e3).tolist(),
            'resolution_deg': fermi_resolution_deg(E_TeV_to_10).tolist()},
        'Aharonian_et_al_5at5': {
            'energy_GeV': (E_TeV_1G*1e3).tolist(),
            'resolution_deg':
                isf.utils.psf_electromagnetic_in_deg(E_TeV_1G).tolist()},
        'Possible_benefits_with_Portal_and_5at5': {
            'energy_GeV': (E_TeV_1G*1e3).tolist(),
            'resolution_deg':
                isf.utils.psf_electromagnetic_low_energy_acp_in_deg(
                    E_TeV_1G).tolist()},
    }

    with open(join(out_dir, 'assumed_angular_resolution.json'), 'wt') as fout:
        fout.write(json.dumps(out))

    figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
    axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

    axes.plot(
        out['Fermi_LAT']['energy_GeV'],
        out['Fermi_LAT']['resolution_deg'],
        'k-.',
        label='Fermi-LAT')

    axes.plot(
        out['Aharonian_et_al_5at5']['energy_GeV'],
        out['Aharonian_et_al_5at5']['resolution_deg'],
        linestyle='-',
        color='k',
        label='Aharonian et al., 5@5')

    axes.plot(
        out['Hofman_limits']['energy_GeV'],
        out['Hofman_limits']['resolution_deg'],
        'ko--',
        markerfacecolor='k',
        label='Hofmann, limits')

    axes.plot(
        out['Hofman_limits_with_earth_magnetic_field']['energy_GeV'],
        out['Hofman_limits_with_earth_magnetic_field']['resolution_deg'],
        'ko:',
        markerfacecolor='white',
        label='Hofmann, limits, with earth-magnetic-field')

    axes.loglog()
    axes.legend(loc='best', fontsize=10)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.set_xlabel('Energy / GeV')
    axes.set_ylabel('Resolution / deg')
    axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    figure.savefig(
        join(out_dir, 'assumed_angular_resolution.png'),
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
    axes.set_ylabel('Acceptance / m$^2$ sr')
    axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    figure.savefig(
        join(out_dir, 'response_to_charged_particles.png'),
        dpi=dpi)

    # Effective-Area gamma-rays
    # -------------------------
    figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
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
    axes.set_ylabel('Acceptance / m$^{-2}$ sr$^{-1}$')
    axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    figure.savefig(
        join(out_dir, 'response_to_charged_particles.png'),
        dpi=dpi)

    # Expected rates
    # --------------
    energy_range = [0.0001, 1]

    # Gamma
    def gamma_diff_rate_roi(energy):
        return (
            gamma_source_flux(energy) *
            gamma_response(energy) *
            isf.utils.psf_electromagnetic_containment)

    def gamma_integrand_roi(energy):
        return (
            gamma_source_flux(np.log10(energy)) *
            gamma_response(np.log10(energy)) *
            isf.utils.psf_electromagnetic_containment)

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

    electron_cutoff_energy = isf.utils.rigidity_to_energy(
        rigidity_cutoff_in_tev,
        charge=1,
        mass=electron_mass)

    electron_positron_flux_cutoff = isf.utils.cutoff_spec(
        charged_spec=electron_positron_flux,
        cutoff_energy_TeV=electron_cutoff_energy,
        relative_flux_below_cutoff=relative_flux_below_cutoff)

    def solid_angle_ratio(energy):
        return (
            isf.utils.solid_angle_of_cone(
                isf.utils.psf_electromagnetic_low_energy_acp_in_deg(energy)) /
            isf.utils.solid_angle_of_cone(fov_in_deg/2.))

    def electron_positron_diff_rate_roi(energy):
        return (
            electron_positron_flux_cutoff(energy) *
            electron_response(energy) *
            solid_angle_ratio(10**energy))

    def electron_positron_integrand_roi(energy):
        return (
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

    proton_mass = 0.938272e-3  # in TeV/c^2

    proton_cutoff_energy = isf.utils.rigidity_to_energy(
        rigidity_cutoff_in_tev,
        charge=1,
        mass=proton_mass)

    proton_flux_cutoff = isf.utils.cutoff_spec(
        charged_spec=proton_flux,
        cutoff_energy_TeV=proton_cutoff_energy,
        relative_flux_below_cutoff=relative_flux_below_cutoff)

    def proton_diff_rate_roi(energy):
        return (
            proton_flux_cutoff(energy) *
            proton_response(energy) *
            solid_angle_ratio(10**energy))

    def proton_integrand_roi(energy):
        return (
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
    with open(join(out_dir, 'expected_trigger_rates.json'), 'wt') as fout:
        fout.write(json.dumps(
            {
                'gamma_rate_roi': gamma_rate_roi,
                'electron_positron_rate_roi': electron_positron_rate_roi,
                'proton_rate_roi': proton_rate_roi,
            }
        ))


    figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
    axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

    log_E_TeV = np.linspace(np.log10(0.0001), np.log10(1), number_points)

    gamma_diff_rate_roi_sTeV = gamma_diff_rate_roi(log_E_TeV)
    axes.plot(
        np.power(10, log_E_TeV)*1e3,
        gamma_diff_rate_roi_sTeV/1e3,
        linestyle='-',
        color='k',
        label='gamma-rays from ' + source_name)

    electron_positron_rate_roi_sTeV = electron_positron_diff_rate_roi(
        log_E_TeV)
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
    axes.set_ylabel('differential Rate / s$^{-1}$ GeV$^{-1}$')
    axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    figure.savefig(
        join(out_dir, 'expected_trigger_rates.png'),
        dpi=dpi)

    # Fluxes of cosmic particles
    # --------------------------
    figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
    axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

    log_E_TeV = np.linspace(np.log10(0.0001), np.log10(1), number_points)

    E = 10**log_E_TeV  # TeV
    F_ep = electron_positron_flux_cutoff(log_E_TeV)  # [cm**2 sr s TeV]**(-1)

    axes.plot(
        E*1e3,
        F_ep*1e-3*1e4,
        linestyle='--',
        color='k',
        label='electrons and positrons')

    F_p = proton_flux_cutoff(log_E_TeV)  # [cm**2 sr s TeV]**(-1)

    axes.plot(
        E*1e3,
        F_p*1e-3*1e4,
        linestyle=':',
        color='k',
        label='protons')

    axes.loglog()
    axes.legend(loc='best', fontsize=10)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.set_xlabel('Energy / GeV')
    axes.set_ylabel('Flux / m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$')
    axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    figure.savefig(
        join(out_dir, 'cosmic_particle_fluxes.png'),
        dpi=dpi)

    n_points_to_plot = 11

    # Integral-Spectral-Exclusion-Zone
    # --------------------------------
    # get the integral bg rate in on-region (roi)
    acp_sigma_bg = electron_positron_rate_roi + proton_rate_roi
    acp_alpha = 1./5.

    # make a coparison of the Fermi-LAT, MAGIC,
    # and ACP integral spectral exclusion zone
    energy_range = [0.1e-3, 10.]  # in TeV
    # get efficiency scaled acp aeff
    acp_aeff_scaled = isf.utils.get_interpol_func_scaled(
        gamma_response,
        gamma_eff=isf.utils.psf_electromagnetic_containment)

    crab_broad_spectrum = isf.utils.get_crab_spectrum(
        resource_paths['crab']['broad_sed'])

    # get magic sensitivity parameters as stated in ul paper
    magic_aeff = gls.get_effective_area(resource_paths['Aeff']['magic'])
    magic_sigma_bg = 0.0020472222222222224  # bg per second in the on region
    magic_alpha = 0.2  # five off regions
    magic_energy_range = gls.get_energy_range(magic_aeff)
    t_obs = 50.*3600.

    # get CTA-south sensitivity parameters
    cta_south_aeff = gls.get_effective_area(resource_paths['Aeff']['cta-south'])
    cta_south_sigma_bg = 0.1017 # bg per second in the on region
    cta_south_alpha = 0.2  # five off regions
    cta_south_energy_range = gls.get_energy_range(cta_south_aeff)

    fermi_lat_isez = isf.utils.get_fermi_lat_integral_spectral_exclusion_zone(
        resource_paths['isez']['fermi_lat'])

    figure = plt.figure(figsize=(pixel_columns/dpi, pixel_rows/dpi))
    axes = figure.add_axes([lmar, bmar, 1-lmar-rmar, 1-bmar-tmar])

    log_resolution = 0.05

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

    # CTA-south
    waste_figure = plt.figure()
    cta_south_energy_x, cta_south_dn_de_y = gls.plot_sens_spectrum_figure(
        sigma_bg=cta_south_sigma_bg,
        alpha=cta_south_alpha,
        t_obs=t_obs,
        a_eff_interpol=cta_south_aeff,
        e_0=cta_south_energy_range[0]*5.,
        n_points_to_plot=n_points_to_plot,
        fmt='g',
        label='')
    axes.plot(
        cta_south_energy_x*1e3,
        cta_south_dn_de_y*1e-3*1e4,
        'g',
        label='CTA-south %2.0fh' % (t_obs/3600.))

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
    axes.set_ylabel('differential Flux / m$^{-2}$ s$^{-1}$ GeV$^{-1}$')
    axes.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    figure.savefig(
        join(out_dir, 'integral_spectral_exclusion_zone.png'),
        dpi=dpi)

    # Times to detection
    # ------------------
    sorted_times_to_detection, reduced_catalog = (
        isf.utils.get_time_to_detections(
            fermi_lat_3fgl_catalog,
            a_eff=gamma_response,
            sigma_bg=acp_sigma_bg,
            alpha=acp_alpha,
            out_path=out_dir))

    sorted_t_est_list = np.array(sorted_times_to_detection)[:, 1]

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
    # ------------------------------------------------
    grb_f0 = 1.e-7
    grb_gamma = -2.
    grb_e0 = 1.

    grb_130427A_time_to_detection = isf.utils.time_to_detection(
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
