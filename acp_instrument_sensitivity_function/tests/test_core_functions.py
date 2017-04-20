'''
This is a set of test in order to check the
core functions
'''
import acp_instrument_sensitivity_function as acp
import gamma_limits_sensitivity as gls

import numpy as np
import pytest
import os.path


def test_get_resources_paths():
    '''
    Test if this function returns a sensible dictionary
    with existing paths
    '''
    resource_dict = acp.get_resources_paths()

    for prim_type in resource_dict:
        for sub_type in resource_dict[prim_type]:
            assert os.path.isfile(resource_dict[prim_type][sub_type])


def test_get_electron_positron_flux():
    '''
    Test if the function really returns sensible lepton fluxes
    and convert it to units of TeV, cm^2, sec, sr
    '''
    resource_dict = acp.get_resources_paths()

    electron_positron = acp.get_cosmic_ray_flux_interpol(
        resource_dict['fluxes']['electron_positron'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=3.,
        base_area_in_cm_2=1e4
        )

    assert electron_positron(-5.) == 0.
    assert electron_positron(-3.24527) > 0.
    assert electron_positron(-3.24527) < 10e0
    assert electron_positron(-0.02) > 0.
    assert electron_positron(-0.02) < 10e0
    assert electron_positron(1.0) == 0.


def test_get_proton_flux():
    '''
    Test if the function really returns sensible proton fluxes
    and convert it to units of TeV, cm^2, sec, sr
    '''
    resource_dict = acp.get_resources_paths()

    proton_flux = acp.get_cosmic_ray_flux_interpol(
        resource_dict['fluxes']['proton'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=2.7,
        base_area_in_cm_2=1e4
        )

    assert proton_flux(-5.) == 0.
    assert proton_flux(-3.) > 0.
    assert proton_flux(-3.) < 10e0
    assert proton_flux(-0.02) > 0.
    assert proton_flux(-0.02) < 10e0
    assert proton_flux(1.0) == 0.


def test_get_fermi_lat_isez():
    '''
    This will test reading in the integral spectral exclusion zone
    from the FermiLAT supplied file
    '''
    resource_dict = acp.get_resources_paths()
    fermi_lat_isez = acp.get_fermi_lat_isez(resource_dict['isez']['fermi_lat'])

    assert fermi_lat_isez(-4.6) == 0.  # lower limit is about 30 MeV
    assert fermi_lat_isez(0.+1e-9) == 0.  # upper limit is 1 TeV
    assert fermi_lat_isez(-2) > 0.
    assert fermi_lat_isez(-2) < 1e0


def test_get_crab_spectrum():
    '''
    This will test reading in crab sed file and transform it into
    my regular units
    '''
    resource_dict = acp.get_resources_paths()
    crab_spectrum = acp.get_crab_spectrum(resource_dict['crab']['broad_sed'])

    assert crab_spectrum(-4.6) == 0.  # lower limit is about 30 MeV
    assert crab_spectrum(1.35) == 0.  # upper limit is 1 TeV
    assert crab_spectrum(-2) > 0.
    assert crab_spectrum(-2) < 1e0

    # check few specific fluxes for sanity
    rel_accuracy_margin = 0.4  # 40 percent accurate should be OK as test
    log10_e1 = -1
    log10_e2 = +1
    result_1 = 7e-9
    result_2 = 5e-14
    assert np.abs(
        crab_spectrum(log10_e1) - result_1
        )/result_1 < rel_accuracy_margin

    assert np.abs(
        crab_spectrum(log10_e2) - result_2
        )/result_2 < rel_accuracy_margin


def test_get_3fgl_catalog():
    '''
    This method tests if the get_3fgl_catalog
    function is doing good work
    '''
    resource_dict = acp.get_resources_paths()

    fermi_lat_3fgl_catalog = acp.get_3fgl_catalog(
        resource_dict['fermi_lat']['3fgl']
        )
    for source in fermi_lat_3fgl_catalog:
        assert source['spectral_index'] < 0.
        assert source['flux_density'] < 1e-1
        assert source['pivot_energy'] > 0.
        if source['spec_type'] == 'LogParabola':
            assert source['beta'] < 0
            assert source['beta'] >= -1
        elif source['spec_type'] == 'PLExpCutoff' or source['spec_type'] == 'PLSuperExpCutoff':
            assert source['cutoff'] > 0
            assert source['exp_index'] > 0
            assert source['spectral_index'] <= 0.5


def test_rigidity_to_energy():
    '''
    this test checks that rigidity to energy conversions work
    from TV -> TeV
    '''
    assert np.isclose(
        acp.rigidity_to_energy(
            10e-3,
            charge=1,
            mass=0.938272e-3  # proton
            ), 0.009105649263430134
        )


def test_time_to_detection():
    '''
    Test if the time to detection gives sensible results
    '''
    resource_dict = acp.get_resources_paths()
    magic_aeff = gls.get_effective_area(resource_dict['Aeff']['magic'])

    f_0 = 1e-9
    e_0 = 1.
    gamma = -2.7
    sigma_bg = 2.7e-3
    alpha = 0.2

    t_est = acp.time_to_detection(
        f_0, gamma, e_0, magic_aeff, sigma_bg, alpha
        )

    assert t_est < 360  # check that the thing is detected faster than 0.1h
    assert t_est > 0  # check that the thing is detectable


def test_solid_angle_of_cone():
    '''
    This test is to check if the solid angle calculation is correct.
    '''
    assert np.isclose(acp.solid_angle_of_cone(apex_angle_in_deg=180), 4*np.pi)
    assert np.isclose(acp.solid_angle_of_cone(apex_angle_in_deg=90), 2*np.pi)
    assert np.isclose(acp.solid_angle_of_cone(apex_angle_in_deg=0.), 0.)


def test_cutoff_spec():
    '''
    Test the cutoff spectrum is actually working
    '''
    resource_dict = acp.get_resources_paths()
    charged_spec = acp.get_cosmic_ray_flux_interpol(
        resource_dict['fluxes']['electron_positron'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=3.,
        base_area_in_cm_2=1e4
        )

    cutoff = 10e-3  # 10GeV
    relative_flux_below_cutoff = 0.1

    cutoff_func = acp.cutoff_spec(
        charged_spec, cutoff, relative_flux_below_cutoff)

    assert bool(np.isclose(
        charged_spec(np.log10(cutoff*0.3)),
        cutoff_func(np.log10(cutoff*0.3))
        )) is False

    assert np.isclose(
        charged_spec(np.log10(cutoff*1.01)),
        cutoff_func(np.log10(cutoff*1.01))
        )
    assert np.isclose(
        charged_spec(np.log10(cutoff*0.9)),
        cutoff_func(np.log10(cutoff*0.9))/relative_flux_below_cutoff
        )

    cutoff = 1.124e-3  # 10GeV
    relative_flux_below_cutoff = 0.461346

    cutoff_func = acp.cutoff_spec(
        charged_spec, cutoff, relative_flux_below_cutoff)
    assert np.isclose(
        charged_spec(np.log10(cutoff*4.01)),
        cutoff_func(np.log10(cutoff*4.01))
        )
    assert np.isclose(
        charged_spec(np.log10(cutoff*0.6)),
        cutoff_func(np.log10(cutoff*0.6))/relative_flux_below_cutoff
        )


def test_log_parabola():
    '''
    Test the log parabola function
    '''
    energy = 0.1
    f_0 = 1e-11
    gamma = -2.6
    e_0 = 0.01
    beta = 0.0
    
    res_buf1 = gls.power_law(energy, f_0=f_0, gamma=gamma, e_0=e_0)
    res_buf2 = acp.log_parabola_3fgl(
        energy, f_0=f_0, alpha=gamma, e_0=e_0, beta=beta
        )

    assert np.isclose(res_buf1, res_buf2)


def test_pl_exp_cutoff():
    '''
    Test the log parabola function
    '''
    energy = 0.1
    f_0 = 1e-11
    gamma = -2.6
    e_0 = 0.01
    cutoff = 10000.
    exp_index = 1

    res_buf1 = gls.power_law(energy, f_0=f_0, gamma=gamma, e_0=e_0)
    res_buf2 = acp.pl_super_exp_cutoff_3fgl(
        energy, f_0=f_0, gamma=gamma, e_0=e_0, cutoff=cutoff, exp_index=exp_index
        )

    assert np.isclose(res_buf1, res_buf2)


def test_get_gamma_spect():
    '''
    Test the function to return gamma ray spectra,
    dependent on the name of the source
    '''
    source = '3FGL J1836.2+5925'
    resource_dict = acp.get_resources_paths()

    fermi_lat_3fgl_catalog = acp.get_3fgl_catalog(
        resource_dict['fermi_lat']['3fgl']
        )

    spectrum = acp.get_gamma_spect(fermi_lat_3fgl_catalog, source=source)

    log10_e1 = np.log10(1e-3)
    log10_e2 = np.log10(10e-3)

    # check that higher energies produce lower fluxes
    assert spectrum(log10_e1) > spectrum(log10_e2)
