'''
This is a set of test in order to check the
core functions
'''
import acp_paper_analysis as acp

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


def test_get_proton_positron_flux():
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
    rel_accuracy_margin = 0.5  # 50 percent accurate should be OK as test
    log10_e1 = -1
    log10_e2 = +1
    result_1 = 7e-9
    result_2 = 5e-14
    assert np.abs(
        crab_spectrum(log10_e1) - result_1
        )/crab_spectrum(log10_e1) < rel_accuracy_margin

    assert np.abs(
        crab_spectrum(log10_e2) - result_2
        )/crab_spectrum(log10_e2) < rel_accuracy_margin


def test_get_3fgl_catalog():
    '''
    This method tests if the read_3fgl_catalog
    function is doing good work
    '''
    resource_dict = acp.get_resources_paths()

    fermi_lat_3fgl_catalog = acp.get_3fgl_catalog(
        resource_dict['fermi_lat']['3fgl']
        )
    for source in fermi_lat_3fgl_catalog:
        assert source['spectral_index'] < 0.
        assert source['flux_density'] < 1e-1
