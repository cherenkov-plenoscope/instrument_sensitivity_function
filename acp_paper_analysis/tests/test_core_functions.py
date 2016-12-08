'''
This is a set of test in order to check the
core functions
'''
import acp_paper_analysis as acp
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
