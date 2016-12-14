'''
This methods shall test my plotting routines
'''
import acp_paper_analysis as acp
import gamma_limits_sensitivity as gls

import numpy
import matplotlib
import pytest


def test_get_rates_over_energy_figure():
    '''
    Test to check a drawing method
    '''
    effective_area_dict = acp.get_interpolated_effective_areas(
            acp.__path__[0] + '/resources/test_infolder/'
    )
    resource_dict = acp.get_resources_paths()

    electron_positron_flux = acp.get_cosmic_ray_flux_interpol(
        resource_dict['fluxes']['electron_positron'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=3.,
        base_area_in_cm_2=1e4
        )
    proton_flux = acp.get_cosmic_ray_flux_interpol(
        resource_dict['fluxes']['proton'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=2.7,
        base_area_in_cm_2=1e4
        )

    figure, data = acp.get_rates_over_energy_figure(
        effective_area_dict,
        proton_spec=proton_flux,
        electron_positron_spec=electron_positron_flux,
        rigidity_cutoff_in_tev=10e-3,
        relative_flux_below_cutoff=0.1,
        roi_radius_in_deg=1.,
        e_0=1.,
        f_0=1e-10,
        gamma=-2.6
        )

    assert isinstance(figure, matplotlib.figure.Figure)
    assert isinstance(data, dict)
    for data_name in data:
        assert isinstance(
            data[data_name], numpy.ndarray
            )

def test_plot_rate_over_energy_charged_diffuse():
    '''
    This test shall see if the rate plotting
    and integration of charged particle spectra works
    '''
    effective_area_dict = acp.get_interpolated_effective_areas(
            acp.__path__[0] + '/resources/test_infolder/'
    )
    resource_dict = acp.get_resources_paths()

    effective_area = effective_area_dict['electron_positron']['trigger']
    style = 'k'
    label = 'e+-'
    charged_spec = acp.get_cosmic_ray_flux_interpol(
        resource_dict['fluxes']['electron_positron'],
        base_energy_in_TeV=1e-3,
        plot_power_slope=3.,
        base_area_in_cm_2=1e4
        )
    cutoff = 0.9
    relative_flux_below_cutoff = 0.1
    roi_radius_in_deg = 0.5

    plot_data, rate = acp.plot_rate_over_energy_charged_diffuse(
        effective_area=effective_area,
        style=style,
        label=label,
        charged_spec=charged_spec,
        cutoff=cutoff,
        relative_flux_below_cutoff=relative_flux_below_cutoff,
        roi_radius_in_deg=roi_radius_in_deg
        )

    plot_data2, rate2 = acp.plot_rate_over_energy_charged_diffuse(
        effective_area=effective_area,
        style=style,
        label=label,
        charged_spec=charged_spec,
        cutoff=cutoff*0.95,
        relative_flux_below_cutoff=relative_flux_below_cutoff,
        roi_radius_in_deg=roi_radius_in_deg
        )

    assert isinstance(plot_data, numpy.ndarray)
    assert isinstance(rate, numpy.ndarray)

    # check that the plot data has two columns
    assert numpy.shape(plot_data)[1] == 2
    # check that energy column is bigger than 100 keV
    assert numpy.any(plot_data[:, 0] > -7)
    assert numpy.any(plot_data2[:, 0] > -7)
    assert rate > 0  # check that the rate is bigger than 0

    # check that with a lower cutoff, you get a higher rate
    assert rate2 > rate


def test_plot_rate_over_energy_power_law_source():
    '''
    This test shall see if the rate plotting
    and integration of charged particle spectra works
    '''
    effective_area_dict = acp.get_interpolated_effective_areas(
            acp.__path__[0] + '/resources/test_infolder/'
    )

    effective_area = effective_area_dict['gamma']['trigger']
    style = 'k'
    label = 'gamma'
    e_0 = 1.
    f_0 = 1e11
    gamma = -2.6

    plot_data, rate = acp.plot_rate_over_energy_power_law_source(
        effective_area=effective_area,
        style=style,
        label=label,
        e_0=e_0,
        f_0=f_0,
        gamma=gamma
        )

    plot_data2, rate2 = acp.plot_rate_over_energy_power_law_source(
        effective_area=effective_area,
        style=style,
        label=label,
        e_0=e_0,
        f_0=f_0*1.05,
        gamma=gamma
        )

    assert isinstance(plot_data, numpy.ndarray)
    assert isinstance(rate, numpy.ndarray)

    # check that the plot data has two columns
    assert numpy.shape(plot_data)[1] == 2
    # check that energy column is bigger than 100 keV
    assert numpy.any(plot_data[:, 0] > -7)
    assert numpy.any(plot_data2[:, 0] > -7)
    assert rate > 0  # check that the rate is bigger than 0

    # check that with a higher flux norm., you get a higher rate
    assert rate2 > rate
