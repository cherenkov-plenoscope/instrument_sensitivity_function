'''
This is a set of test in order to check the
analysis functionality
'''
import acp_paper_analysis as acp
import pytest
import numpy
import matplotlib
import scipy


def test_generate_absolute_filepaths():
    '''
    This test is for checking the
    generate absolute filepath method
    '''
    with pytest.raises(Exception) as e_info:
        acp.generate_absolute_filepaths(in_path='')
        assert e_info is ValueError


def test_get_interpolated_effective_areas():
    '''
    This test checks the parsing of the effective area files
    '''
    effective_area_dict = acp.get_interpolated_effective_areas(
            acp.__path__[0] + '/resources/test_infolder/'
        )

    # chech that there are 3 (gamma, proton, electron/positron) Aeffs
    assert len(effective_area_dict) == 3

    for particle_type in effective_area_dict:
        for cut in effective_area_dict[particle_type]:
            assert isinstance(
                effective_area_dict[particle_type][cut],
                scipy.interpolate.interpolate.interp1d
                )


def test_get_effective_area_figure():
    '''
    Test to check a drawing method. Should return figure
    '''
    effective_area_dict = acp.get_interpolated_effective_areas(
            acp.__path__[0] + '/resources/test_infolder/'
    )

    # start producing plots and data products
    effective_area_figure = acp.get_effective_area_figure(effective_area_dict)

    assert isinstance(effective_area_figure, matplotlib.figure.Figure)


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


def test_analysis():
    '''
    This test checks if the analysis does
    make sense.
    '''
    result_dict = acp.analysis(
        acp.__path__[0] + '/resources/test_infolder/',
        is_test=True
        )

    for plot_name in result_dict['plots']:
        assert isinstance(
            result_dict['plots'][plot_name], matplotlib.figure.Figure
            )

    for data_name in result_dict['data']:
        assert isinstance(
            result_dict['data'][data_name], numpy.ndarray
            )
