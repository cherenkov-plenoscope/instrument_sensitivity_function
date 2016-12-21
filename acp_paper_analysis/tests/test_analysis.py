'''
This is a set of test in order to check the
analysis functionality
'''
import acp_paper_analysis as acp
import gamma_limits_sensitivity as gls
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
        assert isinstance(
            effective_area_dict[particle_type],
            scipy.interpolate.interpolate.interp1d
            )


def test_get_charged_acceptance_figure():
    '''
    Test to check a drawing method. Should return figure
    '''
    effective_area_dict = acp.get_interpolated_effective_areas(
            acp.__path__[0] + '/resources/test_infolder/'
    )

    # start producing plots and data products
    effective_area_figure = acp.get_charged_acceptance_figure(
        effective_area_dict)

    assert isinstance(effective_area_figure, matplotlib.figure.Figure)


def test_get_gamma_effective_area_figure():
    '''
    Test to check a drawing method. Should return figure
    '''
    effective_area_dict = acp.get_interpolated_effective_areas(
            acp.__path__[0] + '/resources/test_infolder/'
    )

    # start producing plots and data products
    effective_area_figure = acp.get_gamma_effective_area_figure(
        effective_area_dict)

    assert isinstance(effective_area_figure, matplotlib.figure.Figure)


# def test_analysis():
#     '''
#     This test checks if the analysis does
#     make sense.
#     '''
#     result_dict = acp.analysis(
#         acp.__path__[0] + '/resources/test_infolder/',
#         is_test=True
#         )

#     for plot_name in result_dict['plots']:
#         assert isinstance(
#             result_dict['plots'][plot_name], matplotlib.figure.Figure
#             )

#     for data_name in result_dict['data']:
#         assert isinstance(
#             result_dict['data'][data_name], numpy.ndarray
#             )


def test_get_time_to_detections():
    '''
    This test checks if the time to detectin method is working
    '''
    resource_dict = acp.get_resources_paths()
    magic_aeff = gls.get_effective_area(resource_dict['Aeff']['magic'])
    sigma_bg_test = 20./3600.
    alpha_test = 1./5.

    fermi_lat_3fgl_catalog = acp.get_3fgl_catalog(
        resource_dict['fermi_lat']['3fgl']
        )

    detection_time_list, reduced_catalog = acp.get_time_to_detections(
        fermi_lat_3fgl_catalog,
        a_eff=magic_aeff,
        sigma_bg=sigma_bg_test,
        alpha=alpha_test,
        is_test=True,
        out_path='/home/mknoetig/Desktop/'
        )

    for detection in detection_time_list:
        assert detection[0] >= 0
        assert detection[0] <= 1e4  # this is the index
        assert detection[1] > 0

    # some cut was applied, check that it kicked out some indices
    assert len(fermi_lat_3fgl_catalog) > len(detection_time_list)

    # check that list is sorted
    detection_numpy_times = numpy.array([i[1] for i in detection_time_list])
    assert numpy.all((
        detection_numpy_times[1:] - detection_numpy_times[:-1]) > 0)

    for i in reduced_catalog:
        assert i[5] in (
            'PowerLaw',
            'LogParabola',
            'PLExpCutoff',
            'PLSuperExpCutoff'
            )
