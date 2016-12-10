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
