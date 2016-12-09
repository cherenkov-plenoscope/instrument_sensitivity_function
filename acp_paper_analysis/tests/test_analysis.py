'''
This is a set of test in order to check the
analysis functionality
'''
import acp_paper_analysis as acp
import pytest
import numpy
import matplotlib


def test_generate_absolute_filepaths():
    '''
    This test is for checking the
    generate absolute filepath method
    '''
    with pytest.raises(Exception) as e_info:
        acp.generate_absolute_filepaths(arguments={'--in': ''})
        assert e_info is ValueError


def test_analysis():
    '''
    This test checks if the analysis does
    make sense.
    '''
    resource_dict = acp.get_resources_paths()
    # test methods on one magic aeff file
    aeff_test = resource_dict['Aeff']['magic']

    result_dict = acp.analysis(
        gamma_aeff=aeff_test,
        gamma_aeff_cut=aeff_test,
        electron_positron_aeff=aeff_test,
        electron_positron_aeff_cut=aeff_test,
        proton_aeff=aeff_test,
        proton_aeff_cut=aeff_test,
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
