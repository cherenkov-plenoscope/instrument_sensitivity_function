import acp_instrument_sensitivity_function as isf
import gamma_limits_sensitivity as gls
import pytest
import numpy
import matplotlib
import scipy
import tempfile


def test_generate_absolute_filepaths():
    '''
    This test is for checking the
    generate absolute filepath method
    '''
    with pytest.raises(Exception) as e_info:
        isf.generate_absolute_filepaths(in_path='')
        assert e_info is ValueError


def test_get_interpolated_effective_areas():
    '''
    This test checks the parsing of the effective area files
    '''
    effective_area_dict = isf.get_interpolated_effective_areas(
            isf.__path__[0] + '/resources/test_infolder/'
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
    effective_area_dict = isf.get_interpolated_effective_areas(
            isf.__path__[0] + '/resources/test_infolder/'
    )

    # start producing plots and data products
    effective_area_figure = isf.get_charged_acceptance_figure(
        effective_area_dict)

    assert isinstance(effective_area_figure, matplotlib.figure.Figure)


def test_get_gamma_effective_area_figure():
    '''
    Test to check a drawing method. Should return figure
    '''
    effective_area_dict = isf.get_interpolated_effective_areas(
            isf.__path__[0] + '/resources/test_infolder/'
    )

    # start producing plots and data products
    effective_area_figure = isf.get_gamma_effective_area_figure(
        effective_area_dict)

    assert isinstance(effective_area_figure, matplotlib.figure.Figure)


def test_get_time_to_detections():
    '''
    This test checks if the time to detectin method is working
    '''
    resource_dict = isf.get_resources_paths()
    magic_aeff = gls.get_effective_area(resource_dict['Aeff']['magic'])
    sigma_bg_test = 20./3600.
    alpha_test = 1./5.

    fermi_lat_3fgl_catalog = isf.get_3fgl_catalog(
        resource_dict['fermi_lat']['3fgl']
        )

    with tempfile.TemporaryDirectory() as tempfolder:
        detection_time_list, reduced_catalog = isf.get_time_to_detections(
            fermi_lat_3fgl_catalog,
            a_eff=magic_aeff,
            sigma_bg=sigma_bg_test,
            alpha=alpha_test,
            is_test=True,
            out_path=tempfolder
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
