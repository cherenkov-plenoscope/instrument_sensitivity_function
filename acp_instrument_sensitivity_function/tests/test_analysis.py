import acp_instrument_sensitivity_function as isf
import gamma_limits_sensitivity as gls
import pytest
import numpy
import matplotlib
import scipy
import tempfile
from os.path import join
import pkg_resources

gamma_collection_area_path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function',
    join('resources', 'test_infolder', 'gamma_aeff.dat')
)

electron_collection_acceptance_path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function',
    join('resources', 'test_infolder', 'electron_positron_aeff.dat')
)

proton_collection_acceptance_path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function',
    join('resources', 'test_infolder', 'proton_aeff.dat')
)

effective_area_dict = {
    'gamma': gls.get_effective_area(
        gamma_collection_area_path),
    'electron_positron': gls.get_effective_area(
        electron_collection_acceptance_path),
    'proton': gls.get_effective_area(
        proton_collection_acceptance_path)
}


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

    # chech that there are 3 (gamma, proton, electron/positron) Aeffs
    assert len(effective_area_dict) == 3

    for particle_type in effective_area_dict:
        assert isinstance(
            effective_area_dict[particle_type],
            scipy.interpolate.interpolate.interp1d
        )


def test_get_time_to_detections():
    '''
    This test checks if the time to detectin method is working
    '''
    resource_dict = isf.utils.get_resources_paths()
    magic_aeff = gls.get_effective_area(resource_dict['Aeff']['magic'])
    sigma_bg_test = 20./3600.
    alpha_test = 1./5.

    fermi_lat_3fgl_catalog = isf.utils.get_3fgl_catalog(
        resource_dict['fermi_lat']['3fgl']
        )

    with tempfile.TemporaryDirectory() as tempfolder:
        detection_times, reduced_catalog = isf.utils.get_time_to_detections(
            fermi_lat_3fgl_catalog,
            a_eff=magic_aeff,
            sigma_bg=sigma_bg_test,
            alpha=alpha_test,
            is_test=True,
            out_path=tempfolder)

    for detection in detection_times:
        assert detection[0] >= 0
        assert detection[0] <= 1e4  # this is the index
        assert detection[1] > 0

    # some cut was applied, check that it kicked out some indices
    assert len(fermi_lat_3fgl_catalog) > len(detection_times)

    # check that list is sorted
    detection_numpy_times = numpy.array([i[1] for i in detection_times])
    assert numpy.all((
        detection_numpy_times[1:] - detection_numpy_times[:-1]) > 0)

    for i in reduced_catalog:
        assert i[5] in (
            'PowerLaw',
            'LogParabola',
            'PLExpCutoff',
            'PLSuperExpCutoff'
            )
