from os.path import join
import pkg_resources
import gamma_limits_sensitivity as gls
import acp_instrument_sensitivity_function as isf


def test_resource_paths():
    resource_paths = isf.utils.get_resources_paths()

    assert 'fluxes' in resource_paths.keys()
    assert 'Aeff' in resource_paths.keys()
    assert 'isez' in resource_paths.keys()
    assert 'crab' in resource_paths.keys()
    assert 'fermi_lat' in resource_paths.keys()


def test_read_effective_areas():
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
