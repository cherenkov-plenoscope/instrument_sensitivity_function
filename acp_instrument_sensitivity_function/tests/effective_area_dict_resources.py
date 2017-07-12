from os.path import join
import pkg_resources
import gamma_limits_sensitivity as gls


gamma_collection_area_path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function', 
    join('resources','test_infolder','gamma_aeff.dat')
)

electron_collection_acceptance_path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function', 
    join('resources','test_infolder','electron_positron_aeff.dat')
)

proton_collection_acceptance_path = pkg_resources.resource_filename(
    'acp_instrument_sensitivity_function', 
    join('resources','test_infolder','proton_aeff.dat')
)

effective_area_dict = {
    'gamma': gls.get_effective_area(gamma_collection_area_path),
    'electron_positron': gls.get_effective_area(electron_collection_acceptance_path),
    'proton': gls.get_effective_area(proton_collection_acceptance_path)
}
