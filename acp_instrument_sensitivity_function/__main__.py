'''
Estimate the Integral Spectral Exclusion Zone (ISEZ) of the ACP.

Usage:
    acp_isez --gamma_area=<g_path> --electron_acceptance=<e_path> --proton_acceptance=<p_path> --cutoff=<arg> --rel_flux=<arg> --fov=<arg> [--src=<arg> --plot_isez_all --out=<path>]
    acp_isez (-h | --help)
    acp_isez --version

Options:
    --gamma_area=<g_path>           Path to the collection area (area) for a point like gamma-ray source.  
    --electron_acceptance=<e_path>  Path to the collection acceptacne (area*solid angle) IRF for a diffuse electrons.
    --proton_acceptance=<p_path>    Path to the collection acceptacne (area*solid angle) IRF for a diffuse protons.
    --cutoff=<arg>        Rigidity cutoff / TV
    --rel_flux=<arg>      Relative flux intensity below rigidity cutoff
    --fov=<arg>           Field of View of the simulated ACP / deg (typical: 6.5 deg)
    --src=<arg>           Optional: Gamma source 3FGL name for gamma-ray rate plot [default: 3FGL J2254.0+1608]
    --plot_isez_all       Optional: Plot many ISEZ curves for analysis
    --out=<path>          Optional: Argument for specifying the output directory
    -h --help             Show this screen.
    --version             Show version.
'''
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import acp_instrument_sensitivity_function as acp
import datetime
from os.path import join


def main():
    '''
    The main.
    '''
    version = pkg_resources.require("acp_instrument_sensitivity_function")[0].version
    arguments = docopt(__doc__, version=version)

    dictionary = acp.analysis(
        gamma_collection_area_path=arguments['--gamma_area'],
        electron_collection_acceptance_path=arguments['--electron_acceptance'], 
        proton_collection_acceptance_path=arguments['--proton_acceptance'],
        rigidity_cutoff_in_tev=float(arguments['--cutoff']),
        relative_flux_below_cutoff=float(arguments['--rel_flux']),
        fov_in_deg=float(arguments['--fov']),
        source=arguments['--src'],
        plot_isez_all=arguments['--plot_isez_all'],
        out_path=arguments['--out']
        )

    # if out path is none, just show the data
    if arguments['--out'] is None:
        plt.show()
    # else save to disk
    else:
        acp.save_results(path=arguments['--out'], dictionary=dictionary)


if __name__ == '__main__':
    main()
