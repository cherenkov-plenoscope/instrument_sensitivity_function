'''
This is the main of the ul method paper demonstration

Usage:
  acp_paper_analysis --in=<path> [--out=<path>]
  acp_paper_analysis (-h | --help)
  acp_paper_analysis --version

Options:
  --in=<path>           Path to folder with effective areas of the acp
  --out=<path>          Optional argument for specifying the output directory
  -h --help             Show this screen.
  --version             Show version.
'''
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import acp_paper_analysis as acp
import datetime


def main_logic(arguments, dictionary):
    # if out path is none, just show the data
    if arguments['--out'] is None:
        plt.show()
    # else save to disk
    else:
        for plot_name in dictionary['plots']:
            dictionary['plots'][plot_name].savefig(
                arguments['--out']+'/'+plot_name+'.png',
                bbox_inches='tight'
                )
            dictionary['plots'][plot_name].savefig(
                arguments['--out']+'/'+plot_name+'.pdf',
                bbox_inches='tight'
                )
        for data_name in dictionary['data']:
            np.savetxt(
                arguments['--out']+'/'+data_name+'.csv',
                dictionary['data'][data_name],
                fmt='%.6e',
                header=(data_name + ', written: ' +
                        datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                            )
                        ),
                delimiter=','
                )


def main():
    '''
    The main.
    '''
    version = pkg_resources.require("acp_paper_analysis")[0].version
    arguments = docopt(__doc__, version=version)

    dictionary = acp.analysis(
        arguments['--in']
        )
    main_logic(arguments, dictionary)


if __name__ == '__main__':
    main()
