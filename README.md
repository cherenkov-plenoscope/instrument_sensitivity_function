ARCHIVE
=======
This repository is out of service.
Thanks to Max L. Ahnen for his work. Parts of repositorry life on in ```starter_kit/plenoirf```.


Instrument Sensitivity Function
-------------------------------
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Arxiv](https://img.shields.io/badge/astro--ph.HE-arXiv%3A1701.06048-B31B1B.svg)](https://arxiv.org/abs/1701.06048) [![DOI](https://img.shields.io/badge/doi-10.3847%2F1538--4357%2Faa5b97-blue.svg)](https://doi.org/10.3847/1538-4357/aa5b97) 

for the Atmospheric Cherenkov Plenoscope (ACP)


Takes the Instrument Response Functions (IRF)s of an ACP to estimate the time-to-detections for known sources in the gamma-ray sky.

__installation__

```
git clone https://github.com/TheBigLebowSky/instrument_sensitivity_function.git
cd instrument_sensitivity_function/
pip install -r requirements.txt
```

__Usage__
Estimate the Integral Spectral Exclusion Zone (ISEZ)

    acp_isez --in=<in folder to where the csvs are>  --cutoff=10e-3 # 10GV --rel_flux=0.05<rel. flux secondary to primary particles> --fov=6.5 <in degree> --src=<3FGL name of comparison source> --out=<folder>
