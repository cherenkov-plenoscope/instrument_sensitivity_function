Instrument Sensitivity Function
-------------------------------
for the Atmospheric Cherenkov Plenoscope (ACP)
[![Arxiv](https://img.shields.io/badge/astro--ph.HE-arXiv%3A1701.06048-B31B1B.svg)](https://arxiv.org/abs/1701.06048) 

Takes the Instrument Response Functions (IRF)s of an ACP to estimate the time-to-detections for known sources in the gamma-ray sky.

__installation__

```
git clone https://github.com/TheBigLebowSky/instrument_sensitivity_function.git
cd acp_paper_analysis/
pip install -r requirements.txt
```

__Usage__

    acp_paper_analysis --in=<in folder to where the csvs are>  --cutoff=10e-3 # 10GV --rel_flux=0.05<rel. flux secondary to primary particles> --fov=6.5 <in degree> --src=<3FGL name of comparison source> --out=<folder>
