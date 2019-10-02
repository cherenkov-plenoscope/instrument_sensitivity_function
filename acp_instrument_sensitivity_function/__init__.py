from . import utils
from .make_figures import analysis
import numpy as np
import pkg_resources
import os
import astropy


def differential_flux_proton():
    """
    AMS-02 precision measurements
    """
    path = pkg_resources.resource_filename(
        'acp_instrument_sensitivity_function',
        os.path.join('resources', 'proton_spec.dat'))
    flux = np.genfromtxt(path)
    flux[:, 0] *= 1  # in GeV
    flux[:, 1] /= flux[:, 0]**2.7
    return {
        "energy": {
            "values": flux[:, 0].tolist(),
            "unit": "GeV"
        },
        "differential_flux": {
            "values": flux[:, 1].tolist(),
            "unit": "m^{-2} s^{-1} sr^{-1} GeV^{-1}"
        },
    }


def differential_flux_electron_positron():
    """
    AMS-02 precision measurements
    """
    path = pkg_resources.resource_filename(
        'acp_instrument_sensitivity_function',
        os.path.join('resources', 'e_plus_e_minus_spec.dat'))
    flux = np.genfromtxt(path)
    flux[:, 0] *= 1  # in GeV
    flux[:, 1] /= flux[:, 0]**3.0
    return {
        "energy": {
            "values": flux[:, 0].tolist(),
            "unit": "GeV"
        },
        "differential_flux": {
            "values": flux[:, 1].tolist(),
            "unit": "m^{-2} s^{-1} sr^{-1} GeV^{-1}"
        },
    }


def __power_law_super_exp_cutoff_according_to_3fgl(
    energy,
    flux_density,
    spectral_index,
    pivot_energy,
    cutoff_energy,
    exp_index
):
    '''
    power-law super exponential cutoff as defined in 3FGL catalog.
    Differential flux in m^{-2} s^{-1} GeV^{-1}
    '''
    return (flux_density*(energy/pivot_energy)**(spectral_index))*np.exp(
        (pivot_energy/cutoff_energy)**exp_index -
        (energy/cutoff_energy)**exp_index)


def __power_law_log_parabola_according_to_3fgl(
    energy,
    flux_density,
    spectral_index,
    pivot_energy,
    beta
):
    '''
    log parabola as defined in 3fgl cat
    but with already negative spectral_index and beta
    '''
    return flux_density*(energy/pivot_energy)**(
        +spectral_index+beta*np.log10(energy/pivot_energy))


def __power_law(
    energy,
    flux_density,
    spectral_index,
    pivot_energy
):
    return flux_density*(energy/pivot_energy)**(spectral_index)


def differential_flux_gamma_ray_source(energy, source):
    diff_flux = None
    spectrum = source['spectrum_type']["value"]
    if spectrum == "PowerLaw":
        diff_flux = __power_law(
            energy=energy,
            flux_density=source['flux_density']["value"],
            spectral_index=source['spectral_index']["value"],
            pivot_energy=source["pivot_energy"]["value"])
    elif spectrum == "LogParabola":
        diff_flux = __power_law_log_parabola_according_to_3fgl(
            energy=energy,
            flux_density=source['flux_density']["value"],
            spectral_index=source['spectral_index']["value"],
            pivot_energy=source["pivot_energy"]["value"],
            beta=source["beta"]["value"])
    elif spectrum == "PLSuperExpCutoff" or spectrum == "PLExpCutoff":
        diff_flux = __power_law_super_exp_cutoff_according_to_3fgl(
            energy=energy,
            flux_density=source['flux_density']["value"],
            spectral_index=source['spectral_index']["value"],
            pivot_energy=source["pivot_energy"]["value"],
            cutoff_energy=source["cutoff_energy"]["value"],
            exp_index=source["exp_index"]["value"])
    else:
        raise RuntimeError("Unknown type of spectrum '{:s}'".format(spectrum))

    return {
        "energy": {
            "values": energy.tolist(),
            "unit": "GeV"
        },
        "differential_flux": {
            "values": diff_flux.tolist(),
            "unit": "m^{-2} s^{-1} GeV^{-1}"
        },
    }


def gamma_ray_sources():
    """
    Fermi-LAT 3FGL gamma-ray-sources
    """
    __sources = []
    fermi_keys = [
        "Source_Name",  # string
        'RAJ2000',  # deg
        'DEJ2000',  # deg
        'GLON',  # deg
        'GLAT',  # deg
        'SpectrumType',  # string
        'Pivot_Energy',  # MeV
        'Spectral_Index',  # 1
        'Flux_Density',  # photons cm^{-2} MeV^{-1} s^{-1}
        'beta',  # 1
        'Cutoff',  # MeV
        'Exp_Index',  # 1
        'Flux1000',  # photons cm^{-2} s^{-1}
    ]

    fermi_3fgl_path = pkg_resources.resource_filename(
        'acp_instrument_sensitivity_function',
        resource_name=os.path.join(
            'resources',
            'FermiLAT_3FGL_gll_psc_v16.fit'))

    with astropy.io.fits.open(fermi_3fgl_path) as fits:
        num_sources = fits[1].header["NAXIS2"]
        for source_idx in range(num_sources):
            s = {}
            for fermi_key in fermi_keys:
                s[fermi_key] = fits[1].data[source_idx][fermi_key]
            __sources.append(s)

    sources = []
    for i in range(len(__sources)):
        f = __sources[i]
        s = {}
        s["name"] = {"value": f["Source_Name"], "unit": "str"}
        s["RAJ2000"] = {"value": f["RAJ2000"], "unit": "deg"}
        s["DEJ2000"] = {"value": f["DEJ2000"], "unit": "deg"}
        s["GLON"] = {"value": f["GLON"], "unit": "deg"}
        s["GLAT"] = {"value": f["GLAT"], "unit": "deg"}
        s["spectral_index"] = {"value": (-1.)*f["Spectral_Index"], "unit": "1"}
        s["spectrum_type"] = {"value": f["SpectrumType"], "unit": "str"}
        s["pivot_energy"] = {"value": f["Pivot_Energy"]*1e-3, "unit": "GeV"}
        s["cutoff_energy"] = {"value": f["Cutoff"]*1e-3, "unit": "GeV"}
        s["exp_index"] = {"value": f["Exp_Index"], "unit": "1"}
        s["flux1000"] = {"value": f["Cutoff"]*1e4, "unit": "m^{-2} s^{-1}"}
        s["flux_density"] = {
            "value": f["Flux_Density"]*1e4*1e3,
            "unit": "m^{-2} s^{-1} GeV^{-1}"}
        s["beta"] = {"value": f["beta"], "unit": "1"}
        sources.append(s)
    return sources
