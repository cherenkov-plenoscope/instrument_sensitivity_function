from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import scipy
import acp_instrument_sensitivity_function as isf


resource_paths = isf.utils.get_resources_paths()

electron_positron_flux = isf.utils.get_cosmic_ray_spectrum_interpolated(
    resource_paths['fluxes']['electron_positron'],
    base_energy_in_TeV=1e-3,
    plot_power_slope=3.,
    base_area_in_cm_2=1e4)


number_points = 101
log_E_TeV = np.linspace(np.log10(0.0001), np.log10(1), number_points)

# Electrons and Positrons

E = 10**log_E_TeV # TeV
F_ep = electron_positron_flux(log_E_TeV) # [cm**2 sr s TeV]**(-1)

gev_in_tev = 1e3
tev_in_gev = 1e-3
cm2_in_m2 = 1e4

plt.figure()
plt.plot(E*gev_in_tev, (F_ep*tev_in_gev*cm2_in_m2)*(E*gev_in_tev)**3, 'x')
plt.xlabel('E / GeV')
plt.ylabel('F / [m**2 s sr GeV GeV**-3]**(-1)')
plt.title('electrons and positrons')
plt.semilogx()
plt.show()

# Protons

proton_flux = isf.utils.get_cosmic_ray_spectrum_interpolated(
    resource_paths['fluxes']['proton'],
    base_energy_in_TeV=1e-3,
    plot_power_slope=2.7,
    base_area_in_cm_2=1e4)

E = 10**log_E_TeV # TeV
F_p = proton_flux(log_E_TeV) # [cm**2 sr s TeV]**(-1)

plt.figure()
plt.plot(E*gev_in_tev, (F_p*tev_in_gev*cm2_in_m2)*(E*gev_in_tev)**2.7, 'x')
plt.xlabel('E / GeV')
plt.ylabel('F / [m**2 s sr GeV GeV**-3]**(-1)')
plt.title('protons')
plt.semilogx()
plt.show()
