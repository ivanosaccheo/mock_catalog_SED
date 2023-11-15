import time 
import numpy as np 
from color_fitting_class import organize_colors
import os

#redshift_bins = np.linspace(0.1,3.5, 52)   Bins are computed automatically 
#redshift_bins = np.hstack([redshift_bins, np.linspace(3.7, 4.5, 3)])


QSO_filename = "DR16_QSOs_luminous.csv"
QSO_directory = os.path.expanduser("~/DATA/samples")
sigma_multiplying_factor = 3
save_chain_filename = f"fitting_luminous_{sigma_multiplying_factor}xsigma".  #also the date is added



colors = organize_colors(filename = QSO_filename, 
                         directory = QSO_directory)
colors.get_filters_name(filename = "lista_filtri.dat")
colors.get_parameters(filename = "flat_priors.dat")
colors.select_magnitudes()
colors.get_filters()
colors.get_bins(user_bins = None, N_objects = [30, 130, 700, 400, 100], N_bins = None, redshift_cuts = [0.6, 1.0, 3, 4])
colors.assign_bin()
colors.get_luminosity(H0 = 70, Om0 = 0.3, magnitudes_are_vega = True)
colors.get_mono_luminosity(2500)
colors.get_M_i()
colors.get_colors()
colors.clean_colors(wavelength_min = 912, wavelength_max = 3e4, clear_all_bin = True)

colors.get_bin_mean(clipping_sigma = 2)
colors.get_table_mean(replace = False)
colors.get_fitting_arrays(scale_sigma = sigma_multiplying_factor)
print("Uncertainties  increased!")

colors.plot_M_i(plot_temple = True)
colors.plot_colors()
colors.plot_redshift_distributions()

tic = time.perf_counter()
colors.get_scipy_fitting()
toc = time.perf_counter()

print(f"Scipy minimization done in {toc-tic} seconds")
colors.print_results()

colors.plot_scipy_fitting()
tic = time.perf_counter()
colors.emcee_fitting(filename = save_chain_filename, nsteps = 850)
toc = time.perf_counter()
print(f"EMCEE fitting done in {(toc-tic)/3600} hours")
