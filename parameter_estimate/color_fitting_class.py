import os 
import warnings
import datetime
import numpy as np
import pandas as pd
import copy 
import matplotlib.pyplot as plt
from astropy import units, constants
from astropy.cosmology import FlatLambdaCDM
from astropy.stats import sigma_clip
import emcee 
from scipy.optimize import minimize
from astropy.table import Table
from my_functions import library as lb
from qsogen.model_colours import get_colours
from qsogen.qsosed import Quasar_sed


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


def get_temple_sed(redshift, parameters, parameters_name, wavlen= np.logspace(2.96, 4.48, 1000)):
    par_dictionary = dict(zip(parameters_name, parameters))
    par_dictionary.update({"fragal" : 0, "gplind" : 0})
    return Quasar_sed(redshift, wavlen = wavlen, **par_dictionary)

def get_temple_colors(redshift, M_i, theta, parameters_name, filters, 
                      colors_to_fit = 'all'):
    par_dictionary = dict(zip(parameters_name, theta))
    if not "fragal" in parameters_name:
        par_dictionary.update({"fragal" : 0, "gplind" : 0})
    
    if colors_to_fit == "all":
        colors_to_fit = np.ones(len(filters)-1, dtype = bool)
    model = get_colours(redshift, M_i = M_i, filters = filters, **par_dictionary)
    model = model[:, colors_to_fit]
    
    return model

def chi_square(y_model, y, yerr):
    return np.sum((y-y_model)**2/yerr**2)

def least_square(theta, redshift, y, yerr, M_i, parameters_name, filters, colors_to_fit):
    model = get_temple_colors(redshift, M_i, theta, parameters_name, filters, colors_to_fit = colors_to_fit)
    return chi_square(model, y, yerr)

def log_prior(theta, priors):
    if np.logical_and.reduce(np.logical_and(theta>=priors[:,0], theta<=priors[:,1])):
        return 0
    return -np.inf

def log_probability(theta, redshift, y, yerr, M_i, parameters_name, filters, colors_to_fit, priors):
    
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, redshift, y, yerr, M_i, parameters_name, filters, colors_to_fit)
    
    return lp + ll

def log_likelihood(theta, redshift, y, yerr, M_i, 
                   parameters_name, filters, colors_to_fit):
    
    model = get_temple_colors(redshift, M_i, theta, parameters_name, filters, colors_to_fit = colors_to_fit)
    sigma2 = yerr**2 
    S = (y - model) ** 2 / sigma2 + np.log(sigma2)
    S = np.ma.masked_invalid(S)
    return -0.5 * np.sum(S)

######### Functions for preparing sample

def find_finite(df, *args):
    are_finite = np.isfinite(df[[*args]])
    return np.logical_and.reduce(are_finite, axis = 1)

def find_positive(df, *args):
    are_positive = df[[*args]] > 0
    return np.logical_and.reduce(are_positive, axis = 1)

def get_AB_vega(filename = "AB_vega_conversion.npy", directory = "input"):
    return np.load(os.path.join(directory, filename), allow_pickle=True).item()


def vega_to_AB(table):
    table_copy = copy.deepcopy(table) 
    AB_vega_dict = get_AB_vega()
    for key, value in zip(AB_vega_dict.keys(), AB_vega_dict.values()):
        if key in table_copy.columns:
            table_copy[key] = table_copy[key] + value
            print(f"Transorming {key} to AB, i.e. adding {value}")
    return table_copy


def AB_to_vega(table):
    table_copy = copy.deepcopy(table)
    AB_vega_dict = get_AB_vega()
    for key, value in zip(AB_vega_dict.keys(), AB_vega_dict.values()):
        if key in table_copy.columns:
            table_copy[key] = table_copy[key] - value
            print(f"Transorming {key} to Vega, i.e. subtracting {value}")
    return table_copy


def select_luminous(table, luminosity_cut_low = 45.053, luminosity_cut_high = np.inf,  wavelength_cut = 5100,
              extra = ["SDSS_NAME", "DR16QSO_RA", "DR16QSO_DEC", "Redshift", "M_I", "EBV"]):
    
    new_table = vega_to_AB(table)
    
    
    filter_names = [i for i in table.columns if "e_" not in i and i not in extra]
    filtri = [lb.filtro(i) for i in filter_names]
    wavelengths = np.array([i.wav for i in filtri])
    
    magnitudes, extra = lb.two_2_three(new_table, extra_features = extra, has_wavelength = False)
    luminosities = np.zeros((magnitudes.shape[0], magnitudes.shape[1], 3))
    luminosities[:, :, 0] = wavelengths
    luminosities[:,:, 1:] = magnitudes
    redshift = extra["Redshift"].to_numpy()
    
    luminosities = lb.get_luminosity(luminosities, redshift)
    mono_luminosities = lb.monochromatic_lum(luminosities, wavelength_cut)

    logic = (np.log10(mono_luminosities) >= luminosity_cut_low) & (np.log10(mono_luminosities) < luminosity_cut_high)
    return logic

def remove_QSO_with_big_errors(table, filtri, err_thresholds):
    """
    filtri = list of filters to take into account 
    err_thresholds = float/list with the maximum uncertainties allowed
    
    """
    from itertools import cycle
    try:
        err_thresholds = err_thresholds if isinstance(err_thresholds, list) else [err_thresholds]
        if len(err_thresholds) != 1 and len(err_thresholds) != len(filtri):
            print("Warning the number of thresholds does not correpond to the number of filters")
        for filtro, threshold in zip(filtri, cycle(err_thresholds)):
            err_name = f"e_{filtro}"
            table = table[table[err_name] <= threshold]
            print(f"Keeping {len(table)} QSOs with err_{filtro} < {threshold}")
    except TypeError:
        err_name = f"e_{filtri}"
        table = table[table[err_name] <= err_thresholds]
        print(f"Keeping {len(table)} QSOs with err_{filtri} < {err_thresholds}")

    return table

def prepare_sample(filename = "DR16_QSOs_frozen.fits",
    return_only_magnitudes = True,
    only_luminous = True, luminosity_cuts_low = 45.053, luminosity_cuts_high = np.inf, wavelength_cuts = 5100,
    select_in_magnitudes = True, magnitudes_min = 0, magnitudes_max = 28, filtro_cuts = "UKIDSS.Y",
    remove_QSOs_with_big_uncertainties = True, filtri_uncertainties = "all",
    uncertainties_thresholds = 0.2, 
    redshift_cut = [0, 10],
    directory = os.path.expanduser("~/DATA/samples"),
    extra_columns = ["SDSS_NAME", "DR16QSO_RA", "DR16QSO_DEC", "Redshift", "M_I", "EBV"]):
    
    """
    Crea una tabella con le sole magnitudini e redshift a partire da una tabella inizale con altre colonne     (extra_columns) e incertezze associate alle magnitudini (devoni iniziare con "e_").
    Il campione selezionato è determinato da tagli in luminosità, magnitudini e ripulito di qso con incertezze    troppo grosse
    """
    table = Table.read(os.path.join(directory, filename), format = "fits")

    table = table.to_pandas()
    ### Cleaning sample
    table = table[find_finite(table, *table.columns[3:])]
    table = table[find_positive(table, *table.columns[6:])]
    
    filter_list = [i for i in table.columns if "e_" not in i and i not in extra_columns]
    
    ## Doing cuts in luminosity/brightness/redshift and removing uncertain data
    if only_luminous:
        try: 
            for L_cut_low, L_cut_high, W_cut in zip(luminosity_cuts_low, luminosity_cuts_high,wavelength_cuts):
                are_luminous = select_luminous(table, 
                                               luminosity_cut_low = L_cut_low, luminosity_cut_high = L_cut_high,
                                               wavelength_cut = W_cut, extra = extra_columns)
                table = table[are_luminous]
                print(f"Keeping {len(table)} QSOs with {L_cut_low} <= L < {L_cut_high}  at lambda = {W_cut} A")
            
        
        except TypeError:
            are_luminous = select_luminous(table, luminosity_cut_low = luminosity_cuts_low, 
                                           luminosity_cut_high = luminosity_cuts_high, wavelength_cut = wavelength_cuts, 
                                           extra = extra_columns)
            table = table[are_luminous]

            print(f"Keeping {len(table)} QSOs with {luminosity_cuts_low} <= L < {luminosity_cuts_high}  at lambda = {wavelength_cuts} A")
            
    
    if select_in_magnitudes:
        
        try:
            for filtro_cut, m_min, m_max in zip(filtro_cuts,magnitudes_min, magnitudes_max): 
                table = table[np.logical_and(table[filtro_cut] >= m_min, table[filtro_cut]<=m_max)]
                print(f"Keeping {len(table)} QSOs with {m_min} <= {filtro_cut} <= {m_max}")
        
        except TypeError:
            table = table[np.logical_and(table[filtro_cuts] >= magnitudes_min, 
                                         table[filtro_cuts]<=magnitudes_max)]
            print(f"Keeping {len(table)} QSOs with {magnitudes_min} <= {filtro_cuts} <= {magnitudes_max}")
            
            
    if remove_QSOs_with_big_uncertainties:
        if filtri_uncertainties == "all":
            filtri_uncertainties = filter_list
        table = remove_QSO_with_big_errors(table, filtri_uncertainties, uncertainties_thresholds)

    redshift_logic = np.logical_and(table["Redshift"]>=redshift_cut[0], table["Redshift"]<=redshift_cut[1])
   
    table = table[redshift_logic]
    print(f"Keeping {len(table)} QSOs with {redshift_cut[0]} <= Redshift <= {redshift_cut[1]}")
   
    if return_only_magnitudes:
        filter_list.append("Redshift")
        return table[filter_list]
        
    return table






class organize_colors:

    def __init__(self, table = None, filename = "krawczyk_no_host_magnitudes.csv", 
                 directory = os.path.expanduser("~/DATA/samples"),
                 **kwargs) -> None:
        """ 
        prende come input una tabella, altrimenti un file csv da cui può leggere la tabella
        """
        
        if table is not None:
            self.all_magnitudes = table
        else:    
            self.all_magnitudes = pd.read_csv(os.path.join(directory, filename), **kwargs)
        
        self.all_redshift = self.all_magnitudes["Redshift"].to_numpy()
        self.today = datetime.date.today()
        
        return None
    
    def get_filters_name(self, filename = "lista_filtri.dat", directory = "input", 
                         my_name_header = "myname", 
                         temple_name_header= "temple_name",
                         necessary_header = "necessary",
                         to_fit_header = "to_fit"):

    
        self.filter_df = pd.read_csv(os.path.join(directory, filename), comment="#", 
                                     delim_whitespace= True)
        self.my_filters = self.filter_df[my_name_header].to_list()
        for filtro in self.my_filters: assert(filtro in self.all_magnitudes.columns)
        
        self.temple_filters = self.filter_df[temple_name_header].to_list()
        self.necessary_filters = self.filter_df[my_name_header][self.filter_df[necessary_header]].to_list()
        self.logic_filter_to_fit = self.filter_df[to_fit_header].to_list()
        


    def get_parameters(self, filename = "flat_priors.dat", directory = "input"):
        self.prior_df = pd.read_csv(os.path.join(directory, filename), delim_whitespace= True,
                                    comment="#")
        self.parameters = self.prior_df["Name"].to_list()
        self.flat_priors = self.prior_df[["low_limit", "upper_limit"]].to_numpy()
        self.temple_best = self.prior_df["temple_best"].to_numpy()
    

    
    def select_magnitudes(self):
        not_nan = ~np.isnan(self.all_magnitudes[self.necessary_filters])
        self.magnitudes = self.all_magnitudes.loc[np.logical_and.reduce(not_nan, axis=1)]
        self.redshift = self.magnitudes["Redshift"].to_numpy()
        print(f"Selecting {len(self.magnitudes)} among {len(self.all_magnitudes)} QSOs")
        self.magnitudes = self.magnitudes.drop(columns= "Redshift")
    
    def get_filters(self, get_transmission = True):
        
        def order_filters(filtri):
            return np.argsort([filtro.wav for filtro in filtri])
        # Ordering filters depending on their wavelength
        self.filters = [lb.filtro(i) for i in self.my_filters]
        self.my_filters =[self.my_filters[idx] for idx in order_filters(self.filters)]
        self.temple_filters =[self.temple_filters[idx] for idx in order_filters(self.filters)]
        self.logic_filter_to_fit = [self.logic_filter_to_fit[idx] for idx in order_filters(self.filters)]
        self.filters = [self.filters[idx] for idx in order_filters(self.filters)]
        self.filters_to_fit = [i for i,j in zip(self.my_filters, self.logic_filter_to_fit) if j]
        self.logic_colors_to_fit = [i and j for i, j in zip(self.logic_filter_to_fit[:-1],
                                                            self.logic_filter_to_fit[1:])]
        
        if get_transmission:
            for filtro in self.filters: filtro.get_transmission()
        self.magnitudes = self.magnitudes[self.my_filters]
        
    def get_bins(self, user_bins = None, N_objects = None, N_bins = None, redshift_cuts = None):
        
        def get_quantiles(redshift, Nbins):
            quantiles_cuts = np.arange(Nbins+1)/Nbins
            quantiles = np.quantile(redshift, quantiles_cuts)
            quantiles[0], quantiles[-1] = np.nextafter(quantiles[0], -np.inf), np.nextafter(quantiles[-1], np.inf)
            return quantiles

        if user_bins is not None:
            for cut in user_bins: assert(isinstance(cut, float))
            self.redshift_bins = user_bins
            self.Nbins = len(self.redshift_bins) #Countig also the rightmost bin
        
        elif redshift_cuts is not None:
            redshift_cuts = np.array(redshift_cuts).flatten()
            self.Nbins = 0
            quantiles = []
            bins = np.digitize(self.redshift, redshift_cuts)
            try:
                for unique_bin, nobj in zip(np.unique(bins), N_objects, strict = True):
                    assert(isinstance(nobj, int))
                    idx = bins == unique_bin
                    nbins = int(np.ceil(np.sum(idx) / nobj))
                    quantiles.append(get_quantiles(self.redshift[idx], nbins))
                    self.Nbins += nbins
                    print(f"Grouping fixed {nobj} QSOs in {nbins} bins")
            
            except TypeError:
                print("Using fixed number of bins for each redshift cut")
                for unique_bin, nbins in zip(np.unique(bins), N_bins, strict = True):
                    assert(isinstance(nbins, int))
                    idx = bins == unique_bin
                    quantiles.append(get_quantiles(self.redshift[idx], nbins))
                    self.Nbins += nbins
                    print(f"Grouping {nobj} QSOs in fixed {nbins} bins")
            
            #L'ultimo elemnto di un quantile è equivalente al primo del successivo 
            self.redshift_bins = np.hstack([i[:-1] for i in quantiles])
        
        else:
            try:
                assert isinstance(N_objects, int)
                self.Nbins = int(np.ceil(len(self.redshift)/N_objects))
                print(f"Returning {self.Nbins} bins with {N_objects} objects each")
            except AssertionError:
                assert isinstance(N_bins, int)
                self.Nbins = N_bins
            
            self.redshift_bins = get_quantiles(self.redshift, self.Nbins)

           
    def get_luminosity(self, H0 = 70, Om0 = 0.3, magnitudes_are_vega = False):
        cosmo = FlatLambdaCDM(H0 = H0, Om0 = Om0)
        self.luminosity_distance = cosmo.luminosity_distance(self.redshift).cgs.value
        L = np.zeros((self.magnitudes.shape))
        
        if magnitudes_are_vega:
            AB_magnitudes = vega_to_AB(self.magnitudes)
        else:
            AB_magnitudes = self.magnitudes

        
        for i, col in enumerate(AB_magnitudes):
            L[:,i] = (10**(-0.4*(AB_magnitudes[col]+48.6))) * 2.998e18/self.filters[i].wav
            L[:,i] = L[:,i] * 4 *np.pi * self.luminosity_distance**2
        self.luminosity = np.zeros((AB_magnitudes.shape[0], AB_magnitudes.shape[1], 2))
        
        for i in range(L.shape[1]):
            self.luminosity[:,i,0] = self.filters[i].wav/(self.redshift+1)
            self.luminosity[:,i,1] = L[:,i]
        
        return None
       
    
    def get_mono_luminosity(self, *wavelengths, out_of_bounds = "extrapolate"):
        if not hasattr(self, "mono_luminosity"):
            self.mono_luminosity = pd.DataFrame()
        for wavelength in wavelengths:
            self.mono_luminosity[f"L_{str(wavelength)}"] =  lb.monochromatic_lum(self.luminosity, 
                                                                           wavelength, out_of_bounds = out_of_bounds)

    def get_M_i(self):
        #Equation 4 from Richards 2006
        if not hasattr(self, "mono_luminosity"):
            self.get_luminosity(2500)
        if not "L_2500" in self.mono_luminosity.columns:
             self.get_luminosity(2500)
        self.M_i = ( -2.5 * np.log10(self.mono_luminosity["L_2500"] * 2500 / 2.998e18) + 2.5 * np.log10(4*np.pi)
                     + 5 * np.log10(10 * constants.pc.cgs.value) - 2.5 * np.log10(1+2) -48.6)
                      

    def get_colors(self):
        self.all_colors = pd.DataFrame()
        for band_1, band_2 in zip(self.my_filters[:-1],self.my_filters[1:]):
            self.all_colors[f"{band_1}-{band_2}"] = self.magnitudes[band_1]- self.magnitudes[band_2]

    def clean_colors(self,  wavelength_min = 912, wavelength_max = 3e4, clear_all_bin = True):
        """
        Remove colors when one of the bands fall below the lyman alpha ore above 3 micron
        clear_all_bin: Non rimuove gli oggetti singoli ma tutti quelli contenuti nel bin
        (se un bin è in parte dentro e in parte fuori gli intervalli di redshift)
        """

        def where_to_ignore(filter_1, filter_2,  wavelength_min,   wavelength_max):
             redshift_max = min(filter_1.wav_min/wavelength_min-1, filter_2.wav_min/wavelength_min-1)
             redshift_min = max(filter_1.wav_max/wavelength_max-1, filter_2.wav_max/wavelength_max-1)
             return redshift_min, redshift_max

        self.colors = self.all_colors.copy(deep = True)

        if clear_all_bin:
            for i, color in enumerate(self.colors):
                filter_1, filter_2 = self.filters[i], self.filters[i+1]
                z_min, z_max = where_to_ignore(filter_1, filter_2, wavelength_min, wavelength_max)
                for n_bin, (z_low, z_high) in enumerate(zip(self.redshift_bins[:-1], self.redshift_bins[1:])):
                    if np.logical_or(z_low < z_min, z_high > z_max):
                        self.colors[color][self.bin == n_bin +1 ] = np.nan
                if self.redshift_bins[-1] > z_max:   #for cycle miss the last bin 
                    self.colors[color][self.bin == self.Nbins] = np.nan
                    
        else:
            for i, color in enumerate(self.colors):
                filter_1, filter_2 = self.filters[i], self.filters[i+1]
                z_min, z_max = where_to_ignore(filter_1, filter_2, wavelength_min, wavelength_max)
                self.colors[color][np.logical_or(self.redshift < z_min, self.redshift>z_max)] = np.nan

    def assign_bin(self, right = False):
        self.bin = np.digitize(self.redshift, self.redshift_bins, right == right)
    
    def get_bin_mean(self, clipping_sigma = 3, statistic = "mean", clipped_colors = True):
        #axis = 0 redshift bins
        #axis = 1 colors
        #axis = 3 mean redshift, mean color, redshift sigma, color variance

        if not hasattr(self, "bin"):
            self.assign_bin()


        self.all_mean_values = np.zeros((self.Nbins, len(self.colors.columns), 4))
        self.mean_M_i = np.zeros(self.Nbins)

        if clipped_colors:
            self.clipped_colors = pd.DataFrame(columns = self.colors.columns, 
                                               index = np.arange(0,len(self.redshift)))
            
        
        for i in range(1, self.Nbins+1):

            logic = self.bin == i
            
            redshift = self.redshift[logic]
            colors = self.colors[logic]
            masked_M_i = sigma_clip(self.M_i[logic], copy = True, 
                                    sigma = clipping_sigma)
            if statistic == "mean": 
                self.mean_M_i[i-1] = masked_M_i.mean()
            elif statistic == "median":
                self.mean_M_i[i-1] = np.ma.median(masked_M_i)
            else:
                raise Exception("statistic should be mean or median")



            for j, color in enumerate(colors):
                masked_color = sigma_clip(colors[color], copy = True, 
                                          sigma= clipping_sigma)
                masked_redshift = np.ma.array(redshift, mask = masked_color.mask)
                
                if statistic == "mean":
                    self.all_mean_values[i-1, j, 0] = masked_redshift.mean()
                    self.all_mean_values[i-1, j, 2] = masked_redshift.std()
                    self.all_mean_values[i-1, j, 1] = masked_color.mean()
                    self.all_mean_values[i-1, j, 3] = masked_color.std()
                
                elif statistic == "median":
                    self.all_mean_values[i-1, j, 0] = np.ma.median(masked_redshift)
                    self.all_mean_values[i-1, j, 2] = np.ma.median(np.ma.abs(masked_redshift-
                                                                             np.ma.median(masked_redshift)))
                    self.all_mean_values[i-1, j, 1] = np.ma.median(masked_color)
                    self.all_mean_values[i-1, j, 3] = np.ma.median(np.ma.abs(masked_color-
                                                                             np.ma.median(masked_color)))

                
                if clipped_colors:
                    self.clipped_colors[color][logic] = masked_color.filled(np.nan)
        
        return None
#

    def get_table_mean(self, replace = False):
        self.mean_values = pd.DataFrame()
        self.mean_values["Redshift"] = np.nanmedian(self.all_mean_values[:, :, 0], axis = 1)
        for i, color in enumerate(self.colors):
            self.mean_values[color] = self.all_mean_values[:, i, 1]
            self.mean_values[f"err_{color}"] = self.all_mean_values[:, i, 3]
            if replace:
                self.mean_values[color] = self.mean_values[color].replace(np.nan, 0)
                self.mean_values[f"err_{color}"] = self.mean_values[f"err_{color}"].replace(np.nan, -99)
        
    

    def get_fitting_arrays(self, scale_sigma = 1):
        self.colors_to_fit = [f"{band1}-{band2}" for band1,band2 in zip(self.filters_to_fit[:-1], self.filters_to_fit[1:])]
        self.x_fit =  np.nanmedian(self.all_mean_values[:, :, 0], axis = 1)
        self.y_fit = np.zeros((len(self.mean_values), len(self.colors.columns)))
        self.yerr_fit = np.zeros_like(self.y_fit)
        
        for i, _ in enumerate(self.colors.columns):
            self.y_fit[:,i] = self.all_mean_values[:, i, 1]
            self.yerr_fit[:,i] = scale_sigma*self.all_mean_values[:, i, 3]

        self.y_fit = pd.DataFrame(self.y_fit, columns = self.colors.columns)
        self.yerr_fit = pd.DataFrame(self.yerr_fit, columns = self.colors.columns)
        self.y_fit = self.y_fit[self.colors_to_fit].to_numpy()
        self.yerr_fit = self.yerr_fit[self.colors_to_fit].to_numpy()
        for color in self.colors_to_fit:
            print(color)
        return None
        
    
    
    def get_scipy_fitting(self):
        
        self.scipy_args = (self.x_fit, self.y_fit, self.yerr_fit,      
                           self.mean_M_i, self.parameters, self.temple_filters, self.logic_colors_to_fit)
        
        self.scipy_results = minimize(least_square, self.temple_best, args = self.scipy_args)
        
    
    def print_results(self):
        if hasattr(self, "scipy_results"):
            for i, name in enumerate(self.parameters):
                print(f"{name} : {self.scipy_results.x[i]}")
   

        
    def emcee_fitting(self, filename = "fitting", directory = "MC_chains",
                      add_date = True,
                      nwalkers = 200, nsteps = 750, starting_noise = 1e-3):
        
        self.emcee_args = (self.x_fit, self.y_fit, self.yerr_fit,      
                           self.mean_M_i, self.parameters, self.temple_filters, 
                           self.logic_colors_to_fit, self.flat_priors)
        
        coordinates = self.scipy_results.x + starting_noise * np.random.randn(nwalkers, len(self.parameters))
        
        _, ndim = coordinates.shape

        if not os.path.isdir(directory):
            os.mkdir(directory)

        if add_date:
            filename = f"{filename}_{self.today}.h5"
        else:
            filename = f"{filename}.h5"
        
        backend = emcee.backends.HDFBackend(os.path.join(directory,filename))
        backend.reset(nwalkers, ndim)

        from multiprocessing import Pool
        with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                             args = self.emcee_args, backend=backend, pool=pool)
            self.sampler.run_mcmc(coordinates, nsteps, progress = True);



    def plot_colors(self, savename = "",
                    directory = "Plot", 
                    all_colors = True, scale_sigma = 1,
                    add_date = True):
        
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if add_date:
            directory = os.path.join(directory, f"{self.today}")
            if not os.path.isdir(directory):
                os.mkdir(directory)

        formato = ".png"
        for i, color in enumerate(self.colors):
            fig, ax = plt.subplots()
            if all_colors:
                 ax.scatter(self.redshift, self.all_colors[color], c = "skyblue", s=1)
            
            if hasattr(self, "clipped_colors"):
                ax.scatter(self.redshift, self.clipped_colors[color], c = "teal", s = 1)
            else:
                ax.scatter(self.redshift, self.colors[color], c = "teal", s = 1)

            if hasattr(self, "all_mean_values"):
                ax.errorbar(self.all_mean_values[:, i, 0], self.all_mean_values[:, i, 1], 
                            yerr = scale_sigma*self.all_mean_values[:, i, 3], 
                            ls ="none", marker = 'o', c='k')
                ymin, ymax = np.nanmin(self.all_mean_values[:, i, 1])-1, np.nanmax(self.all_mean_values[:, i, 1])+1 
                ax.set_ylim(ymin, ymax)
        
            ax.set_xlim(0, np.nanmax(self.redshift))
            ax.set_xlabel("Redshift")
            ax.set_ylabel(color)
            plt.savefig(os.path.join(directory, savename+color+formato), bbox_inches = "tight")

    
    def plot_M_i(self, savename = "M_i", directory = "Plot", plot_temple = False, add_date = True):
        
        if not os.path.isdir(directory):
            os.mkdir(directory)
        
        if add_date:
            directory = os.path.join(directory, f"{self.today}")
            if not os.path.isdir(directory):
                os.mkdir(directory)
        
        formato = ".png"

        ymin ,ymax = np.nanmin(self.M_i), np.nanmax(self.M_i)
        fig, ax = plt.subplots()
        ax.scatter(self.redshift, self.M_i, c = "teal", s = 1)
            
        if hasattr(self, "mean_values"):
            ax.errorbar(self.mean_values["Redshift"], self.mean_M_i,
                            ls ="none", c = "k", marker ="o")
        
        if plot_temple:
            
            M_i_temple = np.loadtxt("input/Mi_temple.dat", skiprows = 1)
            ax.scatter(M_i_temple[:,0], M_i_temple[:,1], marker = "p", c = "r", s = 25, zorder = 5)
            
        
        ax.set_xlabel("Redshift")
        ax.set_ylabel("M_i")
        ax.set_ylim(ymin ,ymax)
        plt.savefig(os.path.join(directory, savename+formato), bbox_inches = "tight")

    def plot_redshift_distributions(self, savename = "Redshift_distribution",
                                  directory = "Plot", **kwargs):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        format = ".png"
        fig, ax = plt.subplots()
        ax.hist(self.redshift, bins = self.redshift_bins, **kwargs)
        ax.set_xlabel("Redshift")
        plt.savefig(os.path.join(directory, savename+format), format ="png", bbox_inches = "tight")


    def plot_scipy_fitting(self, savename = "Least_squares", directory = "Plot",
                            normalization = 2500, krawczyk = True, add_date = True, 
                            **kwargs):
        
        if not os.path.isdir(directory):
            os.mkdir(directory)
        
        if add_date:
            directory = os.path.join(directory, f"{self.today}")
            if not os.path.isdir(directory):
                os.mkdir(directory)
        
        sed = get_temple_sed(0, self.scipy_results.x, self.parameters)
        wav, flux = sed.wavlen, sed.flux*sed.wavlen
        flux = flux/np.interp(normalization, wav, flux)
        formato = ".png"
        fig, ax = plt.subplots()
        if krawczyk:
            sed_krawczyk = lb.get_sed(normalization = [normalization,1])
            ax.plot(sed_krawczyk[:,0], sed_krawczyk[:,1], label = "Krawczyk+13")
        ax.plot(wav, flux, label = "Least squares", **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(900, 50000)
        ax.set_ylim(0.1, 10)
        ax.set_xlabel("Wavelength [A°]")
        ax.set_ylabel("L [Arbitrary]")
        ax.legend()
        plt.savefig(os.path.join(directory, savename+formato), format ="png", bbox_inches = "tight")
    
    def get_observational_errors(self):
        errors_col = [i for i in self.all_magnitudes.columns if "e_" in i]
        self.color_errors = pd.DataFrame()
        for band1, band2 in zip(errors_col[:-1], errors_col[1:]):
            self.color_errors[f"{band1}-{band2}"] = np.sqrt(self.all_magnitudes[band1]**2 + 
                    self.all_magnitudes[band2]**2)

        self.obs_err = np.zeros((self.Nbins,len(self.color_errors.columns))) 
        for i in range(1, self.Nbins+1):
            logic = self.bin == i
            self.obs_err[i-1,:] = np.mean(self.color_errors[logic].to_numpy(), axis = 0)
        return None
    

    