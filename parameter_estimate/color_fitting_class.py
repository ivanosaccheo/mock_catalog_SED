import os 
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units, constants
from astropy.cosmology import FlatLambdaCDM
from astropy.stats import sigma_clip
import emcee 
from scipy.optimize import minimize
from my_functions import library as lb
from qsogen.model_colours import get_colours
from qsogen.qsosed import Quasar_sed


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


def get_temple_sed(redshift, parameters, parameters_name, wavlen= np.logspace(2.96, 4.48, 1000)):
    par_dictionary = dict(zip(parameters_name, parameters))
    par_dictionary.update({"fragal" : 0, "gplind" : 0})
    return Quasar_sed(redshift, wavlen = wavlen, **par_dictionary)

def get_temple_colors(redshift, M_i, theta, parameters_name, filters):
    par_dictionary = dict(zip(parameters_name, theta))
    if not "fragal" in parameters_name:
        par_dictionary.update({"fragal" : 0, "gplind" : 0})
    model = get_colours(redshift, M_i = M_i, filters = filters, **par_dictionary)
    model[~np.isfinite(model)] = 0

    return model

def chi_square(y_model, y, yerr):
    return np.sum((y-y_model)**2/yerr**2)

def least_square(theta, redshift, y, yerr, M_i, parameters_name, filters):
    model = get_temple_colors(redshift, M_i, theta, parameters_name, filters)
    return chi_square(model, y, yerr)

def log_prior(theta, priors):
    if np.logical_and.reduce(np.logical_and(theta>=priors[:,0], theta<=priors[:,1])):
        return 0
    return -np.inf

def log_probability(theta, redshift, y, yerr, M_i, parameters_name, filters, priors):
    
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, redshift, y, yerr, M_i, parameters_name, filters)
    
    return lp + ll

def log_likelihood(theta, redshift, y, yerr, M_i, 
                   parameters_name, filters):
    
    model = get_temple_colors(redshift, M_i, theta, parameters_name, filters)
    sigma2 = yerr**2 
    
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))






class organize_colors:

    def __init__(self, filename = "krawczyk_no_host_magnitudes.csv", 
                 directory = os.path.expanduser("~/DATA/samples"),
                 **kwargs) -> None:
        self.all_magnitudes = pd.read_csv(os.path.join(directory, filename), **kwargs)
        self.all_redshift = self.all_magnitudes["Redshift"].to_numpy()
        self.today = datetime.date.today()
    
    def get_filters_name(self, filename = "lista_filtri.dat", directory = "input", 
                         my_name_header = "myname", 
                         temple_name_header= "temple_name",
                         necessary_header = "necessary"):

    
        self.filter_df = pd.read_csv(os.path.join(directory, filename), comment="#", 
                                     delim_whitespace= True)
        self.my_filters = self.filter_df[my_name_header].to_list()
        for filtro in self.my_filters: assert(filtro in self.all_magnitudes.columns)
        
        self.temple_filters = self.filter_df[temple_name_header].to_list()
        self.necessary_filters = self.filter_df[my_name_header][self.filter_df[necessary_header]].to_list()


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
        self.filters = [self.filters[idx] for idx in order_filters(self.filters)]
        
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
            vega_to_AB = [0, 0, 0, 0, 0,   #SDSS are AB
                          0.6147726766033728, 0.9152606087248839, 1.3520471317250795,  1.8713980229183953, #UKIDSS
                          2.6787715820675913, 3.3151986991103968] #WISE see Temple+21
            print("Transforming temporarily magnitudes from Vega to AB")
            for col, conversion in zip(self.magnitudes.columns, vega_to_AB):
                print(col, conversion)
                self.magnitudes[col] += conversion
        
        for i, col in enumerate(self.magnitudes):
            L[:,i] = (10**(-0.4*(self.magnitudes[col]+48.6))) * 2.998e18/self.filters[i].wav
            L[:,i] = L[:,i] * 4 *np.pi * self.luminosity_distance**2
        self.luminosity = np.zeros((self.magnitudes.shape[0], self.magnitudes.shape[1], 2))
        
        for i in range(L.shape[1]):
            self.luminosity[:,i,0] = self.filters[i].wav/(self.redshift+1)
            self.luminosity[:,i,1] = L[:,i]
        
        if magnitudes_are_vega:
            print("Transforming magnitudes back to Vega")
            for col, conversion in zip(self.magnitudes.columns, vega_to_AB):
                self.magnitudes[col] -= conversion   #back to vega
       
    
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

    
    def plot_colors(self, savename = "",
                    directory = "Plot", 
                    all_colors = True, **kwargs):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        format = ".png"
        for i, color in enumerate(self.colors):
            ymin ,ymax = self.colors[color].min(), self.colors[color].max()
            fig, ax = plt.subplots()
            if all_colors:
                 ax.scatter(self.redshift, self.all_colors[color], c = "skyblue", s=1)
            ax.scatter(self.redshift, self.colors[color], c = "teal", s = 1)
            for cut in self.redshift_bins:
                ax.plot([cut, cut], [ymin, ymax], 'k:', lw = 0.5)
            if hasattr(self, "all_mean_values"):
                ax.errorbar(self.all_mean_values[:, i, 0], self.all_mean_values[:, i, 1], 
                            yerr =self.all_mean_values[:, i, 3], ls ="none", marker = 'o', c='k')
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, np.max(self.redshift))
            ax.set_xlabel("Redshift")
            ax.set_ylabel(color)
            plt.savefig(os.path.join(directory, savename+color+format), bbox_inches = "tight")

    def assign_bin(self, right = False):
        self.bin = np.digitize(self.redshift, self.redshift_bins, right == right)
    
    def plot_redshift_distributions(self, savename = "Redshift_distribution",
                                  directory = "Plot", **kwargs):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        format = ".png"
        fig, ax = plt.subplots()
        ax.hist(self.redshift, bins = self.redshift_bins, **kwargs)
        ax.set_xlabel("Redshift")
        plt.savefig(os.path.join(directory, savename+format), format ="png", bbox_inches = "tight")
    
    def get_bin_mean(self, clipping_sigma = 3):
        #axis = 0 redshift bins
        #axis = 1 colors
        #axis = 3 mean redshift, mean color, redshift sigma, color variance
        self.all_mean_values = np.zeros((self.Nbins, len(self.colors.columns), 4))
        self.mean_M_i = np.zeros(self.Nbins)
        for i in range(1,self.Nbins+1):
            
            redshift = self.redshift[self.bin == i]
            colors = self.colors[self.bin==i]
            masked_M_i = sigma_clip(self.M_i[self.bin == i], copy = True, 
                                    sigma = clipping_sigma)
            self.mean_M_i[i-1] = masked_M_i.mean()

            for j, color in enumerate(colors):
                masked_color = sigma_clip(colors[color], copy = True, 
                                          sigma= clipping_sigma)
                masked_redshift = np.ma.array(redshift, mask = masked_color.mask)
                self.all_mean_values[i-1, j, 0] = masked_redshift.mean()
                self.all_mean_values[i-1, j, 2] = masked_redshift.std()
                self.all_mean_values[i-1, j, 1] = masked_color.mean()
                self.all_mean_values[i-1, j, 3] = masked_color.std()
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
        
    def plot_M_i(self, savename = "M_i", directory = "Plot", plot_temple = False, **kwargs):
        
        if not os.path.isdir(directory):
            os.mkdir(directory)
        
        format = ".png"

        ymin ,ymax = np.min(self.M_i), np.max(self.M_i)
        fig, ax = plt.subplots()
        ax.scatter(self.redshift, self.M_i, c = "teal", s = 1)
        for cut in self.redshift_bins:
            ax.plot([cut, cut], [ymin, ymax], 'k:', lw = 0.5)
            
        if hasattr(self, "mean_values"):
            ax.errorbar(self.mean_values["Redshift"], self.mean_M_i,
                            ls ="none", c = "k", marker ="o")
        
        if plot_temple:
            
            M_i_temple = np.loadtxt("input/Mi_temple.dat")
            ax.scatter(M_i_temple[:,0], M_i_temple[:,1], marker = "p", c = "r", s = 25, zorder = 5)
            
        
        ax.set_xlabel("Redshift")
        ax.set_ylabel("M_i")
        ax.set_ylim(ymin ,ymax)
        plt.savefig(os.path.join(directory, savename+format), bbox_inches = "tight")

    def get_fitting_arrays(self, replace = True, scale_sigma = 1):
        self.x_fit =  np.nanmedian(self.all_mean_values[:, :, 0], axis = 1)
        self.y_fit = np.zeros((len(self.mean_values), len(self.colors.columns)))
        self.yerr_fit = np.zeros_like(self.y_fit)
        for i, _ in enumerate(self.colors.columns):
            self.y_fit[:,i] = self.all_mean_values[:, i, 1]
            self.yerr_fit[:,i] = scale_sigma*self.all_mean_values[:, i, 3]
        if replace:
            self.y_fit[np.isnan(self.y_fit)] = 0
            self.yerr_fit[np.isnan(self.yerr_fit)] = 99
    
    
    def get_scipy_fitting(self):
        
        self.scipy_args = (self.x_fit, self.y_fit, self.yerr_fit,      
                           self.mean_M_i, self.parameters, self.temple_filters)
        
        self.scipy_results = minimize(least_square, self.temple_best, args = self.scipy_args)
        
    def plot_scipy_fitting(self, savename = "Least_squares", directory = "Plot",
                            normalization = 2500, krawczyk = True,  **kwargs):
        
        if not os.path.isdir(directory):
            os.mkdir(directory)
        sed = get_temple_sed(0, self.scipy_results.x, self.parameters)
        wav, flux = sed.wavlen, sed.flux*sed.wavlen
        flux = flux/np.interp(normalization, wav, flux)
        format = ".png"
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
        plt.savefig(os.path.join(directory, savename+format), format ="png", bbox_inches = "tight")

    def print_results(self):
        if hasattr(self, "scipy_results"):
            for i, name in enumerate(self.parameters):
                print(f"{name} : {self.scipy_results.x[i]}")
   

        
    def emcee_fitting(self, filename = "fitting", directory = "MC_chains",
                      add_date = True,
                      nwalkers = 200, nsteps = 750, starting_noise = 1e-3):
        
        self.emcee_args = (self.x_fit, self.y_fit, self.yerr_fit,      
                           self.mean_M_i, self.parameters, self.temple_filters, self.flat_priors)
        
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

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                             args = self.emcee_args, backend=backend)

        self.sampler.run_mcmc(coordinates, nsteps, progress = True);

