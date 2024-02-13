#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slightly changed version of Quasar_Sed by Temple+21 to adapt it to the mock catalog
"""
import numpy as np
from scipy.integrate import quad
from astropy.convolution import Gaussian1DKernel, convolve
from astropy import constants, units as u
from copy import deepcopy

_c_ = 299792458.0   # speed of light in m/s





def pl(wavlen, plslp, const):
    """Define power-law in flux density per unit frequency."""
    return const*wavlen**plslp


def bb(tbb, wav):
    """Blackbody shape in flux per unit frequency.
    -----
    h*c/k_b = 1.43877735e8 KelvinAngstrom
    """
    return (wav**(-3))/(np.exp(1.43877735e8 / (tbb*wav)) - 1.0)


class Quasar_sed:
    """Construct an instance of the quasar SED model.

    Attributes
    ----------
    lum_dens : ndarray
        Luminosity per unit frequency from total SED, i.e. quasar plus host galaxy.
    
    lum :  ndarray
        Luminosity from total SED, i.e. quasar plus host galaxy.
    
    host_galaxy_lum_dens : ndarray
        Luminosity per unit frequency from host galaxy component of the model SED.
    
    wavlen : ndarray
        Wavelength array in the rest frame.
    
    M_i : float
        Intrinsic (i.e. not reddened) absolute i magnitude at z = 2
    
    unextincted_luminosity_density : ndarray
        Luminosity per unit wavelength from AGN before reddening
        
    Lbol : float
        Bolometric luminosities between 12.4 A° and 1e4 A° by default
    



    """
    def __init__(self,
                 LogL2500,
                 AGN_type = 1,
                 ebv = 0,
                 physical_units = False,
                 add_infrared = True,
                 wavlen=np.logspace(2.7, 4.48, num=20001, endpoint=True),
                 LogL2kev = None,
                 emline_scatter = 0,
                 bbnorm_scatter = 0,
                 add_NL = True,
                 gflag = False,
                 **kwargs):
        """Initialises an instance of the Quasar SED model.

        Parameters
        ----------
        LogL2500 : float
            Monochromatic luminosity at 2500A [erg/s Hz^{-1}] of (unreddened) quasar model. 
            The one given by Lusso+10 formula
        
        AGN_type : int, 1 or 2 
             , AGN type to assign the correct emission lines.
        
        ebv : float, optional
            Extinction E(B-V) applied to quasar model. Not applied to galaxy
            component. Default is zero.

        physical_units : Bool, optional
            If true the attributes are stored as astropy.quantity array
        
        add_infrared : Bool, optional
            If True the SED is prolonged in the IR using Krawczyk+13 mean SED
  
      
        wavlen : ndarray, optional
            Rest-frame wavelength array. Default is log-spaced array covering
            ~500 to 30000 Angstroms. `wavlen` must be monotonically increasing,
            and if gflag==True, `wavlen` must cover 4000-5000A to allow the
            host galaxy component to be properly normalised.
            
        LogL2kev : float, optional
            Monochromatic luminosity at 2 keV [erg/s Hz^{-1}]. 
            Necessary only to compute the bolometric luminosity
        
        emline_scatter : float, optional,
            Scatter to add to emline_type parameter
            emline_type_new ~N(emline_type, emline_scatter)
        
        bbnorm_scatter : float, optional
            Same as emline_scatter but for bbnorm parameter



        Other Parameters
        ----------------
        tbb : float, optional
            Temperature of hot dust blackbody in Kelvin.
        bbnorm : float, optional
            Normalisation, relative to power-law continuum at 2 micron, of the
            hot dust blackbody.
        scal_emline : float, optional
            Overall scaling of emission line template. Negative values preserve
            relative equivalent widths while positive values preserve relative
            line fluxes. Default is -1.
        emline_type : float, optional
            Type of emission line template. Minimum allowed value is -2,
            corresponding to weak, highly blueshifed lines. Maximum allowed is
            +3, corresponding to strong, symmetric lines. Zero correspondes to
            the average emission line template at z=2, and -1 and +1 map to the
            high blueshift and high EW extrema observed at z=2. Default is
            None, which uses `beslope` to scale `emline_type` as a smooth
            function of `M_i`.
        scal_halpha, scal_lya, scal_nlr : float, optional
            Additional scalings for the H-alpha, Ly-alpha, and for the narrow
            optical lines. Default is 1.
        beslope : float, optional
            Baldwin effect slope, which controls the relationship between
            `emline_type` and luminosity `M_i`.
        bcnorm : float, optional
            Balmer continuum normalisation. Default is zero as default emission
            line templates already include the Balmer Continuum.
        gflag : bool, optional
            Flag to include host-galaxy emission. Default is True.
        fragal : float, optional
            Fractional contribution of the host galaxy to the rest-frame 4000-
            5000A region of the total SED, for a quasar with M_i = -23.
        gplind : float, optional
            Power-law index dependence of galaxy luminosity on M_i.
        emline_template : array, optional
            Emission line templates. Array must have structure
            [wavelength, average lines, reference continuum,
            high-EW lines, high-blueshift lines, narrow lines]
        reddening_curve : array, optional
            Quasar reddening law.
            Array must have structure [wavelength lambda, E(lambda-V)/E(B-V)]
        galaxy_template : array, optional
            Host-galaxy SED template.
            Array must have structure [lambda, f_lambda].
            Default is an S0 galaxy template from the SWIRE library.

        """
        self.wavlen = wavlen
        if np.any(self.wavlen[:-1] > self.wavlen[1:]):
            raise Exception('wavlen must be monotonic')
        
        self.lum_dens = np.zeros_like(self.wavlen)
        self.host_galaxy_lum_dens = np.zeros_like(self.wavlen)
        self.ebv = ebv
        self.LogL2500 = LogL2500
        self.AGN_type = AGN_type 
        self.M_i = -2.5 * self.LogL2500 -48.6 + 2.5*np.log10(4*np.pi) + 5 *       np.log10(constants.pc.cgs.value) 
        self.LogL2kev = LogL2kev
        self.add_parameters(**kwargs)
        self.bbnorm += np.random.normal(loc = 0, scale = bbnorm_scatter)

        #######################################################
        # READY, SET, GO!
        #######################################################

        self.set_continuum()
        self.add_blackbody()
        
        if self.bcnorm:
            self.add_balmer_continuum()
        
        
        self.normalize(flxnrm = (10**self.LogL2500), wavnrm = 2500)
        
        self.convert_fnu_flambda()
        
        if self.AGN_type == 1:
            self.add_emission_lines_type_1(emline_scatter = emline_scatter)
        
        self.unextincted_luminosity_density = deepcopy(self.lum_dens.astype("float64")) # erg/s A^-1
      
        self.compute_Lbol(wavlen_min = 12.4, wavlen_max = 1e4)

        if self.ebv:
            self.redden_spectrum()
        
        if add_NL:
            self.add_emission_lines_type_2()

        self.lum_dens = self.lum_dens.astype("float64") #non capisco perchè 

        if add_infrared:
            self.add_infrared(wavlen_cut = 2.5e4)


        # add in host galaxy flux
        if gflag:
            print("Warning, adding host galaxy")
            self.host_galaxy()
            self.lum_dens += self.host_galaxy_lum_dens
            self.host_galaxy_lum_dens *= (self.wavlen*self.wavlen/2.998e18)


        self.convert_flambda_fnu()
       

        self.lum = self.lum_dens * (2.998e18/self.wavlen)
        
        if physical_units:
            self.lum *= u.erg/u.s
            self.wavlen *= u.angstrom
            self.lum_dens *= u.erg/(u.s * u.Hz)
            self.host_galaxy_lum_dens *= u.erg/(u.s * u.Hz)
        
        return None
        

    def wav2num(self, wav):
        """Convert a wavelength to an index."""
        return np.argmin(np.abs(self.wavlen - wav))

    def wav2flux(self, wav):
        """Convert a wavelength to a flux.

        Different from self.lum_dens[wav2num(wav)], as wav2flux interpolates in an
        attempt to avoid problems when wavlen has gaps. This mitigation only
        works before the emission lines are added to the model, and so wav2flux
        should only be used with a reasonably dense wavelength array.
        """
        return np.interp(wav, self.wavlen, self.lum_dens)

    def set_continuum(self, flxnrm=1.0, wavnrm=5500):
        """Set multi-powerlaw continuum in flux density per unit frequency."""
        # Flip signs of powerlaw slopes to enable calculation to be performed
        # as a function of wavelength rather than frequency
        sl1 = -self.plslp1
        sl2 = -self.plslp2
        wavbrk1 = self.plbrk1

        # Define normalisation constant to ensure continuity at wavbrk
        const2 = flxnrm/(wavnrm**sl2)
        const1 = const2*(wavbrk1**sl2)/(wavbrk1**sl1)

        # Define basic continuum using the specified normalisation fnorm at
        # wavnrm and the two slopes - sl1 (<wavbrk) sl2 (>wavbrk)
        fluxtemp = np.where(self.wavlen < wavbrk1,
                            pl(self.wavlen, sl1, const1),
                            pl(self.wavlen, sl2, const2))

        # Also add steeper power-law component for sub-Lyman-alpha wavelengths
        sl3 = sl1 - self.plstep
        wavbrk3 = self.plbrk3
        # Define normalisation constant to ensure continuity
        const3 = const1*(wavbrk3**sl1)/(wavbrk3**sl3)

        self.lum_dens = np.where(self.wavlen < wavbrk3,
                             pl(self.wavlen, sl3, const3),
                             fluxtemp)

    def add_blackbody(self, wnorm = 20000.):
        """Add basic blackbody spectrum to the flux distribution."""
        bbnorm = self.bbnorm  # blackbody normalisation at wavelength wnorm
        tbb = self.tbb

        if bbnorm > 0:

            bbval = bb(tbb, wnorm)
            cmult = bbnorm / bbval
            bb_flux = cmult*bb(tbb, self.wavlen)
            self.lum_dens += bb_flux

    def add_balmer_continuum(self,
                             tbc=15000., taube=1., wavbe=3646.,
                             wnorm=3000., vfwhm=5000.):
        """Add Balmer continuum emission to the model.

        Prescription from Grandi 1982ApJ...255...25G.

        Parameters
        ----------
        tbc
            BC temperature in Kelvin.
        taube
            The optical depth at wavelength wavbe, the Balmer edge.
        bcnorm
            Normalisation of the BC at wavelength wnorm Angstroms.
        """
        fnorm = self.bcnorm

        flux_bc = np.zeros_like(self.lum_dens)

        nuzero = _c_/(wavbe*1.0e-10)  # frequency of Balmer edge
        # calculate required normalisation constant at wavelength wnorm

        bbval = bb(tbc, wnorm)
        nu = _c_/(wnorm*1.0e-10)
        tau = taube * (nuzero/nu)**3    # tau is the optical depth at wnorm
        if tau < 50:
            bbval = bbval * (1.0 - np.exp(-tau))
        cmult = fnorm/bbval

        nu = _c_ / self.wavlen
        tau = taube * np.power(nuzero/nu, 3)
        scfact = np.ones(len(flux_bc), dtype=np.float64)
        scfact[tau <= 50.0] = 1.0 - np.exp(-tau[tau <= 50.0])
        bwav = tuple([self.wavlen < wavbe])
        flux_bc[bwav] = cmult * scfact[bwav] * bb(tbc, self.wavlen[bwav])

        # now broaden bc to simulate effect of bulk-velocity shifts
        vsigma = vfwhm / 2.35
        wsigma = wavbe * vsigma*1e3 / _c_  # change vsigma from km/s to m/s
        winc = (self.wavlen[self.wav2num(wnorm)]
                - self.wavlen[self.wav2num(wnorm) - 1])
        psigma = wsigma / winc     # winc is wavelength increment at wnorm
        gauss = Gaussian1DKernel(stddev=psigma)
        flux_bc = convolve(flux_bc, gauss)
        # Performs a Gaussian smooth with dispersion psigma pixels

        # Determine height of power-law continuum at wavelength wnorm to
        # allow correct scaling of Balmer continuum contribution
        self.lum_dens += flux_bc*self.wav2flux(wnorm)

    def convert_fnu_flambda(self):
        """Convert f_nu to f_lamda, using 1/lambda^2 conversion.
        """
        self.lum_dens = self.lum_dens*self.wavlen**(-2)*2.998e18
    
    def convert_flambda_fnu(self):
        """Convert f_lambda to f_nu, using 1/lambda^2 conversion.
        """
        self.lum_dens = (self.lum_dens*self.wavlen**(2))/2.998e18
        
    def normalize(self, flxnrm, wavnrm=2500):
        self.lum_dens = flxnrm * self.lum_dens/self.wav2flux(wavnrm)
    
        
        

    def add_emission_lines_type_1(self, emline_scatter = 0, wavnrm=5500, wmin=6000, wmax=7000):
        """Add emission lines to the model SED.

        Emission-lines are included via 4 emission-line templates, which are
        packaged with a reference continuum. One of these templates gives the
        average line emission for a M_i=-27 SDSS DR16 quasar at z~2. The narrow
        optical lines have been isolated in a separate template to allow them
        to be re-scaled if necesssary. Two templates represent the observed
        extrema of the high-ionisation UV lines, with self.emline_type
        controlling the balance between strong, peaky, systemic emission and
        weak, highly skewed emission. Default is to let this vary as a function
        of redshift using self.beslope, which represents the Baldwin effect.
        The template scaling is specified by self.scal_emline, with positive
        values producing a scaling by intensity, whereas negative values give a
        scaling that preserves the equivalent-width of the lines relative
        to the reference continuum template. The facility to scale the H-alpha
        line by a multiple of the overall emission-line scaling is included
        through the parameter scal_halpha, and the ability to rescale the
        narrow [OIII], Hbeta, etc emission is included through scal_nlr.
        """
        scalin = self.scal_emline
        scahal = self.scal_halpha
        scalya = self.scal_lya
        beslp = self.beslope
        benrm = self.benorm

        if self.emline_type is None:
            if beslp:
                vallum = self.M_i
                self.emline_type = (vallum - benrm)*beslp
            else:
                self.emline_type = 0.  # default median emlines

        varlin = self.emline_type
        varlin += np.random.normal(loc=0, scale = emline_scatter)

        linwav, medval, conval, pkyval, wdyval, _ = self.emline_template

        if varlin == 0.:
            # average emission line template for z~2 SDSS DR16Q-like things
            linval = medval 
        elif varlin > 0:
            # high EW emission line template
            varlin = min(varlin, 3.)
            linval = varlin*pkyval + (1-varlin)*medval
        else:
            # highly blueshifted emission lines
            varlin = min(abs(varlin), 2.)
            linval = varlin*wdyval + (1-varlin)*medval 

        # remove negative dips from extreme extrapolation (i.e. abs(varlin)>>1)
        linval[(linwav > 4930) & (linwav < 5030) & (linval < 0.)] = 0.
        linval[(linwav > 1150) & (linwav < 1200) & (linval < 0.)] = 0.

        linval = np.interp(self.wavlen, linwav, linval)
        conval = np.interp(self.wavlen, linwav, conval)

        imin = self.wav2num(wmin)
        imax = self.wav2num(wmax)
        _scatmp = abs(scalin)*np.ones(len(self.wavlen))
        _scatmp[imin:imax] = _scatmp[imin:imax]*abs(scahal)
        _scatmp[:self.wav2num(1350)] = _scatmp[:self.wav2num(1350)]*abs(scalya)

        # Intensity scaling
        if scalin >= 0:
            # Normalise such that continuum flux at wavnrm equal to that
            # of the reference continuum at wavnrm
            self.lum_dens += (_scatmp * linval *
                          self.lum_dens[self.wav2num(wavnrm)] /
                          conval[self.wav2num(wavnrm)])
            # Ensure that -ve portion of emission line spectrum hasn't
            # resulted in spectrum with -ve fluxes
            self.lum_dens[self.lum_dens < 0.0] = 0.0

        # EW scaling
        else:
            self.lum_dens += _scatmp * linval * self.lum_dens / conval
            # Ensure that -ve portion of emission line spectrum hasn't
            # resulted in spectrum with -ve fluxes
            self.lum_dens[self.lum_dens < 0.0] = 0.0
    


    def add_emission_lines_type_2(self):
        
        if self.nlr_template_idx is not None:
            self.nlr_template = np.genfromtxt(self.nlr_template_list[self.nlr_template_idx], unpack=False)  
        
        else:
            self.nlr_template = np.genfromtxt(np.random.choice(self.nlr_template_list), unpack=False)

        nlr_template = np.interp(self.wavlen, self.nlr_template[:,0],self.nlr_template[:,1])

        #self.AD_luminosity = 10**(self.LogL2500+15.56778047) ##valid for al = -1.7
        self.AD_luminosity = 10**(self.LogL2500+15.67097221)  ## valid for al = -1.4
        
        self.lum_dens += (self.AD_luminosity*nlr_template)
        
        return None
        
    
    

    
    
    def host_galaxy(self, gwnmin=4000.0, gwnmax=5000.0):
        """Correctly normalise the host galaxy contribution."""

        if min(self.wavlen) > gwnmin or max(self.wavlen) < gwnmax:
            raise Exception(
                    'wavlen must cover 4000-5000 A for galaxy normalisation'
                    + '\n Redshift is {}'.format(self.z))

        fragal = min(self.fragal, 0.99)
        fragal = max(fragal, 0.0)

        if self.galaxy_template is not None:
            wavgal, flxtmp = self.galaxy_template
        else:
            # galaxy SED input file
            f3 = 'Sb_template_norm.sed'
            wavgal, flxtmp = np.genfromtxt(f3, unpack=True)

        # Interpolate galaxy SED onto master wavlength array
        flxgal = np.interp(self.wavlen, wavgal, flxtmp)
        galcnt = np.sum(flxgal[self.wav2num(gwnmin):self.wav2num(gwnmax)])

        # Determine fraction of galaxy SED to add to unreddened quasar SED
        qsocnt =np.sum(self.unextincted_luminosity_density[self.wav2num(gwnmin):self.wav2num(gwnmax)])
        # bring galaxy and quasar flux zero-points equal
        cscale = qsocnt / galcnt

        vallum = self.M_i
        galnrm = -23.   # this is value of M_i for gznorm~0.35
        # galnrm = np.interp(0.2, self.zlum, self.lumval)

        vallum = vallum - galnrm
        vallum = 10.0**(-0.4*vallum)
        tscale = vallum**(self.gplind-1)
        scagal = (fragal/(1-fragal))*tscale

        self.host_galaxy_lum_dens = cscale * scagal * flxgal

    def redden_spectrum(self, R = 3.1):
        """Redden quasar component of total SED. R=A_V/E(B-V)."""

        if self.reddening_curve is not None:
            wavtmp, flxtmp = self.reddening_curve
        else:
            # read extinction law from file
            f4 = 'pl_ext_comp_03.sph'
            wavtmp, flxtmp = np.genfromtxt(f4, unpack=True)

        extref = np.interp(self.wavlen, wavtmp, flxtmp)
        exttmp = self.ebv * (extref + R)
        self.lum_dens = self.lum_dens*10.0**(-exttmp/2.5)
    
    def add_infrared(self, wavlen_cut = 2.5e4):

        if self.LogL2500 < 30.33:
            x, y = self.ir_sed[:,0], self.ir_sed[:,2]   #Low Luminosity
        elif self.LogL2500 > 30.77:
            x, y = self.ir_sed[:,0], self.ir_sed[:,4]   #High luminosity
        else:
            x, y = self.ir_sed[:,0], self.ir_sed[:,3]   #Mid luminosity
        idx = self.wavlen >= wavlen_cut
        norm =  np.interp(wavlen_cut, self.wavlen, self.lum_dens)
        norm /= np.interp(wavlen_cut, x, y)

        self.lum_dens[idx] = norm*np.interp(self.wavlen[idx], x, y, right = np.nan)
    
    def monochromatic_luminosity(self, wavelength):
        """ Returns unextincted monochromatic luminosities at the wavelengths specified"""
        
        mono = np.interp(wavelength, self.wavlen, self.unextincted_luminosity_density) #erg/s A^-1
        
        return mono*wavelength #erg/s
    
    
    
    def get_full_SED(self, photon_index = 1.8):
        L2kev = (10**self.LogL2kev)*2.998e18/(6.2*6.2)      #Flambda
        
        xray_wav = np.logspace(-0.39, 1.1, 30) #1-30 keV
        xray_luminosity_density = (L2kev/(6.2**(photon_index-3)))*xray_wav**(photon_index-3)
        
        x0, y0 = np.log10(xray_wav[-1]), np.log10(xray_luminosity_density[-1])
        x1, y1 = np.log10(self.wavlen[0]), np.log10(self.unextincted_luminosity_density[0])
        if x1 < np.log10(500):
            print("Warning the SED extends too far in the EUV, Lbol might be overestimated")
        
        euv_wav = np.logspace(x0, x1, 30)   
        A = ((y1-y0)/(x1-x0))
        euv_luminosity_density = 10**(y0 + A*(np.log10(euv_wav)-x0))
      
        self.fullsed_wavlen = np.concatenate([xray_wav, euv_wav, self.wavlen], axis = 0)
        self.fullsed_luminosity_density = np.concatenate([xray_luminosity_density, 
                                        euv_luminosity_density, self.unextincted_luminosity_density],
                                        axis = 0)
        
        return None 
    
    def compute_Lbol(self, wavlen_min = 12.4, wavlen_max = 1e4):

        if self.LogL2kev is not None:
            self.get_full_SED()
            idx = np.logical_and(self.fullsed_wavlen>=wavlen_min, self.fullsed_wavlen<=wavlen_max)
            self.Lbol = np.trapz(self.fullsed_luminosity_density[idx], self.fullsed_wavlen[idx])
        else:
            #L2500 = 10**(self.LogL2500)*(2.998e18/2500)
            #self.Lbol= 1.85 +0.98*np.log10(L2500)   ##Runnoe+11, actually it is the 3000A° BC
            self.Lbol = 0.94661791*self.LogL2500 + 17.31730527 #Assumes Lusso+10 x-ray-to-Uv ratio
            self.Lbol = 10**self.Lbol
        
        return None
    
    
    def compute_BC(self, wavelength):
        
        return self.Lbol/self.monochromatic_luminosity(wavelength)
        
        
    def add_parameters(self, **kwargs):
        from qsogen_4_catalog.config import params
        _params = params.copy()  # avoid overwriting params dict with kwargs
        for key, value in kwargs.items():
            if key not in _params.keys():
                print('Warning: "{}" not recognised as a kwarg'.format(key))
            _params[key] = value
        self.params = _params
        
        self.plslp1 = _params['plslp1']
        self.plslp2 = _params['plslp2']
        self.plstep = _params['plstep']
        self.tbb = _params['tbb']
        self.plbrk1 = _params['plbrk1']
        self.plbrk3 = _params['plbrk3']
        self.bbnorm = _params['bbnorm']
        self.scal_emline = _params['scal_emline']
        self.emline_type = _params['emline_type']
        self.scal_halpha = _params['scal_halpha']
        self.scal_lya = _params['scal_lya']
        self.emline_template = _params['emline_template']
        self.reddening_curve = _params['reddening_curve']
        self.galaxy_template = _params['galaxy_template']
        self.ir_sed = _params['ir_sed']
        self.nlr_template_list = _params['nlr_template_list']
        self.nlr_template_idx = _params["nlr_template_idx"]

        self.beslope = _params['beslope']
        self.benorm = _params['benorm']
        self.bcnorm = _params['bcnorm']
        self.fragal = _params['fragal']
        self.gplind = _params['gplind'] 
        
        return None
        
