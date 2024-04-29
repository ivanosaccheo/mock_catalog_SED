#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
See Temple+21
"""
import numpy as np
import glob

f1 = 'qsogen_4_catalog/emission_lines.dat'
emline_template = np.genfromtxt(f1, unpack=True)
# wav, median_emlines, continuum, peaky_line, windy_lines, narrow_lines

f2 = 'qsogen_4_catalog/galaxy_template.sed'
galaxy_template = np.genfromtxt(f2, unpack=True)
# S0 galaxy template from SWIRE
# https://ui.adsabs.harvard.edu/abs/2008MNRAS.386..697R/

f3 = 'qsogen_4_catalog/reddening_curve.sph'
reddening_curve = np.genfromtxt(f3, unpack=True)
# Extinction curve, format: [lambda, E(lambda-V)/E(B-V)]
# Recall flux_reddened(lambda) = flux(lambda)*10^(-A(lambda)/2.5)
# where A(lambda) = E(B-V)*[E(lambda-V)/E(B-V) + R] 
# so taking R=3.1, A(lambda) = E(B-V)*[Col#2 + 3.1]

f4 = "qsogen_4_catalog/krawczyk_13.sed"
krawczyk_sed = np.genfromtxt(f4)

#f5 = np.random.choice(glob.glob("qsogen_4_catalog/narrow_lines/NL_templates/nlr*"))
#f5 = glob.glob("qsogen_4_catalog/narrow_lines/NL_templates/nlr*")[0]
#narrow_line_template = np.genfromtxt(f5, unpack=False)
narrow_line_template_list_feltre  = glob.glob("qsogen_4_catalog/narrow_lines/NL_templates/nlr*")
narrow_line_template_list_scaled  = glob.glob("qsogen_4_catalog/narrow_lines/NL_templates_OIII/nlr*")

# fit to DR16Q median 2sigma-clipped colours in multi-imag bins
params = dict(plslp1=-0.349,
              plslp2=0.593,
              plstep=-1.0,    # (not fit for)
              plbrk1=3880.,
              tbb=1243.6,
              plbrk3=1200,   # (not fit for)
              bbnorm=3.961,
              scal_emline=-0.9936,
              emline_type=None,
              scal_halpha=1.,
              scal_lya=1.,
              scal_nlr=2.,
              emline_template = emline_template,
              galaxy_template= galaxy_template,
              reddening_curve = reddening_curve,
              ir_sed = krawczyk_sed,
              nlr_template_list_feltre = narrow_line_template_list_feltre,
              nlr_template_list_scaled = narrow_line_template_list_scaled,
              nlr_template_idx = None, 
              Av_lines = 0,
              zlum_lumval=np.array([[0.23, 0.34, 0.6, 1.0, 1.4, 1.8, 2.2,
                                     2.6, 3.0, 3.3, 3.7, 4.13, 4.5],
                                    [-21.76, -22.9, -24.1, -25.4, -26.0,
                                     -26.6, -27.1, -27.6, -27.9, -28.1, -28.4,
                                     -28.6, -28.9]]),
              M_i=None,
              beslope=0.183,
              benorm=-27.,    # (not fit for)
              bcnorm=False,
              lyForest=True,
              lylim=912,   # (not fit for)
              gflag=False,
              fragal=0.244,
              gplind=0.684)
