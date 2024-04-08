"""
Just to create a smaller table than the original one provided by A. Feltre
"""


import pandas as pd
import numpy as np 
import os
import scipy.stats as stats




def get_reduced_table(directory = "", master_fname = "A_Feltre_master.dat",
                      logU_list = [-1.5,  -2.0, -2.5, -3.0, -3.5, -4.0],  
                      alpha_list = [-1.4,  -1.7],
                      met_list = [0.008, 0.014, 0.017, 0.020, 0.030, 0.040],
                      nh_list = [3],
                      xi_list = [0.3], 
                      normalize = "OIII_5007"):
    lines = pd.read_csv(os.path.join(directory, master_fname), delim_whitespace = True)
    u_cond = np.isin(lines["logU"], logU_list)
    met_cond = np.isin(lines["Z"], met_list)
    al_cond = np.isin(lines["al"], alpha_list)
    nh_cond = np.isin(lines["lnh"], nh_list)
    xi_cond = np.isin(lines["xi"], xi_list)

    if normalize is not None:
        lines.iloc[:, 5:] -= lines[normalize].to_numpy()[:, None]
    
    return lines[np.logical_and.reduce([u_cond, met_cond, al_cond, nh_cond, xi_cond])]


def print_templates(table,  save_directory = "NL_templates",
                    fwhm_list = [500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0],
                    wavlen = np.logspace(3, 4, 2000)):
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)


    directory = ""
    catalog_file = os.path.join(directory, "line_catalog.dat")
    lines_catalog = pd.read_csv(catalog_file, sep =' ')
    line_wavlen = lines_catalog["wavelen"].to_numpy()
    line_labels = lines_catalog["labels"].to_list()


    for i, row in table.iterrows():
        logU = row["logU"]
        Z = row["Z"]
        xi = row["xi"]
        lnh = row["lnh"]
        al = row["al"]
        line_lum = 10**row[line_labels].to_numpy()
        for fwhm in fwhm_list:
            template = get_template(wavlen,line_wavlen,
                                    line_lum, fwhm)
            template = np.vstack([wavlen.T, template.T]).T
            fname = f"nlr_{logU}_{Z}_{xi}_{lnh}_{al}_{fwhm}.dat"
            np.savetxt(os.path.join(save_directory, fname), template)





def get_template(wavlen, line_wavlen, line_lum, fwhm):
    xx = np.repeat(np.expand_dims(wavlen, axis =1), len(line_wavlen), axis =1)
    stddev = ((fwhm/2.998e5)*line_wavlen)/2.335  #fwhm(km/s)---> fwhm(delta_lambda) ---> stddev
    template = (line_lum*stats.norm.pdf(xx, loc = line_wavlen, scale=stddev)).sum(axis =1)
    return template



#table = pd.read_csv("A_Feltre_reduced.dat", sep =' ')

table = get_reduced_table(directory = "", master_fname = "A_Feltre_master.dat",
                        normalize = None)
print_templates(table, save_directory = "NL_templates",
                wavlen = np.logspace(3, np.log10(13000), 10000))







