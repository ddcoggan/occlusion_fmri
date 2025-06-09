# Created by David Coggan on 2023 06 28
import itertools
import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..fMRI.utils.config import PROJ_DIR
from ..fMRI.utils.plot_utils import custom_defaults
plt.rcParams.update(custom_defaults)

def ecc_to_size(eccs, method):

    def voxel_size_adjustment(voxel_size, ndims=1):
        if ndims == 1:
            scale_factor = 2 / np.mean(voxel_size)
        elif ndims == 2:
            my_area = 4 * 6  # 2mm isotropic
            their_area = np.sum([x * y for x, y in itertools.combinations(
                voxel_size, 2)] * 2)
            scale_factor = my_area / their_area
        return scale_factor

    if method == 'HCP':
        scale = voxel_size_adjustment([1.6, 1.6, 1.6])
        unit = 'sigma'
        intercept = .1856 * scale
        slope = .02676 * scale


    elif method == 'poltoratski':
        prfs = pd.read_csv(f'pRFs.csv')
        prfs = prfs[prfs.eccentricity < 3.5]

        # fit linear model for each region
        fits = prfs.groupby('region').apply(
            lambda x: np.polyfit(x['eccentricity'], x['fwhm'], 1))
        slope, intercept = fits['V1']
        unit = 'FWHM'


    sizes = [slope * ecc + intercept for ecc in eccs]
    values = [slope, intercept, *sizes]
    if unit == 'sigma':
        values = [i * 2.355 for i in values]

    return values


def estimate_prfs():

    eccs = np.arange(1, 5)
    methods = ['poltoratski', 'HCP']
    data = {method: ecc_to_size(eccs, method) for method in methods}
    data['parameter'] = ['slope', 'intercept'] + [f'FWHM at {e} dva' for e in eccs]
    pRF_estimates = pd.DataFrame(data)
    outpath = f'pRF_estimates.csv'
    pRF_estimates.to_csv(outpath, index=False)


def plot_prfs():

    # scatter plot

    # poltoratski points
    prfs = pd.read_csv(f'{PROJ_DIR}/data/in_vivo/fMRI/pRF/pRFs.csv')
    prfs = prfs[prfs.eccentricity < 3.5]
    region_data = prfs[prfs['region'] == 'V1']
    plt.figure(figsize=(5, 4))
    plt.scatter(region_data['eccentricity'], region_data['fwhm'], alpha=.5,
                s=10, color='tab:blue', edgecolor='none', clip_on=False)

    # poltoraski linear fit
    slope, intercept = ecc_to_size([], 'poltoratski')
    plt.plot([0,3.5], np.polyval([slope, intercept], [0,3.5]), lw=2,
             label='Poltoratski and Tong (2020)', color='tab:blue', zorder=3)


    # HCP linear fit
    slope, intercept = ecc_to_size([], 'HCP')
    plt.plot([0, 3.5], np.polyval([slope, intercept], [0, 3.5]), lw=2,
             label='Benson et al. (2018)', color='tab:orange', ls='dotted')

    plt.xlabel(r"pRF eccentricity ($\degree$)")
    plt.ylabel(r"pRF FWHM ($\degree$)")
    plt.tick_params(axis='both', which='major', pad=5)
    plt.xlim(0,3.5)
    plt.ylim(0,2.5)
    plt.legend(frameon=False)
    plt.savefig(f'pRF_fits.pdf')
    plt.show()


if __name__ == "__main__":

    #estimate_prfs()
    plot_prfs()
