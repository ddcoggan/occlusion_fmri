import os
import os.path as op
import sys
import glob
import time
import pickle as pkl

import cv2
import numpy as np
from types import SimpleNamespace
import shutil
from itertools import product as itp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
import torch
from PIL import Image
import torchvision.transforms as transforms
from scipy import optimize
from sklearn import linear_model

from utils.plot_utils import make_legend, custom_defaults
plt.rcParams.update(custom_defaults)

from in_vivo.fMRI.utils import CFG as fMRI
from in_vivo.fMRI.utils import RSA_dataset, RSA

sys.path.append(op.expanduser('~/david/repos/vonenet'))
from vonenet.vonenet import VOneNet

PROJ_DIR = op.expanduser('~/david/projects/p022_occlusion')
FMRI_DIR = op.join(PROJ_DIR, 'data/in_vivo/fMRI')

# stimulus parameters
IMAGE_SIZE_PIX = 224
IMAGE_SIZE_DEG = 9
PPD = IMAGE_SIZE_PIX / IMAGE_SIZE_DEG  # 24.888

# human parameters
FIX_STD_DEG_HUM = .25 #[0.291409, 0.197803] #
V1_PRF_FWHM_DEG = 0.74

# VOneNet parameters parameters
RF_METHOD = 'perc90'#['measured', 'mean', 'perc90']
ANALYSIS_DIR = f'{FMRI_DIR}/V1_models_rf-{RF_METHOD}'
NUM_FIX_SAMPLES = 64
NUMS_FILTERS = [256]  # sanity check: higher completion in tallest filters?
KERNEL_SIZE = 223  # ensure this is large enough to contain the filters
CELL_TYPES = ['all']#, 'simple', 'complex']
UNIT_SETS = ['responsive']#, 'all']
FIX_MULTIPLIERS = [1, 1.5, 2, 3]
FIX_STDS_DEG = [np.round(FIX_STD_DEG_HUM * i, 2) for i in FIX_MULTIPLIERS]
PRF_MULTIPLIERS = [1, 1.5, 2, 3]
PRF_SIZES_FWHM_DEG = [np.round(V1_PRF_FWHM_DEG * i, 2) for i in PRF_MULTIPLIERS]

# RSA parameters
INDEX_TYPES = ['prop']#['norm', 'prop', 'base', 'base2', 'rel']
SIMILARITY = 'pearson'
NORM = 'all-conds'
NORM_METHOD = 'z-score'

# plot parameters
ylabel = "correlation ($\it{r}$)"
fix_std_axis_label = 'relative fixation instability\n(VOneNet : humans)'
fix_stds_labels = [f'{i}:1' for i in FIX_MULTIPLIERS]
fix_stds_hum_labels = f'{FIX_STD_DEG_HUM:.2f}' + r"$\degree$"
rf_size_axis_label = 'relative (p)RF size\n(VOneNet : human V1)'
rf_sizes_labels = [f'{i}:1' for i in PRF_MULTIPLIERS]
#fix_std_axis_label = (r"spatial jitter $\sigma$" +
#                      '\nrelative to human fixation ' + r"$\sigma$")
#fix_std_axis_label = r"VOneNet jitter $\sigma$ : human fixation $\sigma$"
#fix_stds_labels = [f'{f:.2f}' + r"$\degree$" for f in FIX_STDS_DEG]
#fix_stds_labels = [v + r"$\times$" for v in ['1','2','3','4']]


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.chdir(ANALYSIS_DIR)
    fit_rf_model()
    get_rf_params()
    evaluate(overwrite=True)
    analyze_effects()
    make_rf_figures()


def get_tallest_filters(model):

    """ Get the indices of the tallest filters in image space. The gabor
    nx and ny are the width and height of the patch in units of a
    single cycle of the sinusoid. We can therefore convert these into sigma
    by dividing by the sf. Since the x and y are relative to the orientation of
    the sinusoid, we then convert these into image coords by rotating the
    distances based on the orientation of the gabor, and selecting those with
    the largest height. """
    sf, nx, ny, theta_deg = [model.gabor_params[key] for key in [
        'sf', 'nx', 'ny', 'theta']]
    sigx = nx / sf
    sigy = ny / sf
    theta = np.deg2rad(theta_deg)
    sig_height = sigx * np.cos(theta) + sigy * np.sin(theta)
    sig_width = sigx * np.sin(theta) + sigy * np.cos(theta)
    tallest_simple = np.argsort(np.abs(sig_height)[:256])[::-1]
    tallest_complex = np.argsort(np.abs(sig_height)[256:])[::-1] + 256

    return tallest_simple.copy(), tallest_complex.copy()


def gaussian(x, amplitude, xo, sigma, offset):
    xo = float(xo)
    a = 1 / (2 * sigma ** 2)
    g = offset + amplitude * np.exp(-a * ((x - xo) ** 2))
    return g


def circular_gaussian(xy, amplitude, xo, yo, sigma, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma ** 2) + (np.sin(theta) ** 2) / (
                2 * sigma ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma ** 2) + (np.sin(2 * theta)) / (
                4 * sigma ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma ** 2) + (np.cos(theta) ** 2) / (
                2 * sigma ** 2)
    g = offset + amplitude * np.exp(
        - (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
           + c * ((y - yo) ** 2)))
    return g.ravel()


def fit_gaussian(filter: np.array) -> SimpleNamespace:
    w, h = filter.shape
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    initial_guess = (filter.max(), w // 2, h // 2, 1, 0, 0)
    popt, pcov = optimize.curve_fit(circular_gaussian, (x, y), filter.ravel(),
                                    p0=initial_guess)
    return SimpleNamespace(amplitude=popt[0], xo=popt[1], yo=popt[2],
                           sigma=popt[3], theta=popt[4], offset=popt[5])


def fit_rf_model():

    out_dir = 'rf_model'
    os.makedirs(out_dir, exist_ok=True)

    # method 1: align mean RF size
    if RF_METHOD == 'measured':

        # test a range of visual angles and get FWHMs, including human fix jitter
        vis_angs = np.arange(0.8, 5, .2)
        get_rf_params(vis_angs=vis_angs, fix_stds=[FIX_STD_DEG_HUM],
                      out_dir=out_dir)
        rf_params = pd.read_csv(f'{out_dir}/rf_params.csv')

        # relationship is non-linear, so fit a linear model to log-transformed data
        vis_angs = rf_params['vis_ang'].to_numpy()
        vis_angs_log = np.log(vis_angs)
        fwhms = rf_params['FWHM_deg'].to_numpy()
        fwhms_log = np.log(fwhms)
        clf = linear_model.LinearRegression()
        clf.fit(fwhms_log.reshape(-1,1), vis_angs_log)

        # predict visual angles for many model RF sizes
        fwhms_model = np.linspace(min(fwhms), max(fwhms), 1000)
        fwhms_model_log = np.log(fwhms_model)
        vis_angs_model_log = clf.predict(fwhms_model_log.reshape(-1, 1))
        vis_angs_model = np.exp(vis_angs_model_log)

        # make plot (visual angle vs. RF size)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(vis_angs, fwhms, color='tab:blue')
        ax.plot(vis_angs_model, fwhms_model, color='tab:orange')
        ax.set_xlabel('visual angle (degrees)')
        ax.set_ylabel('RF size (degrees)')
        fig.savefig(f'{out_dir}/rf_model.png')
        plt.tight_layout()
        plt.close()

        # make plot (log visual angle vs. log RF size)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(vis_angs_log, fwhms_log, color='tab:blue')
        ax.plot(vis_angs_model_log, fwhms_model_log, color='tab:orange')
        ax.set_xlabel('log visual angle (degrees)')
        ax.set_ylabel('log RF size (degrees)')
        fig.savefig(f'{out_dir}/rf_model_log.png')
        plt.tight_layout()
        plt.close()

        with open(f'{out_dir}/rf_model.pkl', 'wb') as f:
            pkl.dump(clf, f)


    # method 2: fit model based on mean or 90th percentils of the area of
    # each individual RF, based on the gabor parameters. This doesn't account
    # for fixational jitter.
    else:

        """ 
        The gabor nx and ny are the width and height of the patch in units of a 
        single cycle of the sinusoid. We can therefore convert these into sigma 
        by dividing by the sf, then calculate the area, and solve for the 
        radius of a circle with the same area.
        """

        model = VOneNet(model_arch=None, visual_degrees=IMAGE_SIZE_DEG,
                        ksize=KERNEL_SIZE, rand_param=False)
        sf, nx, ny = [model.gabor_params[key] for key in ['sf', 'nx', 'ny']]
        sf /= PPD
        sigx = nx / sf
        sigy = ny / sf
        area = sigx * sigy * np.pi
        if RF_METHOD == 'perc90':
            sigma = np.sqrt(np.percentile(area, 90) / np.pi).item()
        elif RF_METHOD == 'mean':
            sigma = np.sqrt(area.mean() / np.pi).item()
        fwhm_actual = (sigma * 2.355) / PPD
        with open(f'{out_dir}/rf_model.pkl', 'wb') as f:
            pkl.dump(fwhm_actual, f)


def get_fwhm_to_dva():

    """
    Loads the linear regression model or the fwhm when configured with
    the human dva. Then creates a function to obtain required dva for a
    desired fwhm.
    """
    with open('rf_model/rf_model.pkl', 'rb') as f:
        params = pkl.load(f)

    if type(params) == linear_model.LinearRegression:
        def fwhm_to_dva(fwhm):
            dva = np.exp(params.predict(np.log(fwhm).reshape(-1, 1))).item()
            return dva
    else:
        def fwhm_to_dva(fwhm):
            dva = (params / fwhm) * IMAGE_SIZE_DEG
            return dva

    return fwhm_to_dva


def get_rf_params(rf_sizes=PRF_SIZES_FWHM_DEG,
                  vis_angs=None,
                  fix_stds=FIX_STDS_DEG,
                  nums_filters=NUMS_FILTERS,
                  cell_types=CELL_TYPES,
                  out_dir='receptive_fields'):

    if vis_angs is None:
        fwhm_to_dva = get_fwhm_to_dva()
        vis_angs = [fwhm_to_dva(i) for i in rf_sizes]

    df = pd.DataFrame()
    for v, (vis_ang) in enumerate(vis_angs):

        model = VOneNet(model_arch=None, visual_degrees=vis_ang,
                        ksize=KERNEL_SIZE, rand_param=False)
        simple_chans, complex_chans = get_tallest_filters(model)

        for num_filters, fix_std, cell_type in itp(
                nums_filters, fix_stds, cell_types):

            params = dict(
                simple=model.simple_conv_q0.weight[simple_chans[:num_filters]],
                complex=torch.cat([
                    model.simple_conv_q0.weight[complex_chans[:num_filters]],
                    model.simple_conv_q1.weight[complex_chans[:num_filters]]], dim=0))
            params['all'] = torch.cat([params['simple'], params['complex']], dim=0)
            filters = params[cell_type]
            mean_filter = filters.abs().mean((0,1)).numpy()

            # add RFs to image, with fixational jitter
            image = np.zeros([IMAGE_SIZE_PIX] * 2)
            center = IMAGE_SIZE_PIX // 2
            for r in range(1000):
                ecc = np.random.normal(loc=0, scale=fix_std * PPD)
                ang = np.random.uniform(low=0, high=2 * np.pi)
                i_start = center + int(ecc * np.cos(ang)) - KERNEL_SIZE // 2
                j_start = center + int(ecc * np.sin(ang)) - KERNEL_SIZE // 2
                i_stop, j_stop = i_start + KERNEL_SIZE, j_start + KERNEL_SIZE
                filter_copy = mean_filter.copy()
                if i_start < 0:
                    filter_copy = filter_copy[-i_start:, :]
                    i_start = 0
                if j_start < 0:
                    filter_copy = filter_copy[:, -j_start:]
                    j_start = 0
                if i_stop > IMAGE_SIZE_PIX:
                    filter_copy = filter_copy[:(IMAGE_SIZE_PIX - i_stop), :]
                if j_stop > IMAGE_SIZE_PIX:
                    filter_copy = filter_copy[:, :(IMAGE_SIZE_PIX - j_stop)]
                image[i_start:i_stop, j_start:j_stop] += filter_copy
            image = np.array(image / image.max() * 255).astype(np.uint8)

            # fit gaussian
            rf = fit_gaussian(image)
            FWHM = rf.sigma * 2.355
            FWHM_deg = FWHM / PPD

            # get  base RF size (calibrated to human-level fixational jitter)
            if rf_sizes is not None and len(rf_sizes) == len(vis_angs):
                rf_size = rf_sizes[v]
            else:
                rf_size = FWHM_deg

            # create output directory
            outdir = (f'{out_dir}/filt-{num_filters}_fixstd-{fix_std:.2f}_'
                      f'rfsize-{rf_size:.2f}_cell-{cell_type}')
            os.makedirs(outdir, exist_ok=True)

            # save montage of filters
            n_filters = filters.shape[0]  # actual filters may != num_filters
            grid_size = np.ceil(np.sqrt(n_filters))
            montage_size = [int(KERNEL_SIZE * grid_size)] * 2
            montage = Image.new(size=montage_size, mode='RGB')
            for i in range(n_filters):
                image_array = np.array(filters[i, :, :, :].permute(1, 2, 0))
                image_pos = image_array - image_array.min()  # rescale to between 0,255 for PIL
                image_scaled = image_pos * (255.0 / image_pos.max())
                image_pil = Image.fromarray(image_scaled.astype(np.uint8))
                offset_x = int(i % grid_size) * KERNEL_SIZE
                offset_y = int(i / grid_size) * KERNEL_SIZE
                montage.paste(image_pil, (offset_x, offset_y))
            montage.save(f'{outdir}/conv_filters.png')

            # 2D plot of RF, with object in background
            #image_pil = Image.fromarray((image * 255).astype(np.uint8))
            # apply colormap
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            lut = cv2.applyColorMap(
                np.arange(256, dtype=np.uint8), cv2.COLORMAP_INFERNO)
            image_lut = cv2.LUT(image_rgb, lut)
            # separate color channels and multiply by alpha (i.e. the RF image)
            fg_r, fg_g, fg_b = cv2.split(image_lut)
            alpha = np.sqrt(image / 255)
            image_rgba = cv2.merge([
                (fg_r * alpha).astype(np.uint8),
                (fg_g * alpha).astype(np.uint8),
                (fg_b * alpha).astype(np.uint8)])
            # do the same for the object, but with 1 - alpha
            object_rgb = np.array(Image.open(
                f'{FMRI_DIR}/exp1/stimuli/images/bear_lower.png'
            ).convert('RGB'))
            mean_object = np.array(Image.open(
                f'{FMRI_DIR}/exp1/stimuli/images/bear_complete.png')).mean()
            object_rgb = np.clip(
                (object_rgb - mean_object) * 0.5 + mean_object, 0, 255
            ).astype(np.uint8)  # lower the contrast
            obj_r, obj_g, obj_b = cv2.split(object_rgb)
            object_rgba = cv2.merge([
                (obj_r * (1 - alpha)).astype(np.uint8),
                (obj_g * (1 - alpha)).astype(np.uint8),
                (obj_b * (1 - alpha)).astype(np.uint8)])
            # add the two images together
            figure_rgba = cv2.add(image_rgba, object_rgba)
            # add the full kernel size relative to the image
            # image_kl = cv2.rectangle(image_rf, (center - KERNEL_SIZE // 2,
            #                                    center - KERNEL_SIZE // 2),
            #                            (center + KERNEL_SIZE // 2,
            #                                center + KERNEL_SIZE // 2),
            #                            color=(0, 255, 0), thickness=1)
            # add the RF FWHM to the image
            # make image larger so RF is smooth
            sf = 2
            image_size_large = (IMAGE_SIZE_PIX * sf, IMAGE_SIZE_PIX * sf)
            figure_rgba = cv2.resize(figure_rgba, image_size_large)
            center_large = (center*sf, center*sf)
            radius_large = int((FWHM / 2) * sf)
            figure_rgba = cv2.circle(figure_rgba, center_large, radius_large,
                                  color=(255, 255, 255), thickness=2)
            # for debugging, show image
            #figure_rgba_pil = Image.fromarray(cv2.cvtColor(
            #    figure_rgba, cv2.BGR2RGB))
            #figure_rgba_pil.show()
            cv2.imwrite(f'{outdir}/RF_size_2D.png', figure_rgba)


            # 1D plot of RF
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(image[center])
            ax.plot(gaussian(np.arange(IMAGE_SIZE_PIX), image.max(),
                             center, rf.sigma, rf.offset)),
            ax.axvline(center - FWHM / 2, color='tab:red')
            ax.axvline(center + FWHM / 2, color='tab:red')
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_xlabel('pixels')
            ax.set_xlim((0, IMAGE_SIZE_PIX))
            ax.set_xticks(np.arange(0, IMAGE_SIZE_PIX, 28))
            plt.tight_layout()
            plt.savefig(f'{outdir}/RF_size_1D.png')
            plt.close()

            # save RF params
            df = pd.concat([df, pd.DataFrame(dict(
                rf_size=[f'{rf_size:.2f}'],
                rf_effective=[f'{FWHM_deg:.2f}'],
                vis_ang=[vis_ang],
                fix_std=[fix_std],
                kernel_size_pix=[KERNEL_SIZE],
                num_filters=[num_filters],
                cell_type=[cell_type],
                sigma_pix=[rf.sigma],
                sigma_deg=[rf.sigma / PPD],
                FWHM_pix=[FWHM],
                FWHM_deg=[FWHM / PPD],
                amplitude=[rf.amplitude],
                xo=[rf.xo],
                yo=[rf.yo],
            ))])

    df.to_csv(f'{out_dir}/rf_params.csv', index=False)


def evaluate(overwrite=False):

    # prepare data objects
    torch.no_grad()
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    image_dir = f'{FMRI_DIR}/exp1/stimuli/images'
    images = sorted(glob.glob(f'{image_dir}/*'))
    dataset = torch.empty((len(images), 3, 224, 224))
    image_counter = 0
    for cond in fMRI.cond_labels['exp1']:
        cond_label = cond.replace('none', 'complete')
        for im, image in enumerate(images):
            if cond_label in image:
                image_PIL = Image.open(image).convert('RGB')
                dataset[image_counter] = transforms.ToTensor()(image_PIL)
                image_counter += 1
    mean_luminance = dataset.mean()

    if not op.isfile('conditions.csv') or overwrite:
        df_conds = pd.DataFrame()
    else:
        df_conds = pd.read_csv('conditions.csv')

    if not op.isfile('indices.csv') or overwrite:
        df_indices = pd.DataFrame()
    else:
        df_indices = pd.read_csv('indices.csv')

    fwhm_to_dva = get_fwhm_to_dva()

    # loop over different levels of fixation stability and visual angle
    for fix_std, rf_size in itp(
            FIX_STDS_DEG, PRF_SIZES_FWHM_DEG):

        if not len(df_conds) or not bool(
                len(df_conds[
                    (df_conds.fix_std == fix_std) &
                    (df_conds.rf_size == rf_size)]) * \
                len(df_indices[
                    (df_indices.fix_std == fix_std) &
                    (df_indices.rf_size == rf_size)])):

            print(f'Measuring responses, '
                  f'{fix_std:.2f} deg fixation jitter, '
                  f'{rf_size:.2f} deg RF size')

            vis_ang = fwhm_to_dva(rf_size)
            model = VOneNet(model_arch=None, visual_degrees=vis_ang,
                            ksize=KERNEL_SIZE, rand_param=False).cuda()

            # store sample image inputs
            sample_dir = (f'sample_inputs/fixstd-{fix_std:.2f}_rfsize-'
                          f'{rf_size:.2f}')
            os.makedirs(sample_dir, exist_ok=True)
            sample_inputs = torch.zeros(NUM_FIX_SAMPLES, 2, 3, 224, 224)

            # loop over reps
            activations = torch.empty((NUM_FIX_SAMPLES, 24, 512, 56, 56))
            for r in range(NUM_FIX_SAMPLES):

                # input dataset with fixational jitter
                inputs = dataset.clone()
                for i, image in enumerate(dataset):
                    ecc = np.random.normal(loc=0, scale=fix_std * PPD)
                    ang = np.random.uniform(low=0, high=2 * np.pi)
                    x = int(ecc * np.cos(ang))
                    y = int(ecc * np.sin(ang))
                    inputs[i] = transforms.functional.affine(
                        img=image, angle=0, translate=(x, y),
                        scale=1., shear=0., fill=mean_luminance)
                #inputs = normalize(inputs).cuda()
                activations[r] = model(inputs.cuda()).detach().cpu()

                # store some inputs
                sample_input = inputs[1:3].clone()
                sample_inputs[r] = sample_input
                for i in range(2):
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(transforms.ToPILImage()(sample_input[i]))
                    ax.spines[['left', 'bottom']].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.tight_layout()
                    plt.savefig(op.join(sample_dir, f'input_{i}_{r}.png'))
                    plt.close()

            # calculate mean input across reps
            sample_means = sample_inputs.mean(0)
            for i in range(2):
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(transforms.ToPILImage()(sample_means[i]))
                ax.spines[['left', 'bottom']].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()
                plt.savefig(op.join(sample_dir, f'input_{i}_mean.png'))
                plt.close()

            for num_filters, cell_type, unit_set in itp(
                    NUMS_FILTERS, CELL_TYPES, UNIT_SETS):

                analysis = (f'num-filters-{num_filters}_fixstd-{fix_std:.2f}_'
                    f'rfsize-{rf_size:.2f}_cell-{cell_type}_units-{unit_set}')

                # restrict to top n channels with tallest filters
                simple_chans, complex_chans = get_tallest_filters(model)
                if cell_type == 'simple':
                    chans = simple_chans[:num_filters]
                elif cell_type == 'complex':
                    chans = complex_chans[:num_filters]
                else:  # all
                    chans = np.concatenate([simple_chans[:num_filters],
                                            complex_chans[:num_filters]])
                acts = activations[:, :, chans]

                # select units with a z-score > 3.1
                if unit_set == 'responsive':
                    unit_means = acts.mean(axis=(0,1))
                    selected_units = unit_means > (unit_means.std() * 3.1)
                    print(f'Number of selected units: {selected_units.sum()}/'
                         f'{np.prod(unit_means.shape)}')
                    acts = acts[:, :, selected_units]

                # split into two halves, flatten, and calculate mean response
                acts = acts.reshape(2, NUM_FIX_SAMPLES // 2, 24, -1).mean(
                    axis=1).unsqueeze(0).numpy()

                # make RSA dataset and calculate RSM
                print(f'Calculating RSM')
                RSM = RSA_dataset(responses=acts).calculate_RSM(
                    NORM, NORM_METHOD, SIMILARITY)

                # plot RSM
                print(f'Plotting RSM ')
                RSM.plot_RSM(
                    vmin=-1, vmax=1,
                    fancy=True,
                    labels=fMRI.cond_labels['exp1'],
                    outpath=(f'RSMs/{analysis}.pdf'),
                    measure=ylabel)

                # MDS
                print(f'Plotting MDS')
                outpath = f'MDS/{analysis}.pdf'
                RSM.plot_MDS(title=None, outpath=outpath)

                # condition-wise similarities
                RSM.RSM_to_table()
                df = RSM.RSM_table.copy(deep=True)
                df = df.drop(columns=['exemplar_b', 'occluder_a',
                    'occluder_b']).groupby(['exemplar_a', 'analysis', 'level']).agg('mean'). \
                    dropna().reset_index()
                df = (df
                    .groupby(['analysis', 'level'])
                    .agg({'similarity': ['mean', 'sem']})
                    .reset_index()
                )
                df.columns = ['analysis', 'level', 'similarity', 'sem']
                df['num_filters'] = num_filters
                df['rf_size'] = rf_size
                df['vis_ang'] = vis_ang
                df['cell_type'] = cell_type
                df['fix_std'] = fix_std
                df['unit_set'] = unit_set
                df['num_units'] = acts.shape[-1]
                df_conds = pd.concat([df_conds, df.copy(deep=True)]).reset_index(
                    drop=True)

                # occlusion robustness indices, including object bias
                RSM.calculate_occlusion_robustness()
                RSM.fit_models()
                df = RSM.occlusion_robustness.copy(deep=True)
                df = pd.concat([df, pd.DataFrame(dict(
                    analysis=['object_bias'],
                    subtype=['object_bias'],
                    index=['object bias'],
                    value=[RSM.exemplar_v_occluder_position]))])
                df['num_filters'] = num_filters
                df['rf_size'] = rf_size
                df['vis_ang'] = vis_ang
                df['cell_type'] = cell_type
                df['fix_std'] = fix_std
                df['unit_set'] = unit_set
                df['num_units'] = acts.shape[-1]
                df_indices = pd.concat([df_indices, df]).reset_index(drop=True)

    df_conds.to_csv('conditions.csv', index=False)
    df_indices.to_csv('indices.csv', index=False)


def analyze_effects():

    df_indices = pd.read_csv('indices.csv')
    df_conds = pd.read_csv('conditions.csv')

    # ensure levels are ordered correctly
    level_order = fMRI.occlusion_robustness_analyses[
                      'object_completion']['conds'] + \
                  fMRI.occlusion_robustness_analyses[
                      'occlusion_invariance']['conds']
    df_conds.level = df_conds.level.astype('category').cat.reorder_categories(
        level_order)

    for (analysis, params), num_filters, cell_type, unit_set in itp(
            fMRI.occlusion_robustness_analyses.items(), NUMS_FILTERS,
            CELL_TYPES, UNIT_SETS):

        out_dir = (f'{analysis}/filt-{num_filters}_cell-{cell_type}_units-'
                    f'{unit_set}')
        os.makedirs(out_dir, exist_ok=True)

        # condition-wise similarities (human-fit subset)
        outpath = f'{out_dir}/condition-wise_similarities_humanlike.pdf'
        fig, ax = plt.subplots(figsize=(2, 3), sharey=True)
        df = df_conds[(df_conds.analysis == analysis) &
                      (df_conds.rf_size == V1_PRF_FWHM_DEG) &
                      (df_conds.fix_std == FIX_STD_DEG_HUM) &
                      (df_conds.num_filters == num_filters) &
                      (df_conds.cell_type == cell_type) &
                      (df_conds.unit_set == unit_set)]
        xvals = np.arange(len(df))
        yvals = df.similarity.values
        yerr = df['sem'].values
        ax.bar(xvals, yvals, color=params['colours'], width=1)
        ax.errorbar(xvals, yvals, yerr=yerr, color='k',
                    linestyle='', capsize=3, clip_on=False)
        ax.set_yticks(np.arange(-1, 1.1, .5))
        ax.set_ylim(-.5, 1)
        ax.set_xlim(-1.5, 4.5)
        ax.set_ylabel("correlation ($\it{r}$)", fontsize=12)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='minor', bottom=False,
                       left=False)
        ax.set_xticks([])
        ax.set_xlabel(None)
        plt.tight_layout()
        fig.savefig(outpath)
        plt.close()

        # robustness indices
        for index_type in INDEX_TYPES:

            # human indices
            fMRI_vals, fMRI_errs = [], []
            for exp, task in zip(
                    ['exp1', 'exp2', 'exp2'],
                    ['occlusion', 'occlusionAttnOn', 'occlusionAttnOff']):
                fMRI_data = pd.read_csv(
                    f'{FMRI_DIR}/{exp}/derivatives/RSA/'
                    f'{task}_space-standard/norm-all-conds_z-score/'
                    f'pearson/{analysis}/indices.csv')
                fMRI_vals.append(fMRI_data['value'][
                    (fMRI_data.level == 'ind') &
                    (fMRI_data.subtype == 'prop2') &
                    (fMRI_data.region == 'V1')].mean())
                fMRI_errs.append(fMRI_data['value'][
                    (fMRI_data.level == 'ind') &
                    (fMRI_data.subtype == 'prop2') &
                    (fMRI_data.region == 'V1')].sem())

            outpath = f'{out_dir}/indices_barplot_{index_type}.pdf'

            colors = matplotlib.cm.tab20.colors
            text_offset = .02
            fontsize = 9
            fig, axes = plt.subplots(
                nrows=1, ncols=2, figsize=(2.5, 4), sharey=True,
                gridspec_kw={'width_ratios': [3, .9]})

            # human panel
            ax = axes[0]
            #human_colors = [colors[i] for i in [6,6,7]]
            ax.bar(np.arange(3), fMRI_vals, color='tab:blue')
            ax.errorbar(np.arange(3), fMRI_vals, fMRI_errs, color='k',
                        linestyle='', capsize=3, clip_on=False)
            ax.set_xticks([])
            ax.set_yticks(np.arange(0, 1.1, .5))
            ax.set_ylim(-.1, 1)
            ax.set_ylabel(f'{analysis.replace("_", " ")} index', size=11)
            ax.set_xlabel("Human V1", size=11)
            ax.text(0, text_offset, f'Exp. 1, attended',
                    fontsize=fontsize, ha='center', rotation=90, c='w')
            ax.text(0, fMRI_vals[0] + fMRI_errs[0] + text_offset,
                    f'{fMRI_vals[0]:.2f}', ha='center')
            ax.text(1, text_offset, f'Exp. 2, attended',
                    fontsize=fontsize, ha='center', rotation=90, c='w')
            ax.text(1, fMRI_vals[1] + fMRI_errs[1] + text_offset,
                    f'{fMRI_vals[1]:.2f}', ha='center')
            ax.text(2, text_offset, f'Exp. 2, unattended',
                    fontsize=fontsize, ha='center', rotation=90, c='w')
            ax.text(2, fMRI_vals[2] + fMRI_errs[2] + text_offset,
                f'{fMRI_vals[2]:.2f}', ha='center')
            ax.spines['bottom'].set_visible(False)

            # model panel
            ax = axes[1]
            model_val = df_indices[
                (df_indices.analysis == analysis) &
                (df_indices.fix_std == FIX_STD_DEG_HUM) &
                (df_indices.rf_size == V1_PRF_FWHM_DEG) &
                (df_indices.subtype == index_type) &
                (df_indices.num_filters == num_filters) &
                (df_indices.cell_type == cell_type) &
                (df_indices.unit_set == unit_set)].value.values[0].item()
            model_colors = [colors[8]]
            ax.bar(0, model_val, color=model_colors)
            if model_val > 0:
                text_y = model_val + text_offset
            else:
                text_y = model_val - text_offset - .04
            ax.text(0, text_y, f'{model_val:.2f}', ha='center')
            ax.set_xticks([])
            ax.set_xlabel("VOneNet", size=11)
            ax.spines[['bottom', 'left']].set_visible(False)
            ax.tick_params(axis='y', which='both', left=False)
            plt.tight_layout()
            #fig.subplots_adjust(bottom=0.13, left=.3, top=.95)
            plt.savefig(outpath)
            plt.close()


        # condition-wise similarities (entire set)
        outpath = f'{out_dir}/condition-wise_similarities.pdf'
        fig, axes = plt.subplots(len(FIX_MULTIPLIERS), len(PRF_MULTIPLIERS),
                                 figsize=(4,4), sharex=True, sharey=True)
        for (f, fix_std), (r, rf_size) in itp(
                enumerate(FIX_STDS_DEG), enumerate(PRF_SIZES_FWHM_DEG)):
            ax = axes[f, r]
            df = df_conds[(df_conds.analysis == analysis) &
                          (df_conds.fix_std == fix_std) &
                          (df_conds.rf_size == rf_size) &
                          (df_conds.num_filters == num_filters) &
                          (df_conds.cell_type == cell_type) &
                          (df_conds.unit_set == unit_set)]
            xvals = np.arange(len(df))
            yvals = df.similarity.values
            yerr = df['sem'].values
            ax.bar(xvals, yvals, color=params['colours'], width=1)
            ax.errorbar(xvals, yvals, yerr=yerr, color='k',
                        linestyle='', capsize=2, clip_on=False)
            ax.set_yticks(np.arange(-1, 1.1, 1), labels=[-1,0,1], size=8)
            ax.set_ylim(-.5, 1)
            ax.set_xlim(-1.5, 4.5)
            if r == 0:
                ax.set_ylabel(fix_stds_labels[f], fontsize=10, rotation=0,
                              labelpad=30, ha='center', va='center')
                ax.tick_params(**{'length': 2})
            else:
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='both', which='both', bottom=False,
                               left=False)
            if f == len(FIX_STDS_DEG)-1 and r == len(PRF_SIZES_FWHM_DEG)//2:
                ax.set_title(rf_size_axis_label, fontsize=12, y=-1.2, x=-.11)
            #elif f < len(FIX_STDS_DEG)-1:
            ax.spines['bottom'].set_visible(False)
            ax.grid(axis='y', linestyle='solid', alpha=.5, zorder=0, clip_on=False)
            ax.tick_params(axis='both', which='minor', bottom=False,
                           left=False)
            ax.set_xticks([])
            if f == len(FIX_STDS_DEG)-1:
                ax.set_xlabel(rf_sizes_labels[r], fontsize=10, labelpad=5)
        fig.text(.06, .55, fix_std_axis_label, fontsize=12, rotation=90,
                 ha='center', va='center')
        fig.text(.25, .86, "$\it{r}$", fontsize=12,# rotation=90,
                 ha='center', va='center')
        #fig.set_title(r"spatial jitter $\sigma$", loc='left')
        fig.subplots_adjust(bottom=.2, left=.3, top=.95)
        #plt.tight_layout()
        fig.savefig(outpath)
        plt.close()


        # robustness indices
        for index_type in INDEX_TYPES:

            # human indices
            fMRI_vals = []
            for exp, task in zip(
                    ['exp1', 'exp2', 'exp2'],
                    ['occlusion', 'occlusionAttnOn', 'occlusionAttnOff']):
                fMRI_data = pd.read_csv(
                    f'{FMRI_DIR}/{exp}/derivatives/RSA/'
                    f'{task}_space-standard/norm-all-conds_z-score/'
                    f'pearson/{analysis}/indices.csv')
                fMRI_vals.append(fMRI_data['value'][
                     (fMRI_data.level == 'ind') &
                     (fMRI_data.subtype == 'prop2') &
                     (fMRI_data.region == 'V1')].mean())
            print(f'{analysis} {index_type} fMRI values: {fMRI_vals}')
            print(np.mean(fMRI_vals))

            outpath = f'{out_dir}/indices_matrix_{index_type}.pdf'
            matrix_values = np.empty((len(FIX_STDS_DEG), len(PRF_SIZES_FWHM_DEG)))
            for (f, fix_std), (r, rf_size) in itp(
                    enumerate(FIX_STDS_DEG), enumerate(PRF_SIZES_FWHM_DEG)):
                matrix_values[f, r] = df_indices[
                    (df_indices.analysis == analysis) &
                    (df_indices.fix_std == fix_std) &
                    (df_indices.rf_size == rf_size) &
                    (df_indices.num_filters == num_filters) &
                    (df_indices.cell_type == cell_type) &
                    (df_indices.unit_set == unit_set) &
                    (df_indices.subtype == index_type)].value.values[0].item()
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.set_aspect('equal', adjustable='box')
            im = ax.imshow(matrix_values, vmin=0, vmax=1, cmap='viridis')
            ax.tick_params(**{'length': 0})
            ax.set_xticks(np.arange(len(PRF_SIZES_FWHM_DEG)), rf_sizes_labels)
            ax.set_yticks(np.arange(len(FIX_STDS_DEG)), fix_stds_labels)
            ax.tick_params(direction='in')
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.set_title(analysis.replace('_', ' ') + ' index', fontsize=12)
            cbar = fig.colorbar(im, fraction=0.0453)
            fMRI_text = ['human V1, Exp. 1, object task',
                         'human V1, Exp. 2, object task',
                         'human V1, Exp. 2, letter task']
            V1_str = f'V1 ({np.mean(fMRI_vals):.2f})'.replace('0.', '.')
            cbar.set_ticks([0, np.mean(fMRI_vals), 1],
                           labels=['0', V1_str, '1'])
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
                     rotation_mode="anchor")
            for f, r in itp(range(len(FIX_STDS_DEG)), range(len(PRF_SIZES_FWHM_DEG))):
                value = matrix_values[f, r]
                text_col = 'w' if value < np.mean(fMRI_vals) else 'k'
                ax.text(r, f, f'{value:.2f}'.replace('0.', '.'),
                        ha='center', va='center', color=text_col)
            ax.set_ylabel(fix_std_axis_label, fontsize=12)
            ax.set_xlabel(rf_size_axis_label, fontsize=12)
            fig.subplots_adjust(left=0.02, top=.94, right=.82, bottom=.16)
            plt.savefig(outpath)
            plt.close()

        #tile(rf_images, f'{out_dir}/RF_images.png', num_rows=6, num_cols=5,
        #     base_gap=8)


def make_rf_figures():


    rf_params = pd.read_csv('receptive_fields/rf_params.csv')
    num_filters = 256
    cell_type = 'all'

    # tile pRF images
    outpath = f'receptive_fields/RF_images.pdf'
    fig, axes = plt.subplots(len(FIX_STDS_DEG), len(PRF_SIZES_FWHM_DEG),
                             figsize=(4.5, 4), sharex=True, sharey=True)
    for (f, fix_std), (r, rf_size) in itp(
            enumerate(FIX_STDS_DEG), enumerate(PRF_SIZES_FWHM_DEG)):
        ax = axes[f, r]
        ax.imshow(plt.imread(f'receptive_fields/filt-{num_filters}_fixstd-'
                             f'{fix_std:.2f}_rfsize-{rf_size:.2f}_cell-'
                             f'{cell_type}/RF_size_2D.png'))
        FWHM = rf_params[
            (rf_params.fix_std == fix_std) &
            (rf_params.rf_size == rf_size) &
            (rf_params.cell_type == cell_type) &
            (rf_params.num_filters == num_filters)].FWHM_deg.item()
        if r == 0:
            ax.set_ylabel(fix_stds_labels[f], fontsize=10, rotation=0,
                          labelpad=20)
        ax.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.grid(axis='y', linestyle='solid', alpha=.5, zorder=0, clip_on=False)
        ax.tick_params(axis='both', which='minor', bottom=False,
                       left=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'FWHM: {FWHM:.2f}' + r"$\degree$", fontsize=6,
                     color='w', y=.7, zorder=3)
        if f == len(FIX_STDS_DEG) - 1:
            ax.set_xlabel(rf_sizes_labels[r], fontsize=10, labelpad=5)
    fig.subplots_adjust(bottom=0.1, left=0.15)
    fig.text(0.05, .5, fix_std_axis_label, fontsize=12,
             ha='center', va='center', rotation=90)
    fig.text(0.55, .04, rf_size_axis_label, fontsize=12,
             ha='center', va='center')
    fig.text(.52, .97, 'effective RF size, accounting for fixation instability',
             fontsize=12, ha='center', va='center')
    plt.subplots_adjust(left=.2, top=.92, bottom=.15)
    fig.savefig(outpath)
    plt.close()

    # matrix of RF sizes
    outpath = f'receptive_fields/effective_relative_RF_sizes.pdf'
    matrix_values = np.empty((len(FIX_STDS_DEG), len(PRF_SIZES_FWHM_DEG)))
    for (f, fix_std), (r, rf_size) in itp(
            enumerate(FIX_STDS_DEG), enumerate(PRF_SIZES_FWHM_DEG)):
        matrix_values[f, r] = (rf_params[
                                   (rf_params.fix_std == fix_std) &
                                   (rf_params.rf_size == rf_size) &
                                   (rf_params.cell_type == cell_type) &
                                   (
                                               rf_params.num_filters == num_filters)].FWHM_deg.item() / .74)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.set_aspect('equal', adjustable='box')
    im = ax.imshow(matrix_values, vmin=0, vmax=4, cmap='viridis')
    ax.tick_params(**{'length': 0})
    ax.set_xticks(np.arange(len(PRF_SIZES_FWHM_DEG)), rf_sizes_labels)
    ax.set_yticks(np.arange(len(FIX_STDS_DEG)), fix_stds_labels)
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_title('effective RF size (relative to humans)', fontsize=12)
    cbar = fig.colorbar(im, fraction=0.0453)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")
    for f, v in itp(range(len(FIX_STDS_DEG)), range(len(PRF_SIZES_FWHM_DEG))):
        value = matrix_values[f, v]
        text_col = 'w' if value <= 1 else 'k'
        ax.text(v, f, f'{value:.2f}'.replace('0.', '.'),
                ha='center', va='center', color=text_col)
    ax.set_ylabel(fix_std_axis_label, fontsize=12)
    ax.set_xlabel(rf_size_axis_label, fontsize=12)
    fig.subplots_adjust(left=0.02, top=.94, right=.82, bottom=.16)
    plt.savefig(outpath)
    plt.close()

    # matrix of RF sizes (absolute)
    outpath = f'receptive_fields/effective_absolute_RF_sizes.pdf'
    matrix_values = np.empty((len(FIX_STDS_DEG), len(PRF_SIZES_FWHM_DEG)))
    for (f, fix_std), (v, rf_size) in itp(
            enumerate(FIX_STDS_DEG), enumerate(PRF_SIZES_FWHM_DEG)):
        matrix_values[f, v] = (rf_params[
                                   (rf_params.fix_std == fix_std) &
                                   (rf_params.rf_size == rf_size) &
                                   (rf_params.cell_type == cell_type) &
                                   (rf_params.num_filters == num_filters)].FWHM_deg.item())

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.set_aspect('equal', adjustable='box')
    im = ax.imshow(matrix_values, vmin=0, vmax=4, cmap='viridis')
    ax.tick_params(**{'length': 0})
    ax.set_xticks(np.arange(len(PRF_SIZES_FWHM_DEG)), rf_sizes_labels)
    ax.set_yticks(np.arange(len(FIX_STDS_DEG)), fix_stds_labels)
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_title('effective RF size (FWHM)', fontsize=12)
    cbar = fig.colorbar(im, fraction=0.0453)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")
    for f, v in itp(range(len(FIX_STDS_DEG)), range(len(PRF_SIZES_FWHM_DEG))):
        value = matrix_values[f, v]
        text_col = 'w' if value <= .74 else 'k'
        ax.text(v, f, f'{value:.2f}'.replace('0.', '.'),
                ha='center', va='center', color=text_col)
    ax.set_ylabel(fix_std_axis_label, fontsize=12)
    ax.set_xlabel(rf_size_axis_label, fontsize=12)
    fig.subplots_adjust(left=0.02, top=.94, right=.82, bottom=.16)
    plt.savefig(outpath)
    plt.close()




if __name__ == '__main__':
    main()

