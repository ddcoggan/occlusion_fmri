# /usr/bin/python
# Created by David Coggan on 2023 06 23

"""
script for running RSA
"""

import os
import os.path as op
import glob
import itertools
import numpy as np
import pickle as pkl
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS, TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import euclidean_distances
from rsatoolbox.util.searchlight import get_volume_searchlight
import pandas as pd
import matplotlib
from sympy.codegen.ast import continue_
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
from scipy.stats import spearmanr
import rsatoolbox
import gc
import pingouin as pg
from statsmodels.stats.anova import AnovaRM


from .config import CFG, TABCOLS, PROJ_DIR
from .plot_utils import export_legend, custom_defaults
plt.rcParams.update(custom_defaults)


def line_plot(df, outpath, title=None, ylabel=None, x_var='region',
              x_tick_labels=None, y_var='value', cond_var='cond',
              col_var='colour', error_var='error', ceiling=None, floor=None,
              ylims=(0, 1.25), hline=None, yticks=np.arange(-1, 1.1, .5),
              figsize=(3.5, 2)):

    if cond_var not in df.columns:
        df[cond_var] = 'all'
        df[col_var] = 'tab:grey'
    if x_tick_labels is None:
        x_tick_labels = df[x_var].unique()
    n_x_ticks = len(x_tick_labels)
    os.makedirs(op.dirname(outpath), exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(n_x_ticks)
    for cond in df[cond_var].unique():
        df_cond = df[df[cond_var] == cond]
        if error_var in df_cond.columns and not df_cond[error_var].isnull(
                ).all():
            ax.errorbar(range(n_x_ticks),
                        df_cond[y_var].values,
                        yerr=df_cond[error_var].values,
                        color=df_cond[col_var].values[0],
                        linestyle='none',
                        capsize=2.5)
    for cond in df[cond_var].unique():
        df_cond = df[df[cond_var] == cond]
        ax.plot(x_pos,
                df_cond[y_var].values,
                color=df_cond[col_var].values[0],
                marker='o',
                markerfacecolor='white')
    if ceiling is not None:
        cl_x = np.arange(n_x_ticks)
        if type(ceiling) is dict:
            lwr, upr = ceiling['lower'], ceiling['upper']
            if lwr[::2] == lwr[1::2]:  # for non-sloped ceilings
                cl_x = np.repeat(np.arange(n_x_ticks + 1), 2)[1:-1] - .5
        elif len(ceiling) == n_x_ticks:
            lwr, upr = ceiling, np.ones(n_x_ticks)*max(ylims)
        else:
            lwr, upr = ceiling, max(ylims)
            cl_x = (-.5, n_x_ticks - .5)
        ax.fill_between(cl_x, lwr, upr, color='black', alpha=.2, lw=0)
    if floor is not None:
        fl_x = np.arange(n_x_ticks)
        if type(floor) is dict:
            lwr, upr = floor['lower'], floor['upper']
        elif len(floor) == n_x_ticks:
            lwr, upr = floor, np.ones(n_x_ticks) * max(ylims)
        else:
            lwr, upr = floor, min(ylims)
            fl_x = (-.5, n_x_ticks - .5)
        ax.fill_between(fl_x, lwr, upr, color='black', alpha=.2, lw=0)
    if hline is not None:
        ax.axhline(y=hline, xmin=-.5, xmax=n_x_ticks + .5, color='black',
                   lw=.25)
    ax.set_xticks(np.arange(n_x_ticks), labels=x_tick_labels)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylims)
    ax.set_title(title)
    ax.set_xlim((-.5, n_x_ticks - .5))
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def clustered_barplot(df, outpath, params, ylabel=None, x_var='region',
                      x_tick_labels=None, figsize=(3.5, 2)):
    os.makedirs(op.dirname(outpath), exist_ok=True)
    ylabel = params['ylabel'] if ylabel is None else ylabel
    df_means = df.pivot(index=x_var, columns='level', values='value')
    df_sems = df.pivot(index=x_var, columns='level', values='error').values
    df_plot = df_means.plot(kind='bar',
                            ylabel=ylabel,
                            yerr=df_sems.transpose(),
                            rot=0,
                            figsize=figsize,
                            color=params['colours'],
                            legend=False)
    fig = df_plot.get_figure()
    plt.yticks(np.arange(0,1, .1))
    plt.ylim(params['ylims'])
    plt.xlabel(None)
    if x_tick_labels:
        plt.xticks(np.arange(len(x_tick_labels)), x_tick_labels)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()

    # legend
    f = lambda m, c: \
    plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
    handles = [f('s', colour) for colour in params['colours']]
    legend = plt.legend(handles, params['labels'], loc=3)
    export_legend(legend, filename=f'{outpath[:-4]}_legend.pdf')
    plt.close()


def clustered_barplot_nc(df, outpath, params, ylabel=None, x_var='region',
                         x_tick_labels=None, figsize=(3.5, 2)):
    os.makedirs(op.dirname(outpath), exist_ok=True)
    ylabel = params['ylabel'] if ylabel is None else ylabel
    df_means = df.pivot(index=x_var, columns='level', values='value')
    df_sems = df.pivot(index=x_var, columns='level', values='error').values
    fig, ax = plt.subplots(figsize=figsize)
    ncs = df_means.iloc[:,0].values
    nc_errors = df_sems[:,0]
    nc_col = params['colours'][0]
    for x, (nc, nc_err) in enumerate(zip(ncs, nc_errors)):
        ax.axhline(y=nc, xmin=x/len(ncs)+.005, xmax=(x+1)/len(ncs)-.005,
                   color=nc_col)
        ax.fill_between((x-.5, x+.5), nc-nc_err, nc+nc_err, color=nc_col,
        alpha=.2, lw=0)
    df_means.iloc[:,1:].plot(
        kind='bar',
        ylabel=ylabel,
        yerr=df_sems[:,1:].transpose(),
        rot=0,
        figsize=figsize,
        color=params['colours'][1:],
        legend=False,
        ax=ax)

    ax.set_yticks(np.arange(0,1, .1))
    ax.set_ylim(params['ylims'])
    ax.set_xlabel(None)
    if x_tick_labels:
        plt.xticks(np.arange(len(x_tick_labels)), x_tick_labels)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()

    # legend
    f = lambda m, c: \
    plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
    handles = [plt.plot([], [], color=params['colours'][0])[0]]
    handles += [f('s', colour) for colour in params['colours'][1:]]
    labels = params['labels'].copy()
    labels[0] = (f'noise ceiling ({labels[0]})')
    legend = plt.legend(handles, params['labels'], loc=3)
    export_legend(legend, filename=f'{outpath[:-4]}_legend.pdf')
    plt.close()


def clustered_barplot_panels(df, outpath, params, ylabel, p_var='region',
                             p_order=None, x_var='subject', y_var='value',
                             p_labels=None, figsize=(6, 10)):
    os.makedirs(op.dirname(outpath), exist_ok=True)
    fig, axes = plt.subplots(ncols=1, nrows=len(df[p_var].unique()),
                           figsize=figsize)
    for p, p_level in enumerate([i for i in p_order if i in df[p_var].unique()]):
        ax = axes[p]
        dfp = df[df[p_var] == p_level]
        df_means = dfp.pivot(index=x_var, columns='level', values=y_var)
        if 'error' in dfp.columns and np.isfinite(dfp.error.values[0]):
            df_sems = dfp.pivot(index=x_var, columns='level',
                                values='error').values
            sems = df_sems.transpose()
        else:
            sems = None
        df_means.plot(kind='bar',
                      ylabel=ylabel,
                      yerr=sems,
                      rot=0,
                      color=params['colours'],
                      legend=False,
                      ax=ax)
        ax.set_yticks(np.arange(0,1, .1))
        ax.set_ylim(-.05,.4)
        ax.set_xlabel(None)
        p_label = p_labels[p] if p_labels else p_level
        ax.set_title(f'{p_var}: {p_label}')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()


def plot_RSM_models(overwrite_plots):

    # RSM plot of each regression model and occlusion robustness contrasts
    model_dir = f'derivatives/RSA/RSM_models'
    os.makedirs(model_dir, exist_ok=True)
    for model_label, model in CFG.RSM_models['matrices'].items():
        outpath = f'{model_dir}/model_{model_label}.pdf'
        if not op.isfile(outpath) or 'RSM_models' in overwrite_plots:
            RSA(RSM=model).plot_RSM(cmap='cividis', vmin=0, vmax=1,
                                    fancy=True,
                                    title=f'model: {model_label}',
                                    outpath=outpath)

    colours = ['white'] + CFG.occlusion_robustness_analyses[
        'object_completion']['colours']
    colours += CFG.occlusion_robustness_analyses[
        'occlusion_invariance']['colours']
    cmap = matplotlib.colors.ListedColormap(colours)
    for label, contrast_mat in CFG.contrast_mats.items():
        outpath = f'{model_dir}/contrasts_{label}.pdf'
        if not op.isfile(outpath) or 'RSM_models' in overwrite_plots:
            RSA(RSM=contrast_mat).plot_RSM(vmin=0, vmax=8,
                                           title=f'contrasts',
                                           cmap=cmap, outpath=outpath,
                                           fancy=True)


def get_responses(subject, region, task, space):

    thr = 3.1
    cond_labels = CFG.cond_labels['exp1']
    n_conds = len(cond_labels)
    subj_dir = f'derivatives/FEAT/sub-{subject}/subject-wise_space-{space}'

    # load localizer contrast map
    loc_path = (f'{subj_dir}/task-{task}_all-runs.gfeat/'
                f'cope25.feat/stats/zstat1.nii.gz')
    loc_data = nib.load(loc_path).get_fdata().flatten()

    # load roi mask
    if space == 'standard':
        roi_dir = f'derivatives/ROIs/MNI152_2mm'
    else:
        roi_dir = f'derivatives/ROIs/sub-{subject}/func_space'
    roi_path = f'{roi_dir}/{region}.nii.gz'
    roi_data = nib.load(roi_path).get_fdata().flatten()
    n_voxels_roi = np.count_nonzero(roi_data)

    # ensure localizer and ROI images have the same orientation
    loc_info = os.popen(f'fslhd {loc_path}').readlines()
    roi_info = os.popen(f'fslhd {roi_path}').readlines()
    for axis in ['x','y','z']:
        loc_ori = [info.split()[1] for info in loc_info if \
                   info.startswith(f'qform_{axis}orient')][0]
        roi_ori = [info.split()[1] for info in roi_info if \
                   info.startswith(f'qform_{axis}orient')][0]
        assert loc_ori == roi_ori, 'localizer and ROI images are not aligned.'
    #for form in 'sq':
    #    loc_form = os.popen(f'fslorient -get{form}form {loc_path}').read()
    #    roi_form = os.popen(f'fslorient -get{form}form {roi_path}').read()
    #    assert loc_form == roi_form, 'localizer and ROI images are not
    #    aligned.'

    # restrict roi mask to all voxels above threshold in localizer contrast
    mask = np.array((roi_data * loc_data) >= thr, dtype=int)
    n_voxels = np.count_nonzero(mask)

    if n_voxels > 8:

        # collate voxelwise responses
        n_splits = len(glob.glob(f'{subj_dir}/task-{task}_split*A.gfeat'))
        responses = np.empty((n_splits, 2, n_conds, n_voxels))
        for sp in range(n_splits):
            for si, side in enumerate(['A', 'B']):
                cope_paths = [
                    f'{subj_dir}/task-{task}_split-{sp}{side}.'
                    f'gfeat/cope{c + 1}.feat/stats/cope1.nii.gz'
                    for c in range(CFG.n_img)]
                for c, cope_path in enumerate(cope_paths):
                    cope_data = nib.load(cope_path).get_fdata().flatten()
                    responses[sp, si, c, :] = cope_data[mask == 1]

        # remove voxels with zeros in any run
        cond_mean = np.mean(responses, axis=2)
        zero_voxels = np.any(cond_mean == 0, axis=(0, 1))
        n_voxels_removed = np.sum(zero_voxels)
        if n_voxels_removed:
            responses = responses[:, :, :, ~zero_voxels]
        final_voxels = responses.shape[-1]
        print(f'{subject} {region}, {final_voxels}/{n_voxels_roi} voxels '
              f'selected ({n_voxels_removed} voxels w/o full set of responses)')

    else:
        print(f'{subject} {region} has too few voxels ({n_voxels}/'
              f'{n_voxels_roi})')
        responses = None

    return responses


class RSA:

    def __init__(self, RSM, similarity='pearson'):
        self.RSM = RSM
        self.similarity = similarity

    def plot_RSM(self, vmin=-1, vmax=1, cmap='rainbow', title='',
                 labels=CFG.cond_labels['exp1'], outpath='temp.png',
                 measure='', fancy=False):

        os.makedirs(op.dirname(outpath), exist_ok=True)
        n_conds = len(labels)
        if not fancy:
            fig, ax = plt.subplots(
                figsize=(7 * (n_conds / 24), 5.25 * (n_conds / 24)))
            im = ax.imshow(self.RSM, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.tick_params(**{'length': 0})
            ax.set_xticks(np.arange(self.RSM.shape[0]))
            ax.set_yticks(np.arange(self.RSM.shape[1]))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.tick_params(direction='in')
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            fig.colorbar(im, fraction=0.0453)
            ax.set_title(title)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            plt.text(24, 15, measure, rotation='vertical', fontsize=12)
            fig.tight_layout()
            plt.savefig(outpath)
            plt.close()

        else:

            imx = f'{PROJ_DIR}/data/in_vivo/fMRI/RSM_pictures_x.png'
            imy = f'{PROJ_DIR}/data/in_vivo/fMRI/RSM_pictures_y.png'
            picx = plt.imread(imx)
            picy = plt.imread(imy)
            fig, unused_ax = plt.subplots(figsize=(7, 5.25))
            unused_ax.axis('off')
            ax = fig.add_axes([.1, .22, .75, .75])
            im = ax.imshow(self.RSM, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.tick_params(**{'length': 0})
            ax.set_xticks(np.arange(self.RSM.shape[0]))
            ax.set_yticks(np.arange(self.RSM.shape[1]))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(direction='in')
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            fig.colorbar(im, fraction=0.0453)
            ax.set_title('')
            plt.text(24, 15, measure, rotation='vertical', fontsize=12)
            newax = fig.add_axes([-.27, 0.2, .79, .79])
            newax.imshow(picy)
            newax.tick_params(**{'length': 0})
            newax.spines['bottom'].set_visible(False)
            newax.spines['left'].set_visible(False)
            newax.set_xticks([])
            newax.set_yticks([])
            newax2 = fig.add_axes([0.2, -0.2, .6, .6])
            newax2.imshow(picx)
            newax2.tick_params(**{'length': 0})
            newax2.spines['bottom'].set_visible(False)
            newax2.spines['left'].set_visible(False)
            newax2.set_xticks([])
            newax2.set_yticks([])
            plt.savefig(outpath)
            plt.close()


    def plot_MDS(self, title=None, outpath='temp.png', figsize=(2,2)):

        os.makedirs(op.dirname(outpath), exist_ok=True)

        # average across diagonal
        RSM = (self.RSM + self.RSM.transpose()) / 2

        # invert RSM to get RDM
        if self.similarity in ['pearson', 'spearman']:
            RSM *= -1

        # Instantiate MDS class, fit using correlations similarity measure
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        Y = mds.fit_transform(RSM)

        colours = [col for col in TABCOLS[:8] for _ in range(3)]
        markers = ['o', 'v', '^'] * 8
        #edgecolours = ['w'] * 24
        labels = CFG.cond_labels['exp1']

        """
        if exp == 'exp2':
            colours += colours
            markers += markers
            tab20c = matplotlib.cm.tab20c.colors
            edgecolours = [tab20c[19]] * 24 + [tab20c[16]] * 24
        """

        # Plot
        # cm = matplotlib.cm.gist_rainbow(np.linspace(0, 1, 8))
        fig, ax = plt.subplots(figsize=figsize)
        for c in range(Y.shape[0]):
            ax.plot(Y[c, 0],
                    Y[c, 1],
                    color=colours[c],
                    marker=markers[c],
                    #markeredgecolor=edgecolours[c],
                    label=labels[c])
        #ax.axis('scaled')
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches='tight')
        plt.close(fig)

        # save legend separately
        colours = [[0., 0., 0., 1.]] * 3 + [[1., 1., 1., 1.]] + TABCOLS[:9]
        shapes = ['o', 'v', '^'] + ['s'] * 9
        labels = CFG.occluder_labels + [''] + CFG.exemplars
        f = lambda m, c: plt.plot([], [], marker=m, color=c, linestyle='None')[
            0]
        handles = [f(shape, colour) for colour, shape in zip(colours, shapes)]
        legend = plt.legend(handles, labels, loc=3)
        export_legend(legend, filename=f'{op.dirname(outpath)}/legend.pdf')

    def RSM_to_table(self):
        conds = CFG.cond_labels['exp1']
        exemplars_a = [c.split('_')[0] for c in conds for _ in conds]
        occluders_a = [c.split('_')[1] for c in conds for _ in conds]
        exemplars_b = [c.split('_')[0] for _ in conds for c in conds]
        occluders_b = [c.split('_')[1] for _ in conds for c in conds]
        analyses = []
        levels = []

        for ea, oa, eb, ob in zip(exemplars_a, occluders_a, exemplars_b,
                                  occluders_b):

            # object completion
            if oa != 'none' and ob != 'none':
                analyses.append('object_completion')
                if ea == eb:
                    levels.append('EsOs') if oa == ob else levels.append('EsOd')
                else:
                    levels.append('EdOs') if oa == ob else levels.append('EdOd')

            # occlusion invariance
            else:
                analyses.append('occlusion_invariance')
                if ea == eb:
                    levels.append(
                        'EsUb') if oa == 'none' and ob == 'none' else levels.append(
                        'EsU1')
                else:
                    levels.append(
                        'EdUb') if oa == 'none' and ob == 'none' else levels.append(
                        'EdU1')

        self.RSM_table = pd.DataFrame({
            'exemplar_a': exemplars_a,
            'occluder_a': occluders_a,
            'exemplar_b': exemplars_b,
            'occluder_b': occluders_b,
            'analysis': analyses,
            'level': levels,
            'similarity': self.RSM.flatten(),
        })
        self.RSM_table.occluder_a = self.RSM_table.occluder_a.astype(
            'category').cat.reorder_categories(CFG.occluders)
        self.RSM_table.occluder_b = self.RSM_table.occluder_b.astype(
            'category').cat.reorder_categories(CFG.occluders)
        level_order = CFG.occlusion_robustness_analyses['object_completion'][
                          'conds'] + \
                      CFG.occlusion_robustness_analyses['occlusion_invariance'][
                          'conds']
        self.RSM_table.level = self.RSM_table.level.astype(
            'category').cat.reorder_categories(level_order)

    def calculate_occlusion_robustness(self):

        self.occlusion_robustness = pd.DataFrame()

        for analysis, params in CFG.occlusion_robustness_analyses.items():
            values = [self.RSM_table['similarity'][
                          self.RSM_table['level'] == level].mean() for level in
                      params['conds']]

            # calculate index
            A, B, C, D = values
            E = np.mean([C, D])

            for index_type in ['norm', 'prop', 'base', 'base2', 'rel']:
                if index_type == 'norm':
                    index_value = 1 - (A - B) / (abs(A) + abs(B))
                elif index_type == 'prop':
                    index_value = B / A
                elif index_type == 'base':
                    index_value = (B - D) / (A - D)
                elif index_type == 'base2':
                    index_value = (B - E) / (A - E)
                elif index_type == 'rel':
                    index_value = ((B - D) - (C - D)) / (A - D)

                self.occlusion_robustness = pd.concat(
                    [self.occlusion_robustness, pd.DataFrame({
                        'analysis': [analysis],
                        'subtype': [index_type],
                        'index': [params['index_label']],
                        'value': [index_value],
                    })]).reset_index(drop=True)

    def fit_models(self):

        # regression models
        self.model_fits = pd.DataFrame()
        for model_label, RSM_model in CFG.RSM_models['matrices'].items():

            # flatten matrices
            model_flat = RSM_model.flatten()
            similarity_flat = self.RSM_table.similarity.values

            # remove elements where model is nan
            model_flat_finite = model_flat[np.isfinite(model_flat)]
            similarity_flat_finite = similarity_flat[np.isfinite(model_flat)]

            # run regression
            regr = LinearRegression()
            fit = regr.fit(model_flat_finite.reshape(-1, 1),
                           similarity_flat_finite.reshape(-1, 1)).coef_[0][0]
            pred = regr.predict(model_flat_finite.reshape(-1, 1))
            mse = mean_squared_error(pred, similarity_flat_finite)

            self.model_fits = pd.concat(
                [self.model_fits, pd.DataFrame({'model': [model_label],
                                                'beta': [fit],
                                                'mse': [mse]})]).reset_index(
                drop=True)

        # compare exemplar (occluded only) and occluder position model fits
        exem = self.model_fits[
            self.model_fits.model == 'exemplar_bothocc'].beta.item()
        occl = self.model_fits[
            self.model_fits.model == 'occluder_position'].beta.item()
        self.exemplar_v_occluder_position = (exem - occl) / (exem + occl)

        # compare exemplar (< 2 occ) and occluder presence model fits
        exem = self.model_fits[
            self.model_fits.model == 'exemplar_lt2occ'].beta.item()
        occl = self.model_fits[
            self.model_fits.model == 'occluder_presence_lt2occ'].beta.item()
        self.exemplar_v_occluder_presence = (exem - occl) / (exem + occl)


    def analyse(self):

        self.RSM_to_table()
        self.calculate_occlusion_robustness()
        self.fit_models()


class RSA_dataset:

    """
    Assumes format of responses based on number of dimensions
    2D = single set of estimates (cond x chan)
    3D = multiple sets of estimates (rep x cond x chan)
    4D = data are already split and averaged (split x side x cond x chan)
    """

    def __init__(self, responses=None):

        # check responses
        assert responses is not None, "No responses provided"
        self.initial_shape = responses.shape
        nchan = self.initial_shape[-1]

        # get data format
        n_conds, n_chan = responses.shape[-2:]

        # for single set of estimates
        if len(responses.shape) == 2:
            self.responses = np.tile(responses, (2, 1, 1))[None, :]

        # for multiple sets of estimates
        elif len(responses.shape) == 3:
            n_splits, n_reps = 8, responses.shape[0]
            self.responses = np.empty(
                (n_splits, 2, n_conds, n_chan))
            for split in range(n_splits):
                resp_a, resp_b = train_test_split(
                    responses, test_size=0.5, random_state=split)
                self.responses[split, 0] = np.mean(resp_a, axis=0)
                self.responses[split, 1] = np.mean(resp_b, axis=0)

        # for presplit data
        else:
            self.responses = responses

        # remove empty channels
        self.responses = self.responses[
             :, :, :, ~np.all(self.responses == 0, axis=(0,1,2))]
        #nchan_final = self.responses.shape[-1]
        #if nchan != nchan_final:
            #nchan_removed = nchan - nchan_final
            #perc = nchan_removed / nchan * 100
            #print(f'{nchan_removed}/{nchan} ({int(perc)}%) channels removed '
            #      f'as they are empty across all conditions')

    def plot_TSNE(self, outpath, figsize=(2,2)):

        os.makedirs(op.dirname(outpath), exist_ok=True)
        exp = 'exp1' if self.responses.shape[-2] == 24 else 'exp2'

        # get mean across runs, splits, and sides if present
        responses = self.responses.copy()
        while len(responses.shape) > 2:
            responses = np.mean(responses, axis=0)

        tsne = TSNE(n_components=2, perplexity=10, n_iter=300,
                    learning_rate="auto")
        TSNE_weights = tsne.fit_transform(responses)

        colours = [col for col in TABCOLS[:8] for _ in range(3)]
        markers = ['o', 'v', '^'] * 8
        edge_colours = ['w'] * 24
        labels = CFG.cond_labels[exp]
        tab20c = matplotlib.cm.tab20c.colors
        if exp == 'exp2':
            colours += colours
            markers += markers
            edge_colours = [tab20c[19]] * 24 + [tab20c[16]] * 24

        fig, ax = plt.subplots(figsize=figsize)
        for c in range(responses.shape[0]):
            ax.plot(TSNE_weights[c, 0],
                    TSNE_weights[c, 1],
                    color=colours[c],
                    marker=markers[c],
                    markeredgecolor=edge_colours[c],
                    label=labels[c])
        ax.axis('scaled')
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)

        # save legend separately
        colours = [[0., 0., 0., 1.]] * 3 + [[1., 1., 1., 1.]] + TABCOLS[:9]
        edge_colours = ['w'] * 24
        shapes = ['o', 'v', '^'] + ['s'] * 9
        labels = CFG.occluders + [''] + CFG.exemplars
        if responses.shape[0] == 48:  # for exp 2
            colours = ['w'] * 3 + colours
            shapes = ['o'] * 3 + shapes
            labels = CFG.attns + [''] + labels
            edge_colours = [tab20c[19], tab20c[16], 'w'] + edge_colours
        f = lambda m, c, e: \
        plt.plot([], [], marker=m, color=c, markeredgecolor=e,
                 linestyle='None')[0]
        handles = [f(shape, colour, edgecolour) for colour, shape, edgecolour in
                   zip(colours, shapes, edge_colours)]
        legend = plt.legend(handles, labels, loc=3)
        export_legend(legend, filename=f'{op.dirname(outpath)}/legend.pdf')


    def calculate_RSM(self, norm, norm_method, similarity):

        responses = self.responses
        n_exem, n_img, n_attn, n_occ = CFG.n_exem, CFG.n_img, 2, 3
        n_splits, n_reps, n_conds, n_chan = self.responses.shape

        if similarity != 'crossnobis':

            # calculate mean and std for normalization
            # if left as is, these will have no effect, i.e. no normalization
            norm_data = {'mean': np.zeros_like(responses),
                         'std': np.ones_like(responses)}

            for op in norm_data:

                np_func = getattr(np, op)

                # use summary statistics across all conditions
                if norm == 'all-conds':
                    norm_data[op] = np.tile(np_func(
                        responses, axis=2, keepdims=True), (1, 1, n_conds, 1))

                # use stats for each occluder condition
                elif norm == 'occluder':
                    for occ in range(n_occ):
                        norm_data[op][:, :, occ::n_occ, :] = np.tile(np_func(
                            responses[:, :, occ::n_occ, :], axis=2,
                            keepdims=True), (1, 1, n_exem, 1))

                # use stats for unoccluded images only
                elif norm == 'unoccluded':
                    norm_data[op] = np.tile(
                        np_func(responses[:, :, :n_occ, :], axis=2,
                                keepdims=True), (1, 1, n_img, 1))

            # remove channels with no variance

            if norm_method == 'z-score' and (norm_data['std'] == 0).any():

                img_dim = len(norm_data['std'].shape) - 2
                good_channels = ~np.all(norm_data['std'] == 0, axis=img_dim)
                while len(good_channels.shape) > 1:
                    good_channels = good_channels.min(axis=-2)
                norm_data['std'] = norm_data['std'][..., good_channels]
                norm_data['mean'] = norm_data['mean'][..., good_channels]
                responses = responses[..., good_channels]

                #n_chan_final = good_channels.sum()
                #n_chan_removed = n_chan - n_chan_final
                #perc = n_chan_removed / n_chan * 100
                #print(
                #    f'{n_chan_removed}/{n_chan} ({int(perc)}%) channels '
                #    f'removed as they have zero variance')

            # apply normalization
            responses_norm = responses - norm_data['mean']
            if norm_method == 'z-score':
                responses_norm /= norm_data['std']

            # calculate RSMs
            RSMs_split = np.empty((n_splits, n_conds, n_conds))

            for split in range(n_splits):

                if similarity == 'pearson':
                    RSMs_split[split] = np.corrcoef(
                        responses_norm[split, 0], responses_norm[split, 1])[
                                        :n_conds, n_conds:]

                elif similarity == 'spearman':
                    RSMs_split[split] = spearmanr(
                        responses_norm[split, 0], responses_norm[split, 1],
                        axis=1).correlation[:n_conds, n_conds:]

                # Euclidean distance, normalized by number of channels
                # Equivalent to the Euclidean method in rsatoolbox
                if similarity == 'euclidean':
                    RSMs_split[split] = np.sqrt(euclidean_distances(
                        responses_norm[split, 0], responses_norm[split, 1],
                        squared=True) / n_chan)

            # mean across splits
            RSM = np.mean(RSMs_split, axis=0)


        else:

            # crossnobis
            assert len(self.initial_shape) == 3
            responses_flat = self.responses.reshape((-1, n_chan))
            conds = np.array([f'{l + 1:02}_{label}' for l, label in
                              enumerate(CFG.cond_labels['exp1'])] * n_reps)
            runs = np.repeat(np.arange(n_reps), n_conds)
            obs_des = {'conds': conds, 'runs': runs}
            chn_des = {'voxels': np.array(
                [f'voxel_{x:06}' for x in np.arange(n_chan)])}
            dataset = rsatoolbox.data.Dataset(
                measurements=responses_flat,
                obs_descriptors=obs_des,
                channel_descriptors=chn_des)
            noise = rsatoolbox.data.noise.prec_from_measurements(
                dataset, obs_desc='conds', method='shrinkage_diag')
            rdm = rsatoolbox.rdm.calc_rdm(
                dataset, descriptor='conds', method='crossnobis', noise=noise,
                cv_descriptor='runs')
            RSM = rdm.get_matrices()[0]

        return RSA(RSM, similarity)
    

def compare_regions(
        RSAs, analysis_dir, overwrite_data, overwrite_plots, overwrite_stats):

    exp = op.basename(os.getcwd())
    subjects = CFG.subjects[exp]
    subjects_final = CFG.subjects_final[exp]
    fw_base, fw_mult, fh = .5, .4, 2.5  # figure size

    # conditions-wise similarities
    for analysis, params in CFG.occlusion_robustness_analyses.items():

        out_dir = f'{analysis_dir}/{analysis}'
        os.makedirs(out_dir, exist_ok=True)

        # data
        conditions_path = f'{out_dir}/cond-wise_sims.csv'
        if not op.isfile(conditions_path) or overwrite_data:

            print(f'Collating {analysis} condition-wise '
                  f'similarities...')

            df = pd.DataFrame()
            for region, subject in itertools.product(CFG.regions, subjects):
                temp = RSAs[region][subject]
                if temp is not None:
                    temp = temp.RSM_table
                    temp['region'] = region
                    temp['subject'] = subject
                    df = pd.concat([df, temp])

            df = df[df.analysis == analysis].drop(columns=[
                'exemplar_a', 'exemplar_b', 'occluder_a','occluder_b']).groupby([
                'analysis', 'level', 'region', 'subject']).agg(
                'mean').dropna().reset_index()

            # reduce to final subject sample and get group means, sems
            df_summary = df[df.subject.isin(subjects_final)].drop(
                columns=['subject']).groupby(
                ['analysis', 'level', 'region']).agg(
                ['mean', 'sem']).dropna().reset_index()
            df_summary.columns = [
                'analysis', 'level', 'region', 'value','error']

            # remove regions for which summary statistics cannot be calculated
            df = df[df.region.isin(df_summary.region.unique())].reset_index(
                drop=True)

            # save out
            df.to_csv(conditions_path, index=False)
            df_summary.to_csv(f'{out_dir}/cond-wise_sims_summary.csv',
                              index=False)
        else:
            df = pd.read_csv(conditions_path)
            df_summary = pd.read_csv(f'{out_dir}/cond-wise_sims_summary.csv')

        # subject-wise plot for each region_set (one panel per region)
        for region_set, regions in CFG.region_sets.items():
            outpath = (f'{analysis_dir}/{analysis}/'
                       f'{region_set}/cond-wise_sims_ind.png')
            if not op.isfile(outpath) or overwrite_plots:
                print(f'Plotting subject-wise {analysis} condition-wise '
                      f'similarities in {region_set}...')
                plot_df = df[(df.analysis == analysis) &
                             (df.region.isin(regions))].copy(
                             deep=True)
                reg_ord = [r for r in regions if r in plot_df.region.unique()]
                plot_df.region = pd.Categorical(
                    plot_df.region, categories=reg_ord, ordered=True)
                plot_df.level = pd.Categorical(
                    plot_df.level, categories=params['conds'], ordered=True)
                plot_df.sort_values(['region', 'level'], inplace=True)
                ylabel = CFG.similarities[op.basename(analysis_dir)]
                p_labels = [CFG.regions[r] for r in reg_ord]
                clustered_barplot_panels(
                    plot_df, outpath, params, ylabel, y_var='similarity',
                    p_order=reg_ord, p_labels=p_labels)

        # region-wise plot for each region_set
        for region_set, regions in CFG.region_sets.items():
            outpath = f'{out_dir}/{region_set}/cond-wise_sims.png'
            outpath_nc = f'{out_dir}/{region_set}/cond-wise_sims_nc.png'
            if (not op.isfile(outpath) or not op.isfile(outpath_nc) or
                    overwrite_plots):
                print(f'Plotting {analysis} condition-wise '
                      f'similarities in {region_set}...')
                plot_df = df_summary[
                    (df_summary.analysis == analysis) &
                    (df_summary.region.isin(regions))].copy(deep=True)
                reg_ord = [r for r in regions if r in plot_df.region.unique()]
                plot_df.region = pd.Categorical(
                    plot_df.region, categories=reg_ord, ordered=True)
                plot_df.level = pd.Categorical(
                    plot_df.level, categories=params['conds'], ordered=True)
                plot_df.sort_values(['region','level'], inplace=True)
                ylabel = CFG.similarities[op.basename(analysis_dir)]
                params['ylims'] = [-.05, .25] if (analysis ==
                                                 'object_completion') else \
                    [-.05, .32]
                x_tick_labels = [CFG.regions[r] for r in reg_ord]

                # all conditions as bars with errors
                clustered_barplot(
                    plot_df, outpath, params, ylabel,
                    x_tick_labels=x_tick_labels,
                    figsize=(fw_base + len(regions)*fw_mult, fh))

                # first cond is noise ceiling
                clustered_barplot_nc(
                    plot_df, outpath_nc, params, ylabel,
                    x_tick_labels=x_tick_labels,
                    figsize=(fw_base + len(regions)*fw_mult, fh))

        # region-wise stats for each region_set
        for region_set, regions in CFG.region_sets.items():
            outpath = f'{out_dir}/{region_set}/cond-wise_sims_anova.csv'
            if not op.isfile(outpath) or overwrite_stats:

                print(f'Performing {analysis} stats for condition-wise '
                      f'similarities in {region_set}...')

                # get stats and calculate z-scores
                stats_df = df[
                    (df.region.isin(regions)) &
                    (df.subject.isin(subjects_final))].copy(
                    deep=True).groupby(['subject','region','level']).agg(
                    'mean', numeric_only=True).dropna().reset_index()
                stats_df['similarity_z'] = np.arctanh(
                    stats_df['similarity'])
                stats_df = stats_df.drop(columns=['similarity'])

                # run statistical tests
                try:

                    # anova (same and different objects)
                    anova = pg.rm_anova(
                        dv='similarity_z', within=['level','region'],
                        subject='subject', data=stats_df, detailed=True)
                    anova.to_csv(outpath, index=False)

                    # post hocs
                    post_hocs = pg.pairwise_tests(
                        dv='similarity_z', within=['region','level'],
                        subject='subject', data=stats_df, padjust='none',
                        return_desc=True, effsize='cohen')
                    post_hocs = post_hocs[   # restrict to interaction effects
                        post_hocs.Contrast == 'region * level'].copy(deep=True)
                    p_corr = np.empty(len(post_hocs))  # correct across ROIs
                    for c in range(6):
                        _, p_corr[c::6] = pg.multicomp(
                            post_hocs['p-unc'][c::6], method='holm')
                    post_hocs['p-corr'] = p_corr
                    post_hocs.to_csv(outpath.replace('anova', 'posthocs'),
                                     index=False)

                    # anova (same object only)
                    stats_df_same_object = stats_df[
                        stats_df.level.isin(params['conds'][:2])].copy(
                        deep=True)
                    anova = pg.rm_anova(
                        dv='similarity_z', within=['level', 'region'],
                        subject='subject', data=stats_df_same_object, detailed=True)
                    anova.to_csv(outpath.replace('anova', 'anova_same_object'),
                                 index=False)

                    # post hocs
                    post_hocs = pg.pairwise_tests(
                        dv='similarity_z', within=['region', 'level'],
                        subject='subject', data=stats_df_same_object,
                        padjust='none', return_desc=True, effsize='cohen')
                    #post_hocs = post_hocs[ # restrict to interaction effects
                    #    post_hocs.Contrast == 'region * level'].copy(deep=True)
                    p_corr = np.empty(len(post_hocs))   # correct across ROIs
                    for c in range(6):
                        _, p_corr[c::6] = pg.multicomp(
                            post_hocs['p-unc'][c::6], method='holm')
                    post_hocs['p-corr'] = p_corr
                    post_hocs.to_csv(outpath.replace(
                        'anova', 'posthocs_same_object'), index=False)

                    stats_df_same_object_diff = stats_df_same_object.groupby(
                        ['subject', 'region']).apply(lambda df: pd.Series(
                            df.similarity_z[df.level == params['conds'][
                                1]].values[0] -
                            df.similarity_z[df.level == params['conds'][
                                0]].values[0],
                            index=['diff'])).reset_index()
                    post_hocs_interaction = pg.pairwise_ttests(
                        dv='diff', within='region', subject='subject',
                        data=stats_df_same_object_diff, padjust='holm',
                        effsize='cohen')
                    post_hocs_interaction.to_csv(outpath.replace(
                        'anova', 'posthocs_same_object_interaction'),
                        index=False)

                    stats_df_same_object_prop = stats_df_same_object.groupby(
                        ['subject', 'region']).apply(lambda df: pd.Series(
                        df.similarity_z[df.level == params['conds'][
                            1]].values[0] /
                        stats_df_same_object.similarity_z[stats_df_same_object.level == params['conds'][
                            0]].mean(),
                        index=['prop'])).reset_index()
                    post_hocs_interaction = pg.pairwise_tests(
                        dv='prop', within='region', subject='subject',
                        data=stats_df_same_object_prop, padjust='holm')
                    post_hocs_interaction.to_csv(outpath.replace(
                        'anova', 'posthocs_same_object_interaction_prop'),
                        index=False)

                    # t-tests of key contrast against 0
                    stats_df_key_contrast = stats_df[
                        stats_df.level == params['conds'][1]].copy(deep=True)
                    one_sided = stats_df_key_contrast.groupby('region').apply(
                        lambda df: pg.ttest(df.similarity_z, 0)).reset_index()
                    one_sided['mu'] = 0
                    one_sided['p-corr'] = pg.multicomp(
                        one_sided['p-val'], method='holm')[1]
                    one_sided.to_csv(outpath.replace(
                        'anova', 'one_sample'), index=False)

                except:
                    pass


    # occlusion robustness indices (object completion and occlusion invariance)
    for analysis, params in CFG.occlusion_robustness_analyses.items():

        out_dir = f'{analysis_dir}/{analysis}'
        os.makedirs(out_dir, exist_ok=True)

        # data
        indices_path = f'{out_dir}/indices.csv'
        if not op.isfile(indices_path) or overwrite_data:
            print(f'Collating {analysis} indices...')
            cond_sims = pd.read_csv(f'{out_dir}/cond-wise_sims.csv')
            df = pd.DataFrame()
            for level, subject_sample in zip(
                    ['ind', 'group'], [subjects_final, ['group']]):
                for region in CFG.regions:
                    df_region = pd.DataFrame()
                    for subject in subject_sample:
                        temp = RSAs[region][subject]
                        if temp is not None:
                            temp = temp.occlusion_robustness
                            temp = temp[temp.analysis == analysis].copy(deep=True)
                            temp = temp.drop(columns='analysis')
                            temp['level'] = level
                            temp['region'] = region
                            temp['subject'] = subject
                            df_region = pd.concat([df_region, temp])

                    # new index, proportion, normed by group mean values
                    if len(df_region) and 'prop2' not in df_region.subtype.unique():
                        cond_sims_groupmean = (cond_sims[
                            (cond_sims.region == region) &
                            (cond_sims.subject.isin(subjects_final))]
                            .groupby(['level', 'region'])
                            .agg({'similarity': 'mean'})
                            .reset_index())
                        ceil = (cond_sims_groupmean
                            [cond_sims_groupmean.level == params['conds'][0]]
                            .similarity.mean())
                        for subject in subject_sample:
                            if subject == 'group':
                                cond_sims_subject = cond_sims_groupmean
                            else:
                                cond_sims_subject = cond_sims[
                                    (cond_sims.region == region) &
                                    (cond_sims.subject == subject)]
                            if len(cond_sims_subject):
                                value = (cond_sims_subject[
                                    cond_sims_subject.level == params['conds'][1]]
                                    .similarity.item()) / ceil
                                df_region = pd.concat([df_region, pd.DataFrame({
                                    'subtype': ['prop2'],
                                    'index': [params['index_label']],
                                    'value': [value],
                                    'level': [level],
                                    'region': [region],
                                    'subject': [subject]})])
                    df = pd.concat([df, df_region])
            df.sort_values(['region', 'subject', 'level', 'subtype'],
                           inplace=True)
            df.to_csv(indices_path, index=False)

        # plots
        df = pd.read_csv(indices_path)
        for (level, subject_sample), subtype, (region_set, regions) in \
                itertools.product(
                    zip(['ind', 'group'], [subjects_final, ['group']]),
                    df.subtype.unique(), CFG.region_sets.items()):
            outpath = (f'{out_dir}/{region_set}/{params["index_label"]}'
                       f'_{subtype}_{level}.png')

            if not op.isfile(outpath) or overwrite_plots:
                print(f'Plotting {level} {analysis} indices ({subtype}) in'
                      f' {region_set}...')
                plot_df = df[
                    (df.level == level) &
                    (df.subtype == subtype) &
                    (df.region.isin(regions)) &
                    (df.subject.isin(subject_sample))].copy(
                    deep=True).reset_index(drop=True).drop(columns=[
                    'level', 'subtype', 'subject'])
                if level == 'ind':
                    plot_df = plot_df.groupby(['index', 'region']).agg([
                        'mean', 'sem'], numeric_only=True).dropna(
                        ).reset_index()
                    plot_df.columns = plot_df.columns[:2].droplevel(
                        1).tolist() + ['value', 'error']
                else:
                    plot_df['error'] = np.nan


                reg_ord = [r for r in regions if r in plot_df.region.unique()]
                plot_df.region = pd.Categorical(
                    plot_df.region, categories=reg_ord, ordered=True)
                plot_df.sort_values('region', inplace=True)


                # remove unqualified ROIs and set significance bars
                ph_path = f'{out_dir}/{region_set}/cond-wise_sims_posthocs.csv'
                if op.isfile(ph_path):
                    ph = pd.read_csv(ph_path)
                    ceil_sigs, floor_sigs, batches, b = [], [], [], []
                    for r, region in enumerate(reg_ord):

                        if region in ph.region.unique():
                            floor_sigs.append(ph['p-corr'][
                                                 (ph.region == region) &
                                                 (ph.A == params['conds'][3]) &
                                                 (ph.B == params['conds'][1])].item()
                                             < .05)
                            ceil_sigs.append(ph['p-corr'][
                                                  (ph.region == region) &
                                                  (ph.A == params['conds'][1]) &
                                                  (ph.B == params['conds'][
                                                      0])].item() < .05)
                            if True:#ph['p-corr'][
                                    #(ph.region == region) &
                                    #(ph.A == params['conds'][3]) &
                                    #(ph.B == params['conds'][0])].item() < .05:
                                b.append(r)
                            else:
                                if len(b):
                                    batches.append(b)
                                    b = []
                        else:
                            if len(b):
                                batches.append(b)
                                b = []
                    if len(b):
                        batches.append(b)
                else:
                    batches = []

                # plot
                x_tick_labels = [CFG.regions[r] for r in reg_ord]
                n_x_ticks = len(x_tick_labels)
                yticks = (0,.5,1)
                ylims = (0, 1.1)
                os.makedirs(op.dirname(outpath), exist_ok=True)
                fig, ax = plt.subplots(
                    figsize=(fw_base + len(regions) * fw_mult, fh))
                if level == 'ind':
                    for batch in batches:
                        ax.errorbar(batch,
                                    plot_df['value'].values[batch],
                                    yerr=plot_df['error'].values[batch],
                                    color='tab:gray',
                                    linestyle='none',
                                    capsize=2.5)
                for batch in batches:
                    ax.plot(batch,
                        plot_df['value'].values[batch],
                        color='tab:grey',
                        marker='o',
                        markerfacecolor='white')
                for x_pos in range(n_x_ticks):
                    xmin = (x_pos + .2) / n_x_ticks
                    xmax = xmin + (.6 / n_x_ticks)
                    if x_pos in [b for batch in batches for b in batch]:
                        if floor_sigs[x_pos]:
                            ax.axhline(y=.05, xmin=xmin, xmax=xmax, lw=1,
                                color=params['colours'][3])
                        if ceil_sigs[x_pos]:
                            ax.axhline(y=.95, xmin=xmin, xmax=xmax, lw=1,
                                color=params['colours'][0])
                    else:
                        ax.axhline(y=.5, xmin=xmin, xmax=xmax, lw=1,
                            color='tab:purple')

                lwr, upr = 1., max(ylims)
                cl_x = (-.5, n_x_ticks - .5)
                ax.fill_between(cl_x, lwr, upr, color='black', alpha=.2, lw=0)
                ax.set_xticks(np.arange(n_x_ticks), labels=x_tick_labels)
                ax.set_yticks(yticks)
                ax.set_ylabel(params['index_label'])
                ax.set_ylim(ylims)
                ax.set_title(f'{analysis.replace("_", " ")} index')
                ax.set_xlim((-.5, n_x_ticks - .5))
                plt.tight_layout()
                plt.savefig(outpath, dpi=300)
                plt.close()

        # stats
        for subtype, (region_set, regions) in itertools.product(
                df.subtype.unique(), CFG.region_sets.items()):
            outpath = (f'{out_dir}/{region_set}/{params["index_label"]}'
                       f'_{subtype}_ind_anova.csv')
            if not op.isfile(outpath) or overwrite_stats:
                print(f'Performing stats on {analysis} indices ({subtype}) in'
                      f' {region_set}...')
                stats_df = df[
                    (df.level == 'ind') &
                    (df.subtype == subtype) &
                    (df.region.isin(regions)) &
                    (df.subject.isin(subjects_final))].copy(deep=True).drop(
                    columns=['level','subtype','index'])
                try:
                    # t-tests against 0 and 1
                    one_sided_floor = stats_df.groupby('region').apply(
                        lambda df: pg.ttest(df.value, 0)).reset_index()
                    one_sided_floor['mu'] = 0
                    one_sided_floor['p-corr'] = pg.multicomp(
                        one_sided_floor['p-val'], method='holm')[1]
                    one_sided_ceiling = stats_df.groupby('region').apply(
                        lambda df: pg.ttest(df.value, 1)).reset_index()
                    one_sided_ceiling['mu'] = 1
                    one_sided_ceiling['p-corr'] = pg.multicomp(
                        one_sided_ceiling['p-val'], method='holm')[1]
                    one_sided = pd.concat([one_sided_floor, one_sided_ceiling])
                    one_sided.to_csv(outpath.replace('anova', 'one_sided'))

                    # one-way AVOVA, effect of region
                    anova = pg.rm_anova(
                        dv='value', within='region', subject='subject',
                        data=stats_df, detailed=True)
                    anova.to_csv(outpath)
                    post_hocs = pg.pairwise_tests(
                        dv='value', within='region', subject='subject',
                        data=stats_df, padjust='holm',
                        return_desc=True, effsize='cohen')
                    post_hocs.to_csv(outpath.replace('anova', 'posthocs'))

                except:
                    pass


    # regression models

    # data
    regression_path = f'{analysis_dir}/regression/regression.csv'
    os.makedirs(op.dirname(regression_path), exist_ok=True)
    if not op.isfile(regression_path) or overwrite_data:
        print(f'Collating regression results...')
        df = pd.DataFrame()
        for level, subject_sample in zip(['ind', 'group'],
                                         [subjects_final, ['group']]):
            for region, subject in itertools.product(
                    CFG.regions, subject_sample):
                temp = RSAs[region][subject]
                if temp is not None:
                    temp = temp.model_fits
                    temp['level'] = level
                    temp['region'] = region
                    temp['subject'] = subject
                    df = pd.concat([df, temp])
        df.to_csv(regression_path, index=False)
    else:
        df = pd.read_csv(regression_path)
    df['model'] = df['model'].astype('category').cat.reorder_categories(
        CFG.RSM_models['matrices'].keys())

    # plots
    for region_set, regions in CFG.region_sets.items():
        for level, subject_sample in zip(['ind', 'group'],
                                         [subjects_final, ['group']]):

            outpath_all = (f'{analysis_dir}/regression/{region_set}/model_fits'
                        f'_{level}.png')
            outpath_sel = (f'{analysis_dir}/regression/{region_set}/model_fits'
                        f'_{level}_select.png')
            if not op.isfile(outpath_sel) or overwrite_plots:
                print(f'Plotting {level} regression results in {region_set}...')
                plot_df = df[(df.level == level) &
                             (df.region.isin(regions)) &
                             (df.subject.isin(subject_sample))]
                if level == 'ind':
                    plot_df = plot_df.drop(
                        columns=['subject', 'mse','level']).groupby(
                        ['model', 'region']).agg(['mean', 'sem']).dropna(
                        ).reset_index()
                    plot_df.columns = ['level', 'region', 'value', 'error']
                else:
                    plot_df = plot_df.drop(columns=['level']).rename(columns={
                        'model': 'level', 'beta': 'value', 'mse': 'error'}
                        ).copy( deep=True)

                reg_ord = [r for r in regions if r in plot_df.region.unique()]
                plot_df.region = pd.Categorical(
                    plot_df.region, categories=reg_ord, ordered=True)
                plot_df.level = pd.Categorical(
                    plot_df.level, categories=list(CFG.RSM_models[
                        'matrices'].keys()), ordered=True)
                plot_df.sort_values(['region', 'level'], inplace=True)
                x_tick_labels = [CFG.regions[r] for r in reg_ord]

                # all models
                clustered_barplot(
                    plot_df, outpath_all, CFG.RSM_models,
                    x_tick_labels=x_tick_labels,
                    figsize=(fw_base + len(regions)*fw_mult, fh))

                # selected models
                dfs = plot_df[plot_df['level'].isin(CFG.RSM_models['final_set'])]
                clustered_barplot(
                    dfs, outpath_sel, CFG.RSM_models, x_tick_labels=x_tick_labels,
                    figsize=(fw_base + len(regions)*fw_mult, fh))

    # stats
    for region_set, regions in CFG.region_sets.items():
        outpath = (f'{analysis_dir}/regression/{region_set}/'
                   f'model_fits_ind_anova.csv')
        if not op.isfile(outpath) or overwrite_stats:
            print(f'Performing stats on regression results in {region_set}...')
            stats_df = df[(df.level == 'ind') &
                          (df.region.isin(regions)) &
                          (df.subject.isin(subjects_final))].copy(
                deep=True).drop(columns=['level','mse'])
            anova = pg.rm_anova(
                dv='beta', within=['model', 'region'], subject='subject',
                data=stats_df, detailed=True)
            anova.to_csv(outpath)
            post_hocs = pg.pairwise_tests(
                dv='beta', within=['region', 'model'], subject='subject',
                data=stats_df, padjust='holm', return_desc=True,
                effsize='cohen')
            post_hocs.to_csv(outpath.replace('anova', 'posthocs'))


    # occluder versus exemplar regression indices
    contrasts = ['exemplar_v_occluder_presence',
                 'exemplar_v_occluder_position']
    titles = ['object identity versus occluder presence',
              'object identity versus occluder position']

    # data
    reg_indices_path = f'{analysis_dir}/regression/regression_indices.csv'
    if (not op.isfile(reg_indices_path) or overwrite_data):
        print(f'Collating regression indices...')
        df = pd.DataFrame()
        for level, subject_sample in zip(['ind', 'group'],
                                            [subjects_final, ['group']]):
            for region, subject, contrast in itertools.product(
                        CFG.regions, subject_sample, contrasts):
                temp = RSAs[region][subject]
                if temp is not None:
                    value = getattr(temp, contrast)
                    df = pd.concat([df, pd.DataFrame({
                        'level': [level],
                        'region': [region],
                        'subject': [subject],
                        'contrast': [contrast],
                        'value': value})])
        df.to_csv(reg_indices_path, index=False)

    # plots
    df = pd.read_csv(reg_indices_path)
    for (level, subject_sample), (contrast, title), (region_set, regions) in (
            itertools.product(
                zip(['ind', 'group'], [subjects_final, ['group']]),
                zip(contrasts, titles), CFG.region_sets.items())):
        outpath = (f'{analysis_dir}/regression/{region_set}/'
                   f'{contrast}_{level}.png')
        if not op.isfile(outpath) or overwrite_plots:
            print(f'Plotting {level} regression indices for {title} in '
                  f'{region_set}...')

            plot_df = df[(df.level == level) &
                         (df.contrast == contrast) &
                         (df.region.isin(regions)) &
                         (df.subject.isin(subject_sample))].copy(
                deep=True).drop(columns=['level', 'contrast','subject'])

            if level == 'ind':
                plot_df = (plot_df.groupby(['region']).agg(
                        ['mean', 'sem']).dropna().reset_index())
                plot_df.columns = ['region','value', 'error']
            else:
                plot_df = plot_df.rename(columns={'model': 'level'})
                plot_df['error'] = np.nan

            reg_ord = [r for r in regions if r in plot_df.region.unique()]
            plot_df.region = pd.Categorical(
                plot_df.region, categories=reg_ord, ordered=True)
            plot_df.sort_values('region', inplace=True)
            x_tick_labels = [CFG.regions[r] for r in reg_ord]
            line_plot(plot_df, outpath, ylabel='object bias',
                ceiling=[1], floor=[-1], ylims=(-1.3, 1.3), hline=0,
                yticks=(-1,0,1), x_tick_labels=x_tick_labels,
                figsize=(fw_base + len(regions)*fw_mult, fh), title=title)

    # stats
    for (contrast, title), (region_set, regions) in itertools.product(
            zip(contrasts, titles), CFG.region_sets.items()):
        outpath = (f'{analysis_dir}/regression/{region_set}/'
                       f'{contrast}_ind_anova.csv')
        if not op.isfile(outpath) or overwrite_stats:
            print(f'Performing stats on regression indices for {title} in '
                  f'{region_set}...')
            try:
                stats_df = df[(df.level == 'ind') &
                              (df.contrast == contrast) &
                              (df.region.isin(regions)) &
                              (df.subject.isin(subjects_final))].copy(
                    deep=True).drop(columns=['level', 'contrast'])
                anova = pg.rm_anova(
                    dv='value', within='region', subject='subject',
                    data=stats_df, detailed=True)
                anova.to_csv(outpath)
                post_hocs = pg.pairwise_tests(
                    dv='value', within='region', subject='subject',
                    data=stats_df, padjust='holm', return_desc=True,
                    effsize='cohen')
                post_hocs.to_csv(outpath.replace('anova', 'posthocs'))
            except:
                pass

    # calculate noise ceiling
    new_results = False
    for region in df.region.unique():

        group_RSA = RSAs[region]['group']

        # skip if group RSA does not exist
        if not hasattr(group_RSA, 'noise_ceiling') or overwrite_data:

            print(f'Calculating noise ceiling for {region}...')
            new_results = True
            subject_RSMs = pd.DataFrame()
            for subject in subjects_final:
                temp = RSAs[region][subject]
                if temp is not None:
                    temp = temp.RSM_table
                    temp['subject'] = subject
                    subject_RSMs = pd.concat([subject_RSMs, temp])

            if len(subject_RSMs.subject.unique()) < 3:
                setattr(group_RSA, 'noise_ceiling', None)
                continue

            noise_ceiling = {}
            for bound in ['lower', 'upper']:
                vals = []
                for subject in subject_RSMs.subject.unique():

                    # get individual RSM, remove diagonal, and flatten
                    RSM_ind = RSAs[region][subject].RSM_table.similarity.values
                    RSM_ind_flat_offdiag = RSM_ind.flatten()[CFG.off_diag_mask_flat]

                    # lower bound requires mean RSM across remaining subjects
                    if bound == 'lower':
                        RSM_grp = subject_RSMs[
                            subject_RSMs.subject != subject].drop(
                            columns='subject').groupby(
                            subject_RSMs.columns[:6].tolist()).agg(
                            'mean', numeric_only=True).dropna().reset_index()

                    # upper bound requires mean RSM across all subjects
                    else:
                        RSM_grp = RSAs[region]['group'].RSM_table

                    # flatten and remove diagonal
                    RSM_grp = RSM_grp.similarity.values
                    RSM_grp_flat_offdiag = RSM_grp.flatten()[CFG.off_diag_mask_flat]

                    # calculate noise ceiling
                    nc = np.corrcoef(
                        RSM_ind_flat_offdiag, RSM_grp_flat_offdiag)[0, 1]
                    vals.append(nc)

                noise_ceiling[bound] = np.mean(vals)
            noise_ceiling['N'] = len(vals)

            # store noise ceiling values in group mean RSA_dataset
            setattr(group_RSA, 'noise_ceiling', noise_ceiling)

    if new_results:
        pkl.dump(RSAs, open(f'{analysis_dir}/RSA.pkl', 'wb'))


def stats_exp2(space, norm, norm_method, similarity, overwrite):


    for analysis, params in CFG.occlusion_robustness_analyses.items():

        out_dir = f'derivatives/RSA/ANOVAs/{analysis}'

        # occlusion robustness indices (object completion and occlusion invariance)

        # data
        df = pd.DataFrame()
        for attn, attn_path in zip(CFG.attns, CFG.attns_path):
            indices_path = (f'derivatives/RSA/occlusion{attn_path}_space'
                            f'-{space}/norm-{norm}_{norm_method}/{similarity}/'
                            f'{analysis}/indices.csv')
            indices = pd.read_csv(indices_path)
            indices['attn'] = attn
            df = pd.concat([df, indices])


        # stats
        for subtype, (region_set, regions) in itertools.product(
                df.subtype.unique(), CFG.region_sets.items()):
            outpath = (f'{out_dir}/{region_set}/{params["index_label"]}'
                       f'_{subtype}_ind_anova.csv')
            os.makedirs(op.dirname(outpath), exist_ok=True)
            if not op.isfile(outpath) or True:#overwrite:
                print(f'Performing stats on {analysis} indices ({subtype}) in'
                      f' {region_set}...')
                stats_df = df[
                    (df.level == 'ind') &
                    (df.subtype == subtype) &
                    (df.region.isin(regions)) &
                    (df.subject.isin(CFG.subjects_final['exp2']))].copy(deep=True).drop(
                    columns=['level', 'subtype', 'index'])
                try:

                    # two-way AVOVA, effect of attn, region and post-hocs
                    anova = pg.rm_anova(
                        dv='value', within=['attn', 'region'],
                        subject='subject',
                        data=stats_df, detailed=True)
                    anova.to_csv(outpath)
                    post_hocs = pg.pairwise_tests(
                        dv='value', within=['attn', 'region'],
                                 subject='subject',
                        data=stats_df, padjust='holm',
                        return_desc=True, effsize='cohen')
                    post_hocs.to_csv(outpath.replace('anova', 'posthocs'))

                    # t-tests against 0 and 1
                    one_sided_floor = stats_df.groupby(
                        'region').apply(
                        lambda df: pg.ttest(df.value,
                                            0)).reset_index()
                    one_sided_floor['mu'] = 0
                    one_sided_floor['p-corr'] = pg.multicomp(
                        one_sided_floor['p-val'], method='holm')[1]
                    one_sided_ceiling = stats_df.groupby(
                        'region').apply(
                        lambda df: pg.ttest(df.value,
                                            1)).reset_index()
                    one_sided_ceiling['mu'] = 1
                    one_sided_ceiling['p-corr'] = pg.multicomp(
                        one_sided_ceiling['p-val'], method='holm')[
                        1]
                    one_sided = pd.concat(
                        [one_sided_floor, one_sided_ceiling])
                    one_sided.to_csv(
                        outpath.replace('anova', 'one_sided'))


                except:
                    pass


        # condition comparisons (object completion and occlusion invariance)

        # data
        df = pd.DataFrame()
        for attn, attn_path in zip(CFG.attns, CFG.attns_path):
            sims_path = (
                f'derivatives/RSA/occlusion{attn_path}_space'
                f'-{space}/norm-{norm}_{norm_method}/{similarity}/'
                f'{analysis}/cond-wise_sims.csv')
            sims = pd.read_csv(sims_path)
            sims['attn'] = attn
            df = pd.concat([df, sims])

        # stats
        for region_set, regions in CFG.region_sets.items():
            outpath = (f'{out_dir}/{region_set}/cond_sims_prop_anova.csv')
            os.makedirs(op.dirname(outpath), exist_ok=True)
            if not op.isfile(outpath) or overwrite:
                print(f'Performing stats on {analysis} in {region_set}...')
                stats_df = (df[
                    (df.level.isin(['EsOs', 'EsOd', 'EsUb', 'EsU1'])) &
                    (df.region.isin(regions)) &
                    (df.subject.isin(CFG.subjects_final['exp2']))]
                    .copy(deep=True))
                stats_df['similarity_z'] = np.arctanh(stats_df['similarity'])
                stats_df = stats_df.drop(columns=['similarity'])
                try:

                    # three-way AVOVA, effect of attn, level, region
                    anova = (AnovaRM(stats_df, within=['attn', 'region', 'level'],
                                    depvar='similarity_z', subject='subject')
                             .fit().anova_table)
                    anova.to_csv(outpath)

                    # two-way AVOVA, effect of level, region
                    anova = pg.rm_anova(
                        dv='similarity_z', within=['region', 'level'],
                        subject='subject', data=stats_df, detailed=True)
                    anova.to_csv(outpath.replace('anova', 'anova_noattn'))

                    # post hocs
                    stats_df_mean = stats_df.groupby(['subject', 'region',
                                                      'level']).agg(
                        {'similarity_z': 'mean'}).reset_index()
                    post_hocs = pg.pairwise_tests(
                        dv='similarity_z', within=['level', 'region'],
                        subject='subject', data=stats_df_mean,
                        padjust='holm', return_desc=True, effsize='cohen')
                    post_hocs.to_csv(outpath.replace(
                        'anova', 'posthocs_same_object'), index=False)

                    stats_df_diff = stats_df_mean.groupby(
                        ['subject', 'region']).apply(lambda df: pd.Series(
                        df.similarity_z[df.level == params['conds'][
                            0]].values[0] -
                        df.similarity_z[df.level == params['conds'][
                            1]].values[0],
                        index=['diff'])).reset_index()
                    post_hocs_interaction = pg.pairwise_tests(
                        dv='diff', within='region', subject='subject',
                        data=stats_df_diff, padjust='holm')
                    post_hocs_interaction.to_csv(outpath.replace(
                        'anova', 'posthocs_interaction'),
                        index=False)

                    stats_df_prop = stats_df_mean.groupby(
                        ['subject', 'region']).apply(lambda df: pd.Series(
                        df.similarity_z[df.level == params['conds'][
                            1]].values[0] /
                        stats_df.similarity_z[stats_df.level == params['conds'][
                            0]].mean(),
                        index=['prop'])).reset_index()
                    post_hocs_interaction = pg.pairwise_tests(
                        dv='prop', within='region', subject='subject',
                        data=stats_df_prop, padjust='holm')
                    post_hocs_interaction.to_csv(outpath.replace(
                        'anova', 'posthocs_interaction_prop'),
                        index=False)

                    # t-tests of key contrast against 0
                    stats_df_key_contrast = stats_df_mean[
                        stats_df.level == params['conds'][1]].copy(deep=True)
                    one_sided = stats_df_key_contrast.groupby('region').apply(
                        lambda df: pg.ttest(df.similarity_z, 0)).reset_index()
                    one_sided['mu'] = 0
                    one_sided['p-corr'] = pg.multicomp(
                        one_sided['p-val'], method='holm')[1]
                    one_sided.to_csv(outpath.replace(
                        'anova', 'one_sample'), index=False)

                except:
                    pass


def RSA_ROI(responses_dir, norm, norm_method,
            similarity, overwrite_analyses, overwrite_plots, overwrite_stats):

    exp = 'exp1' if 'exp1' in os.getcwd() else 'exp2'
    subjects = CFG.subjects[exp]
    subjects_final = CFG.subjects_final[exp]
    responses = pkl.load(open(f'{responses_dir}/responses.pkl', 'rb'))
    task = op.basename(responses_dir).split('_')[0]
    similarity_label = CFG.similarities[similarity]

    norm_dir = f'norm-{norm}'
    if norm != 'none':
        norm_dir += f'_{norm_method}'
    analysis_dir = f'{responses_dir}/{norm_dir}/{similarity}'
    os.makedirs(analysis_dir, exist_ok=True)

    RSA_path = f'{analysis_dir}/RSA.pkl'
    if op.isfile(RSA_path) and not 'RSA_ROI' in overwrite_analyses:
        RSAs = pkl.load(open(RSA_path, 'rb'))
    else:
        RSAs = {}

    new_results = False  # flag to save RSA data

    for region in CFG.regions:

        if region not in RSAs:
            RSAs[region] = {}

        # do group last as it requires all subjects
        for subject in subjects + ['group']:

            new_result = False  # flag to remake plots
            subject_RSA = None

            if subject not in RSAs[region]:

                print(f'Performing ROI-wise RSA | {region} | {subject} | '
                      f'norm-{norm} | {norm_method} | {similarity}')

                new_result = new_results = True

                if subject != 'group':

                    subject_responses = responses[region][subject]
                    if subject_responses is not None:

                        # get responses
                        print('Retrieving responses...')
                        subject_data = RSA_dataset(
                            responses=subject_responses)

                        # calculate RSM
                        print('Calculating RSM...')
                        subject_RSA = subject_data.calculate_RSM(
                            norm=norm,
                            norm_method=norm_method,
                            similarity=similarity)
                        subject_RSA.analyse()

                else:

                    ind_RSMs = [
                        RSAs[region][s].RSM for s in \
                        subjects_final if RSAs[region][s] \
                        is not None]

                    # calculate RSM
                    print(f'Calculating RSM...')
                    if len(ind_RSMs):
                        subject_RSM = np.mean(np.array(ind_RSMs),
                                              axis=0)
                        subject_RSA = RSA(RSM=subject_RSM,
                                        similarity=similarity)
                        subject_RSA.analyse()

                RSAs[region][subject] = subject_RSA

            # RSM plots
            file_name = f'{subject}_{region}.png'
            outpath = f'{analysis_dir}/RSMs/{file_name}'
            make_plot = subject_RSA is not None and (not op.isfile(
                outpath) or 'RSM' in overwrite_plots or new_result)
            if make_plot:
                print(f'Plotting RSM...')
                col_lim = .3 if subject == 'group' else .5
                plot_title = f'{task}, subject: {subject}, ROI: {region}'
                subject_RSA.plot_RSM(vmin=-col_lim, vmax=col_lim,
                                     fancy=subject == 'group',
                                     title=plot_title,
                                     labels=CFG.cond_labels['exp1'],
                                     outpath=outpath,
                                     measure=similarity_label)

            # MDS plots
            outpath = f'{analysis_dir}/MDS/{file_name}'
            make_plot = subject_RSA is not None and (not op.isfile(
                outpath) or 'MDS' in overwrite_plots or new_result)
            if make_plot:
                print(f'Plotting MDS...')
                subject_RSA.plot_MDS(outpath=outpath)

    # save RSA data
    if new_results:
        pkl.dump(RSAs, open(RSA_path, 'wb'))

    # compare regions of interest
    ow_data = new_results or 'compare_regions' in overwrite_analyses
    ow_plots = True if ow_data else 'compare_regions' in overwrite_plots
    compare_regions(RSAs, analysis_dir, ow_data, ow_plots, overwrite_stats)


def RSA_searchlight(task, space, norm, norm_method,
            similarity, num_procs, index_level, subtype, overwrite):


    exp = 'exp1' if 'exp1' in os.getcwd() else 'exp2'
    subjects = CFG.subjects_final[exp]
    norm_dir = f'norm-{norm}'
    if norm != 'none':
        norm_dir += f'_{norm_method}'
    analysis_dir = (f'derivatives/RSA_searchlight/task-{task}_space-{space}/'
               f'{norm_dir}/{similarity}/{subtype}')

    # completion calculated at individual level
    if index_level == 'ind':
        for s, subject in enumerate(subjects):

            out_dir = f'{analysis_dir}/sub-{subject}'
            out_path = f'{out_dir}/completion.nii.gz'

            if not op.isfile(out_path) or overwrite:

                print(f'Performing searchlight RSA | derivatives | {task} | '
                      f'{space} space | {subject} | norm-{norm} | '
                      f'{norm_method} | {similarity}')

                os.makedirs(out_dir, exist_ok=True)

                # load brain mask
                if space == 'func':
                    brain_mask_path = (f'derivatives/ROIs/sub-{subject}/'
                                       f'func_space/brain_mask.nii.gz')
                else:
                    brain_mask_path = (f'derivatives/ROIs/MNI152_2mm/'
                                       f'brain_mask.nii.gz')
                brain_mask = nib.load(brain_mask_path)
                header, affine = brain_mask.header, brain_mask.affine
                brain_mask_3D = brain_mask.get_fdata()
                vol_dims = brain_mask_3D.shape
                brain_mask_2D = brain_mask_3D.flatten()
                n_vox_vol = np.prod(vol_dims)

                # find centers and neighbors within flattened tensor
                radius = 3
                centers, neighbors = get_volume_searchlight(
                    brain_mask_3D, radius=radius, threshold=.4)
                pkl.dump({'centers': centers, 'neighbors': neighbors},
                            open(f'{out_dir}/sl_params.pkl', 'wb'))

                # save 3D nifti of centers
                centers_2D = np.zeros_like(brain_mask_2D)
                centers_2D[centers] = 1
                centers_3D = centers_2D.reshape(vol_dims)
                nib.Nifti1Image(centers_3D, header=header, affine=affine,
                                dtype=np.float32).to_filename(
                    f'{out_dir}/centers.nii.gz')

                # collate responses
                print('Collating responses...')
                subj_dir = (f'derivatives/FEAT/sub-{subject}/'
                            f'subject-wise_space-{space}')
                n_splits = len(
                    glob.glob(f'{subj_dir}/task-{task}_split*A.gfeat'))
                responses = np.empty((n_splits, 2, CFG.n_img, n_vox_vol))
                for sp in range(n_splits):
                    for si, side in enumerate(['A', 'B']):
                        cope_paths = [f'{subj_dir}/task-{task}_split-{sp}' \
                                      f'{side}.gfeat/cope{c + 1}.feat/stats/' \
                                      f'cope1.nii.gz' for c in range(CFG.n_img)]
                        for c, cope_path in enumerate(cope_paths):
                            responses[sp, si, c, :] = \
                                nib.load(cope_path).get_fdata().flatten()

                # analysis function
                def get_robustness(responses, norm, norm_method, similarity):
                    sl_data = RSA_dataset(responses=responses)
                    sl_RSM = sl_data.calculate_RSM(
                        norm=norm,
                        norm_method=norm_method,
                        similarity=similarity)
                    sl_RSM.RSM_to_table()
                    sl_RSM.calculate_occlusion_robustness()
                    index_df = sl_RSM.occlusion_robustness
                    score = index_df[(index_df.analysis == 'object_completion') &
                                     (index_df.subtype == subtype)].value.item()
                    return score

                # run searchlight
                print('Performing searchlight...')
                with np.errstate(divide='ignore', invalid='ignore'):
                    robustness = Parallel(n_jobs=num_procs)(
                        delayed(get_robustness)(
                            responses[:,:,:,nb], norm, norm_method, similarity
                        ) for nb in tqdm(neighbors))

                # reshape into 3D volume and save out nifti file
                print('Writing statistical map to NIFTI...')
                robustness_2D = -np.ones_like(brain_mask_2D)
                for c, center in enumerate(centers):
                    if np.isfinite(robustness[c]):
                        robustness_2D[center] = robustness[c]
                robustness_3D = robustness_2D.reshape(vol_dims)
                nib.Nifti1Image(robustness_3D, header=header, affine=affine,
                                dtype=np.float32).to_filename(out_path)

    # completion calculated at group level (standard space only)
    elif space == 'standard':

        out_dir = f'{analysis_dir}/sub-fsaverage'
        out_path = f'{out_dir}/completion_group.nii.gz'

        if not op.isfile(out_path) or overwrite:

            print(f'Performing searchlight RSA | derivatives | {task} | '
                  f'{space} space | fs-average (group) | norm-{norm} | '
                  f'{norm_method} | {similarity}')

            os.makedirs(out_dir, exist_ok=True)

            # get voxels for which most subjects (>= 5) have data
            brain_mask_path = f'{out_dir}/FOV_mask.nii.gz'
            if not op.isfile(brain_mask_path):
                ind_masks = [(
                    f'derivatives/FEAT/sub-{subj}/subject-wise_space-standard/'
                    f'/task-{task}_all-runs.gfeat/cope1.feat/mask.nii.gz')
                    for subj in CFG.subjects_final[exp]]
                cmd = f'fslmaths ' + ' -add '.join(ind_masks)
                cmd += f' -thr 5 -bin {brain_mask_path}'
                os.system(cmd)
            brain_mask = nib.load(brain_mask_path)
            header, affine = brain_mask.header, brain_mask.affine
            brain_mask_3D = brain_mask.get_fdata()
            vol_dims = brain_mask_3D.shape
            brain_mask_2D = brain_mask_3D.flatten()
            n_vox_vol = np.prod(vol_dims)

            # find centers and neighbors within flattened tensor
            radius = 3
            centers, neighbors = get_volume_searchlight(
                brain_mask_3D, radius=radius, threshold=.4)
            pkl.dump({'centers': centers, 'neighbors': neighbors},
                     open(f'{out_dir}/sl_params.pkl', 'wb'))

            # save 3D nifti of centers
            centers_2D = np.zeros_like(brain_mask_2D)
            centers_2D[centers] = centers
            centers_3D = centers_2D.reshape(vol_dims)
            nib.Nifti1Image(centers_3D, header=header, affine=affine,
                            dtype=np.float32).to_filename(
                f'{out_dir}/searchlight_centers.nii.gz')

            # collate responses
            print('Collating responses...')
            n_splits = len(
                glob.glob(f'derivatives/FEAT/sub-F016/subject-wise_space-{space}/'
                          f'task-{task}_split*A.gfeat'))
            responses = np.empty(
                (len(subjects), n_splits, 2, CFG.n_img, n_vox_vol))
            combos = itertools.product(enumerate(subjects), range(n_splits),
                                       enumerate(['A', 'B']))
            for (s, subject), sp, (si, side) in combos:
                if sp + si == 0:
                    print(f'subject {s + 1}/{len(subjects)}')
                subj_dir = (f'derivatives/FEAT/sub-{subject}/'
                            f'subject-wise_space-{space}')
                cope_paths = [f'{subj_dir}/task-{task}_split-{sp}' \
                              f'{side}.gfeat/cope{c + 1}.feat/stats/' \
                              f'cope1.nii.gz' for c in range(CFG.n_img)]
                for c, cope_path in enumerate(cope_paths):
                    responses[s, sp, si, c, :] = \
                        nib.load(cope_path).get_fdata().flatten()

            # analysis function
            def get_robustness(responses, norm, norm_method, similarity):
                RSMs = np.empty((len(subjects), CFG.n_img, CFG.n_img))
                for s in range(responses.shape[0]):
                    sl_data = RSA_dataset(responses=responses[s])
                    sl_RSM = sl_data.calculate_RSM(
                        norm=norm,
                        norm_method=norm_method,
                        similarity=similarity)
                    RSMs[s] = sl_RSM.RSM
                RSM_group = RSA(
                    RSM=np.nanmean(RSMs, axis=0), similarity=similarity)
                RSM_group.RSM_to_table()
                RSM_group.calculate_occlusion_robustness()
                index_df = RSM_group.occlusion_robustness
                score = index_df[
                    (index_df.analysis == 'object_completion') &
                    (index_df.subtype == subtype)].value.item()
                del RSMs, RSM_group, sl_data, sl_RSM
                gc.collect()
                return score

            # run searchlight
            print('Performing searchlight...')
            batch_size = 2048
            robustness = []
            for b, batch in enumerate(range(0, len(neighbors), batch_size)):
                print(f'batch {b + 1}/{np.ceil(len(neighbors)/batch_size)}')
                with np.errstate(divide='ignore', invalid='ignore'):
                    robustness += Parallel(n_jobs=num_procs)(
                        delayed(get_robustness)(
                            responses[:, :, :, :, nb], norm, norm_method, similarity
                        ) for nb in tqdm(neighbors[batch:batch + batch_size]))

            # reshape into 3D volume and save out nifti file
            print('Writing statistical map to NIFTI...')
            robustness_2D = -np.ones_like(brain_mask_2D)
            for c, center in enumerate(centers):
                if np.isfinite(robustness[c]):
                    robustness_2D[center] = robustness[c]
            robustness_3D = robustness_2D.reshape(vol_dims)
            nib.Nifti1Image(robustness_3D, header=header, affine=affine,
                            dtype=np.float32).to_filename(out_path)


def do_RSA(exp, overwrite_analyses, overwrite_plots, overwrite_stats,
           num_procs):

    print('Running RSA...')

    tasks = [task for task in CFG.scan_params[exp] if 'occlusion' in task]
    subjects = CFG.subjects[exp]

    # plot model and contrast matrices
    plot_RSM_models(overwrite_plots)

    for space, task in itertools.product(CFG.spaces, tasks):

        out_dir = f'derivatives/RSA/{task}_space-{space}'
        os.makedirs(out_dir, exist_ok=True)


        # get ROI responses

        # set up / get responses data structure
        responses_path = f'{out_dir}/responses.pkl'
        if op.isfile(responses_path) and not 'responses' in \
                overwrite_analyses:
            responses = pkl.load(open(responses_path, 'rb'))
        else:
            responses = {}

        # get voxelwise responses
        save_responses = False
        for region, subject in itertools.product(CFG.regions, subjects):

            if region not in responses:
                responses[region] = {}

            # get responses
            new_responses = False
            if subject not in responses[region]:
                print(f'Getting responses | {region} | {subject}')
                responses_subject = get_responses(
                    subject, region, task, space)
                responses[region][subject] = responses_subject
                save_responses = new_responses = True

            # make TSNE plots
            outpath = f'{out_dir}/TSNE/{subject}_{region}.png'
            plot_TSNE = (responses[region][subject] is not None
                         and (not op.isfile(outpath) or 'TSNE' in
                         overwrite_plots or new_responses))
            if plot_TSNE:
                print(f'Generating TSNE plots | {region} | {subject}')
                RSA_dataset(np.mean(responses[region][subject], axis=0)).\
                    plot_TSNE(outpath)  # use mean across runs

        # save responses
        if save_responses:
            pkl.dump(responses, open(responses_path, 'wb'))

        # perform RSA using various methods
        for norm, norm_method, similarity in itertools.product(
                CFG.norms, CFG.norm_methods, CFG.similarities):

            # run RSA
            RSA_ROI(out_dir, norm, norm_method, similarity, overwrite_analyses,
                    overwrite_plots, overwrite_stats)

            # run searchlight RSA
            overwrite = 'RSA_searchlight' in overwrite_analyses
            #RSA_searchlight(task, space, norm, norm_method,
            #        similarity, num_procs, 'group', 'prop', overwrite)

    # stats involving effect of attention, exp2 only
    if exp == 'exp2':
        for space in ['standard']:#CFG.spaces:
            for norm, norm_method, similarity in itertools.product(
                    CFG.norms, CFG.norm_methods, CFG.similarities):
                stats_exp2(space, norm, norm_method, similarity,
                           overwrite_stats)

    # make summary of ROI sizes
    ROI_summary = pd.DataFrame()
    for region, subject in itertools.product(CFG.regions, subjects):
        if responses[region][subject] is not None:
            n_voxels = responses[region][subject].shape[-1]
        else:
            n_voxels = 0
        ROI_summary = pd.concat([ROI_summary, pd.DataFrame({
            'region': [region],
            'subject': [subject],
            'n_voxels': [n_voxels],
        })])
    ROI_summary.to_csv(f'derivatives/ROIs/ROI_summary.csv',
                       index=False)


if __name__ == "__main__":

    # overwrite_analyses = ['RSA_ROI', 'RSA_searchlight']
    overwrite_analyses = ['responses', 'RSA_ROI']  # to overwrite all analyses
    # overwrite_plots = ['TSNE', 'RSM', 'MDS', 'RSM_models', 'contrasts']
    overwrite_plots = []  # to overwrite specific plots
    overwrite_stats = False
    for exp in ['exp1']:
        os.chdir(f'{PROJ_DIR}/in_vivo/fMRI/{exp}')
        do_RSA(exp, overwrite_analyses, overwrite_plots)