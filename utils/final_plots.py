# /usr/bin/python
# Created by David Coggan on 2023 06 23

"""
script for mkaing final plots for manuscript
"""

import os
import os.path as op
import itertools
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import sys
import seaborn as sns

from .plot_utils import export_legend, custom_defaults, make_legend
plt.rcParams.update(custom_defaults)

from .config import CFG, PROJ_DIR
from .RSA import RSA

figdir = f'{PROJ_DIR}/figures'
region_set = 'EVC_IT'
regions = CFG.region_sets[region_set]

def main():

    make_ROI_plots(overwrite=True)
    make_RSM_model_plots(overwrite=False)
    for exp in ['exp1','exp2']:
        for task in CFG.scan_params[exp]:
            make_RSM_plots(exp, task)
            make_MDS_plots(exp, task)
    make_condwise_plots_exp1()
    make_condwise_plots_exp2()
    make_index_plots_exp1()
    make_index_plots_exp2()


def RGB_tuple_to_string(colour):
    return f'{colour[0]},{colour[1]},{colour[2]}'

def make_ROI_plots(overwrite=False):

    hemis = ['lh', 'rh']
    views = ['posterior', 'inferior', 'lateral', 'medial']
    fs_dir = f'{os.environ["SUBJECTS_DIR"]}/fsaverage'
    out_dir = f'{figdir}/ROIs'
    os.makedirs(out_dir, exist_ok=True)

    for ROI_set, hemi, view in itertools.product(
            ['EVC_IT'], hemis, views):

        regions = CFG.region_sets[ROI_set]
        cmap = matplotlib.colormaps['plasma'].colors
        colours = [cmap[c] for c in np.linspace(0,255,len(regions), dtype=int)]

        image_path = (f'{out_dir}/{ROI_set}_{view}_{hemi}.png')
        if not op.isfile(image_path) or overwrite:

            cmd_plot = (f'freeview -f '
                        f'{fs_dir}/surf/{hemi}.inflated'
                        f':curvature_method=binary')

            for region, colour in zip(regions, colours):

                # get existing surface labels
                if not 'ventral' in region:
                    if region in ['V1','V2','V3']:
                        subregions = [f'{region}d', f'{region}v']
                    else:
                        subregions = [region]
                    labels = [f'{fs_dir}/surf/{hemi}.wang15_mplbl.{reg}.label' \
                              for reg in subregions]

                # make remaining surface labels
                else:
                    volume = (f'{PROJ_DIR}/exp1/derivatives/'
                              f'ROIs/MNI152_2mm/{region}.nii.gz')
                    surface = f'{volume[:-7]}_{hemi}.mgh'
                    if not op.isfile(surface):
                        cmd = (f'mri_vol2surf '
                               f'--mov {volume} '
                               f'--out {surface} '
                               f'--regheader fsaverage '
                               f'--hemi {hemi}')
                        os.system(cmd)
                    label = f'{volume[:-7]}_{hemi}.label'
                    if not op.isfile(label):
                        cmd = (f'mri_cor2label '
                               f'--i {surface} '
                               f'--surf fsaverage {hemi} '
                               f'--id 1 '
                               f'--l {label}')
                        os.system(cmd)
                    labels = [label]


                col_str = ','.join([str(int(255 * c)) for c in colour])
                for label in labels:
                    cmd_plot += f':label={label}:label_color={col_str}'

            cmd_plot += (f' -layout 1 -viewport 3d -view {view} '
                         f'-ss {image_path} 2 autotrim')
            os.system(cmd_plot)


def make_RSM_model_plots(overwrite=False):

    out_dir = f'{figdir}/RSMs'
    os.makedirs(out_dir, exist_ok=True)
    for model_label, model in CFG.RSM_models['matrices'].items():
        outpath = f'{out_dir}/model_{model_label}.pdf'
        if not op.isfile(outpath) or overwrite:
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
        outpath = f'{out_dir}/contrasts_{label}.pdf'
        if not op.isfile(outpath) or overwrite:
            RSA(RSM=contrast_mat).plot_RSM(vmin=0, vmax=8,
                                           title=f'contrasts',
                                           cmap=cmap, outpath=outpath,
                                           fancy=True)


def make_RSM_plots(exp, task, overwrite=False):

    out_dir = f'{figdir}/RSMs'
    os.makedirs(out_dir, exist_ok=True)
    vmax = .3 if exp == 'exp1' else .2
    vmin = -vmax
    for region in ['V1', 'V2', 'V3', 'hV4', 'ventral_stream_sub_V1-V4']:
        outpath = f'{out_dir}/{exp}_{task}_{CFG.regions[region]}.pdf'
        if not op.isfile(outpath) or overwrite:
            with open(f'{PROJ_DIR}/{exp}/derivatives/RSA/'
                     f'{task}/RSA.pkl', 'rb') as f:
                RSMs = pkl.load(f)

            print(f'Plotting RSM...')
            RSM = RSMs[region]['group'].RSM
            imx = f'{PROJ_DIR}/figures/RSM_pictures_x.png'
            imy = f'{PROJ_DIR}/figures/RSM_pictures_y.png'
            picx = plt.imread(imx)
            picy = plt.imread(imy)

            fig, unused_ax = plt.subplots(figsize=(7.5, 5.25))
            unused_ax.axis('off')
            ax = fig.add_axes([.1, .22, .75, .75])
            im = ax.imshow(RSM, vmin=vmin, vmax=vmax, cmap='rainbow')
            ax.tick_params(**{'length': 0})
            ax.set_xticks(np.arange(RSM.shape[0]))
            ax.set_yticks(np.arange(RSM.shape[1]))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(direction='in')
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            cbar = fig.colorbar(im, fraction=0.0453, pad=0.1, ax=ax)
            cbar.set_ticks([vmin, 0, vmax], fontsize=16, labels=[vmin, 0, vmax])
                           #labels=['-0.3', '0.0','0.3'], )
            ax.set_title('')
            plt.text(25, 15, "correlation ($\it{r}$)", rotation='vertical',
                     fontsize=16)
            newax = fig.add_axes([-.27, 0.2, .79, .79])
            newax.imshow(picy)
            newax.tick_params(**{'length': 0})
            newax.spines['bottom'].set_visible(False)
            newax.spines['left'].set_visible(False)
            newax.set_xticks([])
            newax.set_yticks([])
            hw_ratio = (.79*(5.25/7.5))
            newax2 = fig.add_axes([.2, -0.19, hw_ratio, hw_ratio+.02])
            newax2.imshow(picx)
            newax2.tick_params(**{'length': 0})
            newax2.spines['bottom'].set_visible(False)
            newax2.spines['left'].set_visible(False)
            newax2.set_xticks([])
            newax2.set_yticks([])
            plt.savefig(outpath)
            plt.close()


def make_MDS_plots(exp, task, overwrite=False):
    out_dir = f'{figdir}/MDS'
    os.makedirs(out_dir, exist_ok=True)
    for region in regions:
        outpath = f'{out_dir}/{exp}_{task}_{CFG.regions[region]}.pdf'
        if not op.isfile(outpath) or overwrite:
            print(f'Plotting MDS...')
            RSMs = pkl.load(
                open(f'{PROJ_DIR}/{exp}/derivatives/RSA/'
                     f'{task}/RSA.pkl', 'rb'))
            RSMs[region]['group'].plot_MDS(outpath=outpath)


def make_condwise_plots_exp1():

    out_dir = f'{figdir}/condwise_sims'
    os.makedirs(out_dir, exist_ok=True)
    figsize = (4.2, 2.5)
    exp = 'exp1'

    # conditions-wise similarities
    for analysis, params in CFG.occlusion_robustness_analyses.items():

        df_summary = pd.read_csv(
            f'{PROJ_DIR}/{exp}/derivatives/RSA/'
            f'occlusion/{analysis}/cond-wise_sims_summary.csv')
        os.makedirs(out_dir, exist_ok=True)

        # region-wise plot for each region_set
        outpath = f'{out_dir}/{exp}_occlusion_{analysis}.pdf'

        plot_df = df_summary[
            (df_summary.analysis == analysis) &
            (df_summary.region.isin(regions))].copy(deep=True)
        reg_ord = [r for r in regions if r in plot_df.region.unique()]
        plot_df.region = pd.Categorical(
            plot_df.region, categories=reg_ord, ordered=True)
        plot_df.level = pd.Categorical(
            plot_df.level, categories=params['conds'], ordered=True)
        plot_df.sort_values(['region', 'level'], inplace=True)
        ylabel = similarity_label

        yticks = np.arange(0,1,.1)
        ylims = (-.025, .25) if analysis == 'object_completion' else (-.04, .31)

        x_tick_labels = [CFG.regions[r] for r in reg_ord]

        # all conditions as bars with errors
        df_means = plot_df.pivot(index='region', columns='level', values='value')
        df_sems = plot_df.pivot(index='region', columns='level',
                           values='error').values
        fig, ax = plt.subplots(figsize=figsize)
        df_means.plot(
            kind='bar',
            ylabel=ylabel,
            yerr=df_sems.transpose(),
            rot=0,
            figsize=figsize,
            color=params['colours'],
            legend=False,
            width=.8,
            capsize=2.5,
            ax=ax)
        plt.yticks(yticks)
        plt.ylim(ylims)
        plt.xlabel(None)
        ax.set_xticks(np.arange(len(x_tick_labels)), x_tick_labels)
        plt.tick_params(axis='x', pad=7)
        ax.xaxis.set_ticks_position('none')  # no ticks
        ax.spines['bottom'].set_color('none')  # no x-axis
        plt.tight_layout()
        fig.savefig(outpath)
        plt.close()

    # legend
    outpath = f'{out_dir}/legend_{analysis}.png'
    labels = [i.replace(', ', '\n') for i in params['labels']]
    f = lambda m, c: \
        plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
    handles = [f('s', colour) for colour in params['colours']]
    legend = plt.legend(handles, labels, loc=3)
    export_legend(legend, filename=outpath)
    plt.close()


def make_condwise_plots_exp2():

    out_dir = f'{figdir}/condwise_sims'
    os.makedirs(out_dir, exist_ok=True)
    figsize = (5, 2.5)  # figure size
    exp = 'exp2'

    # conditions-wise similarities
    for analysis, params in CFG.occlusion_robustness_analyses.items():

        df_summary_attended = pd.read_csv(
            f'{PROJ_DIR}/{exp}/derivatives/RSA/'
            f'occlusionAttnOn/{analysis}/cond-wise_sims_summary.csv')
        df_summary_attended['task'] = 'attended'
        df_summary_unattended = pd.read_csv(
            f'{PROJ_DIR}/{exp}/derivatives/RSA/'
            f'occlusionAttnOff/{analysis}/cond-wise_sims_summary.csv')
        df_summary_unattended['task'] = 'unattended'
        df_summary = pd.concat([df_summary_attended, df_summary_unattended])
        os.makedirs(out_dir, exist_ok=True)

        # region-wise plot for each region_set
        outpath = f'{out_dir}/{exp}_{analysis}.pdf'

        plot_df = df_summary[
            (df_summary.analysis == analysis) &
            (df_summary.region.isin(regions))].copy(deep=True)
        reg_ord = [r for r in regions if r in plot_df.region.unique()]
        plot_df.region = pd.Categorical(
            plot_df.region, categories=reg_ord, ordered=True)
        plot_df.level = pd.Categorical(
            plot_df.level, categories=params['conds'], ordered=True)
        plot_df.sort_values(['task', 'region', 'level'], inplace=True)
        ylabel = similarity_label

        yticks = np.arange(0, 1, .05)
        ylims = (-.02, .125) if analysis == 'object_completion' else (-.04, .25)

        x_tick_labels = [CFG.regions[r] for r in reg_ord]

        # all conditions as bars with errors
        fig, axes = plt.subplots(ncols=2, sharey=True, figsize=figsize)
        for i, task in enumerate(['attended', 'unattended']):
            ax = axes[i]
            df_means = (plot_df[plot_df.task == task]
                .pivot(index='region', columns='level', values='value'))
            df_sems = (plot_df[plot_df.task == task]
                .pivot(index='region', columns='level', values='error').values)

            df_means.plot(
                kind='bar',
                yerr=df_sems.transpose(),
                rot=0,
                figsize=figsize,
                color=params['colours'],
                legend=False,
                width=.8,
                capsize=1.5,
                ax=ax,
                xlabel=task)
            plt.ylim(ylims)
            if i == 0:
                plt.yticks(yticks)
                ax.set_ylabel(ylabel)
                #plt.tick_params(axis='x', pad=7)
                # turn off minor ticks on y axis
                ax.tick_params(axis='y', which='minor', length=0)
            else:
                plt.yticks(None)
                plt.ylabel(None)
                ax.yaxis.set_ticks_position('none')  # no ticks
                ax.spines['left'].set_color('none')  # no y-axis
            ax.set_xticks(np.arange(len(x_tick_labels)), x_tick_labels)
            ax.xaxis.set_ticks_position('none')  # no ticks
            ax.spines['bottom'].set_color('none')  # no x-axis
        plt.tight_layout()
        fig.savefig(outpath)
        plt.close()

        # legend
        outpath = f'{out_dir}/legend_{analysis}.png'
        labels = [i.replace(', ', '\n') for i in params['labels']]
        f = lambda m, c: \
            plt.plot([], [], marker=m, markerfacecolor=c, color='white')[0]
        handles = [f('s', colour) for colour in params['colours']]
        legend = plt.legend(handles, labels, loc=3)
        export_legend(legend, filename=outpath)
        plt.close()


def make_index_plots_exp1():


    out_dir = f'{figdir}/indices'
    os.makedirs(out_dir, exist_ok=True)
    figsize = (3.2, 2.5)
    yticks = (0, .5, 1)
    ytick_labels = ['0.0', '0.5', '1.0']
    exp, task = 'exp1', 'occlusion'

    for analysis, params in CFG.occlusion_robustness_analyses.items():

        outpath = f'{out_dir}/{exp}_{task}_{analysis}.pdf'
        ylims = (0, 1)

        # data
        RSA_dir = (f'{PROJ_DIR}/{exp}/derivatives/RSA/'
            f'{task}/{analysis}')
        df = pd.read_csv(f'{RSA_dir}/indices.csv')
        plot_df = df[
            (df.level == 'ind') &
            (df.subtype == 'prop2') &
            (df.region.isin(regions))].copy(
            deep=True).reset_index(drop=True).drop(columns=[
            'level', 'subtype', 'subject'])
        reg_ord = [r for r in regions if
                   r in plot_df.region.unique()]
        plot_df.region = pd.Categorical(
            plot_df.region, categories=reg_ord, ordered=True)
        plot_df.sort_values('region', inplace=True)

        # plot
        xposs = np.arange(len(regions))
        x_tick_labels = [CFG.regions[r] for r in reg_ord]
        n_x_ticks = len(x_tick_labels)
        os.makedirs(op.dirname(outpath), exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize)
        means = plot_df.groupby('region').value.mean()
        sems = plot_df.groupby('region').value.sem()
        ax.plot(xposs,
                means,
                color=params['colours'][1],
                clip_on=False)
        ax.errorbar(xposs,
                    means,
                    yerr=sems,
                    fmt='o',
                    color=params['colours'][1],
                    capsize=3,
                    clip_on=False,
                    markersize=2.5,
                    linewidth=1)
        """
        sns.stripplot(
            x=[reg_ord.index(r) for r in plot_df.region],
            y=plot_df.value.to_list(),
            zorder=3,
            clip_on=False,
            native_scale=True, dodge=True, ax=ax,
            color='k', size=1,
            alpha=1, linewidth=0)
        """
        lwr, upr = 1., max(ylims)
        cl_x = (-.5, n_x_ticks - .5)
        #ax.fill_between(cl_x, lwr, upr, color='#e4e4e4', lw=0)
        ax.axhline(1, color='#d3d3d3', clip_on=False, zorder=2)
        ax.set_xticks(np.arange(n_x_ticks), labels=x_tick_labels)
        ax.set_yticks(yticks, labels=ytick_labels)
        ax.set_ylabel(params['index_label'])
        ax.set_ylim(ylims)
        #ax.set_title(f'{analysis.replace("_", " ")} index')
        ax.set_xlim((-.5, n_x_ticks - .5))
        if exp == 'exp2':
            plt.subplots_adjust(top=.48, left=.25)
        else:
            plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.savefig(outpath.replace('.pdf', '.svg'))
        plt.close()


def make_index_plots_exp2():


    out_dir = f'{figdir}/indices'
    os.makedirs(out_dir, exist_ok=True)
    figsize = (3.5, 3.2) # figure size
    yticks = (0, .5, 1)
    ytick_labels = ['0.0', '0.5', '1.0']
    exp = 'exp2'

    for analysis, params in CFG.occlusion_robustness_analyses.items():

        outpath = f'{out_dir}/{exp}_{analysis}.pdf'
        ylims = (0, 1)

        # data
        RSA_dir_attended = (f'{PROJ_DIR}/{exp}/derivatives/RSA/'
                            f'occlusionAttnOn/{analysis}')
        df_attended = pd.read_csv(f'{RSA_dir_attended}/indices.csv')
        df_attended['task'] = 'attended'
        RSA_dir_unattended = (f'{PROJ_DIR}/{exp}/derivatives/RSA/'
                              f'occlusionAttnOff/{analysis}')
        df_unattended = pd.read_csv(f'{RSA_dir_unattended}/indices.csv')
        df_unattended['task'] = 'unattended'
        df = pd.concat([df_attended, df_unattended])
        plot_df = df[
            (df.level == 'ind') &
            (df.subtype == 'prop2') &
            (df.region.isin(regions))].copy(
            deep=True).reset_index(drop=True).drop(columns=[
            'level', 'subtype', 'subject'])
        reg_ord = [r for r in regions if
                   r in plot_df.region.unique()]
        plot_df.region = pd.Categorical(
            plot_df.region, categories=reg_ord, ordered=True)
        plot_df.sort_values(['task', 'region'], inplace=True)

        # plot

        x_tick_labels = [CFG.regions[r] for r in reg_ord]
        n_x_ticks = len(x_tick_labels)
        os.makedirs(op.dirname(outpath), exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize)
        for i, (task, color) in enumerate(zip(
                ['unattended', 'attended'],
                ['tab:gray', params['colours'][1]])):
            means = plot_df[plot_df.task == task].groupby(
                'region').value.mean()
            sems = plot_df[plot_df.task == task].groupby(
                'region').value.sem()
            xposs = np.arange(len(regions), dtype=float) + (i * .1)
            ax.plot(xposs,
                    means,
                    color=color,
                    clip_on=False,
                    zorder=3)
            ax.errorbar(xposs,
                        means,
                        yerr=sems,
                        fmt='o',
                        color=color,
                        capsize=3,
                        clip_on=False,
                        markersize=2.5,
                        linewidth=1,
                        zorder=2)
            """
            sns.stripplot(
                x=[reg_ord.index(r) for r in plot_df.region],
                y=plot_df.value.to_list(),
                zorder=3,
                clip_on=False,
                native_scale=True, dodge=True, ax=ax,
                color='k', size=1,
                alpha=1, linewidth=0)
            """
            lwr, upr = 1., max(ylims)
            cl_x = (-.5, n_x_ticks - .5)
            # ax.fill_between(cl_x, lwr, upr, color='#e4e4e4', lw=0)
            ax.axhline(1, color='#d3d3d3', clip_on=False, zorder=1)
            ax.set_xticks(np.arange(n_x_ticks), labels=x_tick_labels)
            ax.set_yticks(yticks, labels=ytick_labels)
            ax.set_ylabel(params['index_label'])
            ax.set_ylim(ylims)
            # ax.set_title(f'{analysis.replace("_", " ")} index')
            ax.set_xlim((-.5, n_x_ticks - .5))
        plt.subplots_adjust(top=.48, left=.25)
        plt.savefig(outpath, dpi=300)
        plt.savefig(outpath.replace('.pdf', '.svg'))
        plt.close()


if __name__ == "__main__":
   main()