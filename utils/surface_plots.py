# /usr/bin/python
# Created by David Coggan on 2023 06 23

"""
script for making and plotting robustness measures on the surface
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
from rsatoolbox.util.searchlight import get_volume_searchlight
import pandas as pd
import matplotlib
from tqdm import tqdm
import datetime
from config import CFG, TABCOLS, PROJ_DIR
from plot_utils import export_legend, custom_defaults
now = datetime.datetime.now

"""
# scripts for flattening surface
subjects_surf = ['F135', 'F019']
subject = 'F019'
hemi = 'rh'
subj_dir = f'{os.environ["SUBJECTS_DIR"]}/sub-{subject}'
cmd = (
    f'freeview '
    f'-f {subj_dir}/surf/{hemi}.inflated'
    f':curvature_method=binary'
    f':label={subj_dir}/label/{hemi}.wang15_mplbl.V1v.label'
    f':label_color=255,255,255:label_outline=yes'
    f':label={subj_dir}/label/{hemi}.wang15_mplbl.V1d.label'
    f':label_color=255,255,255:label_outline=yes'
    f':annot={subj_dir}/label/{hemi}.aparc.a2009s.annot'
    f' -layout 1 -viewport 3d')
os.system(cmd)

# make edits, save patch as {subj_dir}/surf/{hemi}}.full.patch.3d

cmd = (f'mris_flatten -O inflated -distances 64 32 '
    f'{subj_dir}/surf/{hemi}.full.patch.3d {subj_dir}/surf/{hemi}.full.flat.patch.3d')
print(cmd)
os.system(cmd)
"""

def make_surface_maps(nifti, subject, overwrite):

    fs_subj = f'sub-{subject}' if subject != 'fsaverage' else subject
    for hemi in ['lh', 'rh']:
        overlay = nifti.replace('.nii.gz', f'_{hemi}.mgh')
        if not op.isfile(overlay) or overwrite:
            print(f'converting {hemi} from anat space to surface space')
            cmd = (f'mri_vol2surf --mov {nifti} --out {overlay} ' 
                   f'--regheader {fs_subj} --hemi {hemi} --interp nearest '
                   f'--cortex')
            os.system(cmd)


def plot_surface_maps(subject, overlay, hemi, out_path):

    print(f'{now().strftime("%d/%m/%Y %H:%M:%S")} {subject}')
    fs_subj = f'sub-{subject}' if subject != 'fsaverage' else subject
    fs_subj_dir = f'{os.environ["SUBJECTS_DIR"]}/{fs_subj}'
    subj_dir = op.dirname(overlay)
    thr, uthr = 0, 1

    ecc_colour = '255,80,80'
    ROI_colour = '255,255,255'
    # perceptually uniform colormaps are:
    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    cmap_orig = matplotlib.colormaps['viridis'].colors[::32]
    cmap = [','.join([str(int(255*c)) for c in rgb]) for rgb in cmap_orig]
    cvals = np.linspace(thr, uthr, len(cmap))
    cmap_str = f'0,{cmap[0]}'
    for i, c in zip(cvals[1:], cmap[1:]):
        cmap_str += f',{i:.4f},{c}'

    cmd = f'freeview '
    #FOV_mask = f'{subj_dir}/FOV_{hemi}.label'
    ecc_label = f'{fs_subj_dir}/label/{hemi}.benson14_eccen_4pt5.label'
    ROIs = glob.glob(f'{fs_subj_dir}/label/{hemi}.wang15_mplbl.*.label')
    #offset = '0,-115,0' if hemi == 'lh' else '0,155,0'
    offset = '0,0,0'
    cmd += (
        f'-f {fs_subj_dir}/surf/{hemi}.inflated'
        f':curvature_method=binary'
        f':offset={offset}'
        f':patch={fs_subj_dir}/surf/{hemi}.full.flat.patch.3d'
        f':overlay={overlay}'
        f':overlay_custom={cmap_str}'
        #f':overlay_mask={FOV_mask}'
    )
    for ROI in ROIs:
        cmd += f':label={ROI}:label_color={ROI_colour}:label_outline=yes'
    #cmd += f':label={ecc_label}:label_color={ecc_colour}:label_outline=yes'

    cmd += (f' -layout 1 -viewport 3d -view right -cam elevation 90 '
            f'zoom 1.4 -colorscale -ss {out_path} 2 autotrim')
    os.system(cmd)


def surface_plots(overwrite=False, overwrite_plots=False,
                  derdir='derivatives', index_type='prop'):

    exp = op.basename(os.getcwd())
    tasks = [task for task in CFG.scan_params[exp] if 'occlusion' in task]
    norm = 'all-conds'
    norm_method = 'z-score'
    similarity = 'pearson'

    """
    # convert func space maps to standard space before averaging
    for task, subject in itertools.product(tasks, CFG.subjects_final[exp]):

        print(f'{exp} | {task} | {subject} ')

        sl_dir = f'{derdir}/RSA_searchlight/task-{task}_space-func/norm' \
                 f'-{norm}_{norm_method}/{similarity}/{index_type}'
        subj_dir = f'{sl_dir}/sub-{subject}'

        ref_std = (f'{os.environ["FSLDIR"]}/data/standard/'
                   f'MNI152_T1_2mm.nii.gz')
        nifti = f'{subj_dir}/completion.nii.gz'
        nifti_std = nifti.replace('.nii', '_standard.nii')
        if not op.isfile(nifti_std) or overwrite:
            print('converting from func space to standard space')
            warp = (f'{derdir}/registration/sub-{subject}/'
                    f'example_func2standard_warp.nii.gz')
            os.system(f'applywarp '
                      f'-i {nifti} '
                      f'-r {ref_std} '
                      f'-o {nifti_std} '
                      f'-w {warp} '
                      f'--interp=nn')
    """
    # make average searchlight map
    for task, space in itertools.product(tasks, ['standard']):

        sl_dir = f'{derdir}/RSA_searchlight/task-{task}_space-{space}/norm' \
                 f'-{norm}_{norm_method}/{similarity}/{index_type}'
        subj_dir = f'{sl_dir}/sub-fsaverage'
        os.makedirs(subj_dir, exist_ok=True)

        nifti = f'{subj_dir}/completion_group.nii.gz'
        ind_name = 'completion_standard' if space == 'func' else 'completion'
        ind_paths = [f'{sl_dir}/sub-{subject}/{ind_name}.nii.gz' for subject in
                     CFG.subjects_final[exp]]
        collated = nifti.replace('.nii.gz', '_all-subjects.nii.gz')
        if not op.isfile(nifti) or overwrite:
            cmd = f'fslmerge -t {collated}'
            for ind_path in ind_paths:
                cmd += f' {ind_path}'
            os.system(cmd)
            os.system(f'fslmaths {collated} -Tmean {nifti}')


    # preprocess maps
    for task, subject, space in itertools.product(
            tasks, CFG.subjects_surf[exp], ['standard']):

        sl_dir = f'{derdir}/RSA_searchlight/task-{task}_space-{space}/norm' \
                 f'-{norm}_{norm_method}/{similarity}/{index_type}'
        subj_dir = f'{sl_dir}/sub-{subject}'
        fs_subj = 'fsaverage' if subject == 'fsaverage' else f'sub-{subject}'
        nifti = f'{subj_dir}/completion_group.nii.gz'

        # dilate completion map
        nifti_dil = nifti.replace('.nii.gz', '_dil.nii.gz')
        if not op.isfile(nifti_dil) or overwrite:
            os.system(f'fslmaths {nifti} -add 1 -kernel sphere 1 -dilall -sub 1 '
                      f'{nifti_dil}')

        # clip completion values to 0-1 range
        nifti_clip = nifti_dil.replace('.nii.gz', '_clip.nii.gz')
        if not op.isfile(nifti_clip) or overwrite:
            # set voxels < 0 or > 1 to zero
            os.system(f'fslmaths {nifti_dil} -thr 0 -uthr 1 {nifti_clip}')
            # add 1 to voxels that were > 1
            os.system(f'fslmaths {nifti_dil} -thr 1 -bin -add {nifti_clip}'
                      f' {nifti_clip}')



        # apply voxel selection at this stage as freeview CLI screenshot has a
        # bug where overlay masks aren't applied. Set undesired voxels to -1.

        # get field of view mask
        if subject == 'fsaverage':
            FOV = f'{subj_dir}/FOV_mask.nii.gz'
        else:
            FOV = (
                f'{derdir}/FEAT/sub-{subject}/subject-wise_space-{space}/'
                f'task-{task}_all-runs.gfeat/cope1.feat/mask.nii.gz')

        # mask the completion map
        nifti_FOV = nifti_clip.replace('.nii.gz', '_FOV.nii.gz')
        if not op.isfile(nifti_FOV) or overwrite:
            #os.system(f'fslmaths {nifti_clip} -mas {FOV} {nifti_FOV}')
            os.system(f'fslmaths {nifti_clip} -add 1 -thr .999 -sub 1'
                      f' {nifti_FOV}')
            # set non-FOV voxels to -1
            os.system(f'fslmaths {FOV} -sub 1 -add {nifti_FOV} {nifti_FOV}')

            """ 
            code for making FOV label to be applied in freeview 
            # convert to mgz, add dilation here if desired. mri_binarize function
            # used as mri_convert makes mri_vol2surf barf.
            FOV_mgz = FOV.replace('.nii.gz', '.mgz')
            if not op.isfile(FOV_mgz) or overwrite:
                cmd = f'mri_binarize --i {FOV} --min 1 --o {FOV_mgz}'
                os.system(cmd)
            for hemi in ['lh', 'rh']:
                FOV_overlay = f'{subj_dir}/FOV_{hemi}.mgh'
                if not op.isfile(FOV_overlay) or overwrite:
                    cmd = (f'mri_vol2surf --mov {FOV_mgz} --out {FOV_overlay} '
                           f'--regheader {fs_subj} --hemi {hemi}')
                    os.system(cmd)
                FOV_label = f'{subj_dir}/FOV_{hemi}.label'
                if not op.isfile(FOV_label) or overwrite:
                    cmd = (f'mri_cor2label '
                           f'--i {op.abspath(FOV_overlay)} '
                           f'--surf {fs_subj} {hemi} '
                           f'--id 1 '
                           f'--l {op.abspath(FOV_label)}')
                    os.system(cmd)
            """

        # make active voxels mask
        mask_act = f'{subj_dir}/active_mask.nii.gz'
        if not op.isfile(mask_act) or overwrite:
            active_task = 'occlusion' if exp == 'exp1' else 'occlusionAttnOn'

            if subject == 'fsaverage':

                # group mean FEAT dir
                #loc_path = sorted(glob.glob(
                #    f'{derdir}/FEAT/task-{task}/cope25.gfeat/cope1.feat/stats/'
                #    f'zstat1.nii.gz'))[0]

                # get voxels for which most subjects (>= 5) show z > thr
                collated = mask_act.replace('.nii.gz', '_all-subjects.nii.gz')
                ind_paths = [(
                    f'{derdir}/FEAT/sub-{subj}/subject-wise_space-standard/'
                    f'/task-{active_task}_all-runs.gfeat/cope25.feat/stats/'
                    f'zstat1.nii.gz') for subj in CFG.subjects_final[exp]]
                cmd = f'fslmerge -t {collated}'
                for ind_path in ind_paths:
                    cmd += f' {ind_path}'
                os.system(cmd)
                cmd = (f'fslmaths {collated} -Tmean -thr 2.4 -bin {mask_act}')
                os.system(cmd)

            else:  # if subject != 'fsaverage'
                loc_path = (
                    f'{derdir}/FEAT/sub-{subject}/subject-wise_space-{space}/'
                    f'task-{active_task}_all-runs.gfeat/cope25.feat/stats/'
                    f'zstat1.nii.gz')
                cmd = f'fslmaths {loc_path} -thr 3.1 -bin {mask_act}'
                os.system(cmd)


        # mask the completion map
        nifti_act = nifti_clip.replace('.nii.gz', '_active.nii.gz')
        if not op.isfile(nifti_act) or overwrite:
            os.system(f'fslmaths {nifti_clip} -mas {mask_act} {nifti_act}')
            # set non-active voxels to -1
            os.system(f'fslmaths {mask_act} -sub 1 -add {nifti_act} '
                      f'{nifti_act}')

            """ 
            scripts for making active voxels label to be applied in freeview 
            loc_path_mask = loc_path.replace('.nii.gz', '_thr-3pt1_mask.mgz')
            if not op.isfile(loc_path_mask) or overwrite:
                cmd = (
                    f'mri_binarize --i {loc_path} --min 3.1 --o {loc_path_mask}')
                os.system(cmd)
            for hemi in ['lh', 'rh']:
                act_overlay = f'{subj_dir}/active_{hemi}.mgh'
                if not op.isfile(act_overlay) or overwrite:
                    cmd = (
                        f'mri_vol2surf --mov {loc_path_mask} --out {act_overlay} '
                        f'--regheader {fs_subj} --hemi {hemi}')
                    os.system(cmd)
                act_label = f'{subj_dir}/active_{hemi}.label'
                if not op.isfile(act_label):
                    cmd = (
                        f'mri_vol2label --i {act_overlay} --surf {fs_subj} {hemi} '
                        f'--id 1 --l {op.abspath(act_label)}')
                    os.system(cmd)
            """

        # make surface plots using each type of voxel mask
        for mask_type in ['FOV', 'active']:

            nifti_final = nifti_clip.replace('.nii.gz', f'_{mask_type}.nii.gz')

            # individual maps must first be converted to anatomical space
            if subject != 'fsaverage':

                nifti = nifti_clip
                nifti_final = nifti_final.replace('.nii', '_anat.nii')
                ref_anat = f'{derdir}/ROIs/sub-{subject}/anat_space/ref_anat.nii.gz'

                if not op.isfile(nifti_final) or overwrite_plots:

                    if space == 'func':
                        print('converting from func space to anat space')
                        xform_func_to_anat = (
                            f'{derdir}/registration/sub-{subject}/'
                            f'example_func2highres.mat')
                        os.system(
                            f'flirt -in {nifti} -ref {ref_anat} -applyxfm '
                            f'-init {xform_func_to_anat} '
                            f'-out {nifti_final} '
                            f'-interp nearestneighbour')

                    else:  # if space == 'standard'
                        if not op.isfile(nifti_final) or overwrite:
                            print('converting from standard space to anat space')
                            warp = (f'{derdir}/registration/sub-{subject}/'
                                    f'standard2highres_warp.nii.gz')
                            os.system(f'applywarp '
                                      f'-i {nifti} '
                                      f'-r {ref_anat} '
                                      f'-o {nifti_final} '
                                      f'-w {warp} '
                                      f'--interp=nn')


            # make surface overlay
            make_surface_maps(nifti_final, subject, overwrite)

            # plot on surface
            for hemi in ['lh', 'rh']:
                overlay = op.abspath(nifti_final.replace(
                    '.nii.gz', f'_{hemi}.mgh'))
                out_path = overlay.replace('.mgh', '.png')
                if not op.isfile(out_path) or overwrite_plots:
                    print(f'plotting surface at {out_path}')
                    plot_surface_maps(subject, overlay, hemi, out_path)

            # make colourbar
            colourbar_path = f'{op.dirname(nifti_final)}/colourbar.pdf'
            if not op.isfile(colourbar_path) or overwrite:
                fig = plt.figure(figsize=(5,1.2))
                ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
                cb = matplotlib.colorbar.ColorbarBase(
                    ax, orientation='horizontal', cmap='viridis')
                cb.outline.set_visible(False)
                plt.text(0.5,-3, 'object completion index', ha='center')
                plt.savefig(colourbar_path, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":

    for exp in ['exp1']:#, 'exp2']:
        os.chdir(f'{PROJ_DIR}/data/in_vivo/fMRI/{exp}')
        surface_plots(overwrite=False, overwrite_plots=True,
                      derdir=f'derivatives', index_type='prop')


