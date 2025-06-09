# /usr/bin/python
# Created by David Coggan on 2023 06 28
import os
import os.path as op
import glob
import json
import shutil
from .config import CFG, PROJ_DIR
from .get_wang_atlas import get_wang_atlas


def make_ROIs(subjects, overwrite):

    print('Making ROIs...')

    # make copy of standard space masks in project directory
    mask_dir_std_local = f'derivatives/ROIs/MNI152_2mm'
    os.makedirs(mask_dir_std_local, exist_ok=True)
    std_dir = f'{mask_dir_std_local}'
    os.makedirs(std_dir, exist_ok=True)
    for region in CFG.regions:
        mask_path_local = f'{std_dir}/{region}.nii.gz'
        if not op.exists(mask_path_local):
            mask_path = glob.glob(
                f'{CFG.mask_dir_std}/*/{region}.nii.gz')
            assert len(mask_path) == 1
            mask_path = mask_path[0]
            os.system(f'ln -sf {op.abspath(mask_path)} {mask_path_local}')

    # std brain
    brain_link = f'derivatives/ROIs/MNI152_2mm/standard.nii.gz'
    if not op.exists(brain_link):
        ref_std = (f'{os.environ["FSLDIR"]}/data/standard/'
                   f'MNI152_T1_2mm_brain.nii.gz')
        os.system(f'ln -s {ref_std} {brain_link}')

    # std brain mask
    mask_link = f'derivatives/ROIs/MNI152_2mm/brain_mask.nii.gz'
    if not op.exists(mask_link):
        brain_mask_std = (f'{os.environ["FSLDIR"]}/data/standard/'
                          f'MNI152_T1_2mm_brain_mask.nii.gz')
        os.system(f'ln -s {brain_mask_std} {mask_link}')


    # freesurfer based ROIs
    for subj in ['fsaverage'] + list(subjects.keys()):

        fs_subj = f'sub-{subj}' if subj != 'fsaverage' else subj
        fs_subj_dir = f'{os.environ["SUBJECTS_DIR"]}/{fs_subj}'

        # get retinotopy estimates
        get_wang_atlas(fs_subj)  # automatically skips if done

    # native space ROIs
    for subject in subjects:

        reg_dir = f'derivatives/registration/sub-{subject}'
        ref_func = (f'sub-{subject}/ses-1/fmap/sub-{subject}_ses-1_'
                    f'acq-funcNoEPI_magnitude.nii')
        fs_subj = f'sub-{subject}'
        fs_subj_dir = f'{os.environ["SUBJECTS_DIR"]}/{fs_subj}'
        ref_anat = f'{fs_subj_dir}/mri/orig/001.nii'

        # make eccentricity ROI within 4.5 degrees
        for hemi in ['lh', 'rh']:
            ecc_label = f'{fs_subj_dir}/surf/eccen_{hemi}.label'
            if not op.isfile(ecc_label) or overwrite:
                all_ret = f'{fs_subj_dir}/surf/{hemi}.wang15_mplbl.mgz'
                ecc_map = f'{fs_subj_dir}/surf/{hemi}.benson14_eccen.mgz'
                ecc_bin = f'{fs_subj_dir}/surf/eccen_{hemi}.mgz'
                cmd = (f'mri_binarize '
                       f'--i {ecc_map} '
                       f'--max 4.5 '
                       f'--mask {all_ret} '
                       f'--o {ecc_bin}')
                os.system(cmd)
                cmd = (f'mri_cor2label '
                       f'--i {ecc_bin} '
                       f'--surf {fs_subj} {hemi} '
                       f'--id 1 '
                       f'--l {op.abspath(ecc_label)}')
                os.system(cmd)

        # create ROI masks
        mask_dir = f'derivatives/ROIs/sub-{subject}'
        std_dir = f'{mask_dir_std_local}'
        anat_dir = f'{mask_dir}/anat_space'
        os.makedirs(anat_dir, exist_ok=True)
        func_dir = f'{mask_dir}/func_space'
        os.makedirs(func_dir, exist_ok=True)

        for region in CFG.regions:

            # std space to anat space
            mask_path_std = f'{std_dir}/{region}.nii.gz'
            mask_path_anat = f'{anat_dir}/{region}.nii.gz'
            standard2highres_warp = (f'{fs_subj_dir}/mri/transforms/fnirt/'
                                f'standard2highres_warp.nii.gz')
            if not op.isfile(mask_path_anat) or overwrite:
                os.makedirs(op.dirname(mask_path_anat), exist_ok=True)
                os.system(f'applywarp '
                          f'-i {mask_path_std} '
                          f'-o {mask_path_anat} '
                          f'-r {ref_anat} '
                          f'-w {standard2highres_warp} '
                          f'--interp=nn')

            # anat space to func space
            mask_path_func = f'{func_dir}/{region}.nii.gz'
            if not op.isfile(mask_path_func) or overwrite:
                highres2example_func = f'{reg_dir}/highres2example_func.mat'
                os.system(
                    f'flirt -in {mask_path_anat} -ref {ref_func} -applyxfm '
                    f'-init {highres2example_func} -out {mask_path_func} '
                    f'-interp nearestneighbour')
                """
                example_func2highres = f'{reg_dir}/example_func2highres.lta'
                os.system(
                    f'mri_vol2vol --mov {ref_func} --targ {mask_path_anat} '
                    f'--lta {example_func2highres} --inv --o {mask_path_func}'
                    f' --nearest')
                """

            # ROI plots
            plot_dir = f'derivatives/ROIs/plots'
            os.makedirs(plot_dir, exist_ok=True)
            for space, ref, mask_path in zip(['anat', 'func'],
                                             [ref_anat, ref_func],
                                             [mask_path_anat, mask_path_func]):

                plot_file = f'{plot_dir}/sub-{subject}_{space}_{region}.png'
                if not op.isfile(plot_file) or overwrite:
                    ref_range = os.popen(f'fslstats {ref} -R')
                    ref_max = float(ref_range.read().split()[1])
                    coords = os.popen(f'fslstats {mask_path} -C').read()[:-2].split(
                        ' ')
                    coords = [int(float(c)) for c in coords]
                    cmd = f'fsleyes render --outfile {plot_file} --size 3200 ' \
                          f'600 --scene ortho --autoDisplay -vl {coords[0]} ' \
                          f'{coords[1]} {coords[2]} ' \
                          f'{ref} -dr 0 {ref_max} {mask_path} -dr 0 1 -cm ' \
                          f'greyscale'
                    os.system(cmd)

        # make links to reference anat image
        local_anat = f'{mask_dir}/anat_space/ref_anat.nii.gz'
        if op.exists(local_anat) and overwrite:
            os.remove(local_anat)
        if not op.exists(local_anat):
            os.system(f'ln -s {op.abspath(ref_anat)} {local_anat}')

        # make links to reference func image
        local_func = f'{mask_dir}/func_space/ref_func.nii.gz'
        if op.exists(local_func) and overwrite:
            os.remove(local_func)
        if not op.exists(local_func):
            os.system(f'ln -sf {op.abspath(ref_func)} {local_func}')

        # native func brain mask
        mask_path = f'{mask_dir}/func_space/brain_mask.nii.gz'
        if not op.exists(mask_path):
            if not op.isfile(mask_path):  # or overwrite:
                os.system(f'mri_synthstrip -i {ref_func} -m {mask_path} -g')


if __name__ == "__main__":

    overwrite = False
    for exp in ['exp1','exp2']:
        os.chdir(f'{PROJ_DIR}/in_vivo/fMRI/{exp}')
        subjects = json.load(open("participants.json", "r+"))
        make_ROIs(subjects, overwrite)
