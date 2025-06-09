# /usr/bin/python
# Created by David Coggan on 2023 06 23
import os
import os.path as op
import json
import glob
import shutil
import pandas as pd
from .apply_topup import apply_topup
from .get_wang_atlas import get_wang_atlas

def preprocess():

    subjects = json.load(open("participants.json", "r+"))
    print('Preprocessing data...')

    # mriqc (data quality measures)

    version="22.0.6"
    indir = op.abspath("")
    outdir = op.abspath(f"derivatives/mriqc-{version}")
    os.makedirs(outdir, exist_ok=True)

    # individual subjects
    new_subjects = False
    for subject in subjects:
        if not op.isdir(f"{outdir}/sub-{subject}"):
            cmd = f"docker run --rm " \
                  f"--mount type=bind,src={indir},dst=/data " \
                  f"--mount type=bind,src={outdir},dst=/out " \
                  f"--memory=32g " \
                  f"--memory-swap=32g " \
                  f"nipreps/mriqc:{version} " \
                  f"--nprocs 12 " \
                  f"--verbose-reports " \
                  f"/data /out participant --participant-label {subject}"
            os.system(cmd)
            new_subjects = True

    # group level
    if not os.path.isfile(f"{outdir}/group_bold.html") or new_subjects:
        cmd = cmd.replace(f" --participant-label {subject}", "")
        cmd = cmd.replace("participant", "group")
        os.system(cmd)


    # fMRIprep (preprocessing)

    # These are performed on individual basis as fmriprep does not check which
    # subjects are already processed
    indir = op.abspath("")
    version = "23.0.2"
    outdir = op.abspath(f"derivatives/fmriprep-{version}")
    os.makedirs(outdir, exist_ok=True)
    fs_subjs_dir = os.environ['SUBJECTS_DIR']
    workdir = f"derivatives/fmriprep_work"
    if op.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    for subject in subjects:
        if not op.isdir(f"{outdir}/sub-{subject}"):
            cmd = f"docker run --rm " \
                  f"--mount type=bind,src={indir},dst=/data " \
                  f"--mount type=bind,src={op.dirname(outdir)},dst=/out " \
                  f"--mount type=bind,src={fs_subjs_dir}," \
                  f"dst=/fs_subjects " \
                  f"--mount type=bind,src={op.abspath(workdir)},dst=/work " \
                  f"--memory=32g " \
                  f"--memory-swap=64g " \
                  f"nipreps/fmriprep:{version} " \
                  f"--clean-workdir " \
                  f"--nprocs 12 " \
                  f"--mem-mb 64000 " \
                  f"--fs-license-file /fs_subjects/license.txt " \
                  f"--fs-subjects-dir /fs_subjects " \
                  f"--output-spaces func " \
                  f"-w /work " \
                  f"/data /out/{op.basename(outdir)} " \
                  f"participant --participant-label {subject}"
            os.system(cmd)

    shutil.rmtree(workdir)

    # final steps
    for subject in subjects:

        fs_subj_dir = sorted(glob.glob(f"{fs_subjs_dir}/sub-{subject}*"))[-1]

        # make link to freesurfer subject dir in local project dir
        os.makedirs(f'derivatives/freesurfer', exist_ok=True)
        fs_link = f'derivatives/freesurfer/sub-{subject}'
        if not op.exists(fs_link):
            os.system(f"ln -s {fs_subj_dir} {fs_link}")

        # convert anatomicals to nifti and extract brain
        # (final preprocessed anatomical and original anatomical)
        fs_dir = f'derivatives/freesurfer/sub-{subject}'
        for mgz in [f'{fs_dir}/mri/T1.mgz', f'{fs_dir}/mri/orig/001.mgz']:

            # convert to nifti
            nii = f'{mgz[:-4]}.nii'
            if not op.isfile(nii):
                os.system(f'mri_convert {mgz} {nii}')

            # extract brain
            nii_brain = f'{mgz[:-4]}_brain.nii.gz'
            if not op.isfile(nii_brain):
                os.system(f'mri_synthstrip -i {nii} -o {nii_brain} -g')

        # convert head motion confounds, including temporal derivatives
        # and quadratics from .tsv to .txt
        confounds_paths = glob.glob(
            f"{outdir}/sub-{subject}**/*/*desc-confounds_timeseries.tsv",
            recursive=True)
        for confounds_path in confounds_paths:
            outpath = f"{confounds_path[:-4]}.txt"
            if not op.isfile(outpath):
                confounds = pd.read_csv(confounds_path, delimiter='\t')
                motion_confounds = [x for x in confounds.columns if \
                    "trans" in x or "rot" in x]
                extracted_confounds = confounds[motion_confounds]
                extracted_confounds.to_csv(outpath, sep="\t", index=False)
