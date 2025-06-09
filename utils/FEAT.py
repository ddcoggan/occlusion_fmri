# /usr/bin/python
# Created by David Coggan on 2023 06 23

import multiprocessing as mp
import subprocess
import shutil
import os
import os.path as op
import glob
import math
import datetime
now = datetime.datetime.now
dtstr = "%y/%m/%d %H:%M:%S"
import json
import itertools
import random
from .config import PROJ_DIR, CFG

def FEAT_runwise(n_procs=None):

    print('Running FEAT run-wise analyses...')
    exp = op.basename(os.getcwd())
    design_paths = []
    input_dir = 'fmriprep-23.0.2'

    for task in CFG.scan_params[exp]:

        design_dir = f'derivatives/FEAT/designs/runwise/{task}'
        os.makedirs(design_dir, exist_ok=True)
        conds = CFG.cond_labels['loc'] if task == 'objectLocaliser' else \
            CFG.cond_labels['exp1']
        n_contrasts = len(CFG.FEAT_contrasts[exp][task])

        # find all runs of this scan
        run_paths = sorted(glob.glob(f"derivatives/{input_dir}/**/*/" \
                                     f"*{task}*_desc-preproc" \
                                     f"_bold.nii.gz", recursive=True))
        assert len(run_paths)

        for run_path in run_paths:

            # parameters for this analysis
            subject = [x for x in op.basename(run_path).split("_") if
                       x.startswith('sub-')][-1][4:]
            session = [x for x in op.basename(run_path).split("_") if
                       x.startswith('ses-')][-1]
            run_num = [x for x in op.basename(run_path).split("_") if
                       x.startswith('run-')][-1]

            # determine which analyses need to be (re)performed
            outdir = op.abspath(f"derivatives/FEAT/sub-{subject}/" \
                     f"{session}/task-{task}/{run_num}.feat")
            print_str = f'{subject} | {session} | {task} | {run_num} |'
            if op.isdir(outdir):
                if op.isfile(f"{outdir}/stats/zstat{n_contrasts}.nii.gz"):
                    #print(f'{print_str} Analysis found and appears complete, '
                    #      f'skipping...')
                    run_feat = False
                else:
                    print(f'{print_str} Incomplete analysis found, deleting and adding '
                          f'to job list...')
                    shutil.rmtree(outdir)
                    run_feat = True
            else:
                print(f'{print_str} Analysis not found, adding to job list...')
                run_feat = True

            # design analysis
            if run_feat:

                # input basic parameters for first-level analysis
                design = ''
                for key, val in CFG.FEAT_designs['base'].items():
                    design += f'\nset fmri({key}) {val}'
                for key, val in CFG.FEAT_designs['runwise'].items():
                    design += f'\nset fmri({key}) {val}'

                # subject-specific variables
                reg_dir = op.abspath(f'derivatives/registration/sub-{subject}')
                roi_dir = op.abspath(f'derivatives/ROIs/sub-{subject}')
                ref_func = f'{reg_dir}/example_func.nii.gz'
                brain_mask = f'{roi_dir}/func_space/brain_mask.nii.gz'

                design += f'\nset feat_files(1) "{op.abspath(run_path)}"'
                design += f'\nset fmri(outputdir) "{op.abspath(outdir)}"'
                design += f'\nset alt_ex_func(1) "{ref_func}"'
                design += f'\nset fmri(alternative_mask) "{brain_mask}"'


                # different analyses for fmriprep or custom preprocessing
                preproc = 0
                design += f'\nset fmri(mc) {preproc}'  # motion correction
                design += f'\nset fmri(st) {preproc}'  # slice timing correction
                design += f'\nset fmri(motionevs) {preproc}'  # motion regressors
                design += f'\nset fmri(confoundevs) {1 - preproc}'  # confound regressors
                confound_evs = glob.glob(f'{run_path.split("-preproc")[0]}' 
                               f'-confounds_timeseries.txt')
                assert len(confound_evs) == 1
                design += f'\nset confoundev_files(1) ' \
                          f'"{op.abspath(confound_evs[0])}"'

                # GLM design
                for key, val in CFG.FEAT_designs['modeling'][exp][task].items():
                    design += f'\nset fmri({key}) {val}'
                for c, cond in enumerate(conds):
                    if task == 'objectLocaliser':
                        events_path = glob.glob(f'derivatives/FEAT' \
                              f'/events_3column/task-{task}_{cond}.txt')
                    else:
                        events_path = glob.glob(f'derivatives/FEAT' \
                              f'/events_3column/sub-{subject}/{session}/task-' \
                              f'{task}/*_{run_num}_{cond}.txt')
                    assert len(events_path) == 1
                    design += f'\nset fmri(custom{c+1}) ' \
                              f'"{op.abspath(events_path[0])}"'

                # write the file out with a unique filename
                design_path = f'{design_dir}/{subject}_{session}_{run_num}.fsf'
                with open(design_path, 'w') as file:
                    file.write(design)
                file.close()

                # store FEAT command for parallel processing
                design_paths.append(f'{design_path}')

    # run jobs in parallel
    if n_procs is None:
        n_procs = mp.cpu_count() - 4  # spare some cpus for other processes
    total_jobs = len(design_paths)
    if total_jobs:
        print(f'Total of {total_jobs} FEAT jobs, '
              f'processing in batches of {n_procs}')
        for batch in range(math.ceil(len(design_paths) / n_procs)):
            processes = []
            first_job = batch * n_procs
            last_job = min(((batch + 1) * n_procs), len(design_paths))
            jobs = design_paths[first_job:last_job]
            for job in jobs:
                print(f'{now().strftime(dtstr)} | {job}')
                p = subprocess.Popen(['feat', job])
                processes.append(p)
            for p in processes:
                p.communicate()


def FEAT_subjectwise(n_procs=None, space='func'):

    print(f'Running FEAT subject-wise analyses in {space} space...')

    subjects = json.load(open('participants.json', 'r+'))
    exp = op.basename(os.getcwd())
    design_paths = {}


    for subject in subjects:

        for task in CFG.scan_params[exp]:

            design_paths[f'{subject}_{task}'] = []
            design_dir = f"derivatives/FEAT/designs/subjectwise/{task}"
            os.makedirs(design_dir, exist_ok=True)
            n_contrasts = len(CFG.FEAT_contrasts[exp][task])

            # find run-wise dirs
            runs_all = sorted(glob.glob(f"derivatives/FEAT/sub-{subject}/"
                                        f"*/task-{task}/run-*.feat"))
            n_runs_all = len(runs_all)

            # replace reg dir depending on space
            reg_orig = f'derivatives/registration/sub-{subject}'
            if space == 'func':
                reg_orig += '_no-reg'
            for run_dir in runs_all:
                reg_dir = f'{run_dir}/reg'
                if op.islink(reg_dir):
                    subprocess.Popen(['rm', reg_dir]).wait()
                subprocess.Popen(f'ln -s {op.abspath(reg_orig)} '
                                 f'{run_dir}/reg'.split(' ')).wait()

            # make sets of runs to analyse
            run_sets = {'all-runs': runs_all}
            if 'occlusion' in task:
                n_splits = 8
                for split in range(n_splits):
                    runs_A = random.sample(runs_all, n_runs_all // 2)
                    runs_B = [x for x in runs_all if x not in runs_A]
                    run_sets[f'split-{split}A'] = runs_A
                    run_sets[f'split-{split}B'] = runs_B

            for label, runs in run_sets.items():

                n_runs = len(runs)
                outdir = op.abspath(f"derivatives/FEAT/sub-"
                          f"{subject}/subject-wise_space-{space}/task"
                          f"-{task}_{label}.gfeat")

                # determine if this analysis needs to be (re)performed
                prnt_str = f'{subject} | {task} | {label} |'
                if os.path.isdir(outdir):
                    if os.path.isfile(f"{outdir}/cope{n_contrasts}.feat/stats/"
                                      f"zstat1.nii.gz"):
                        #print(f'{prnt_str} Analysis found and appears '
                        #      f'complete, skipping...')
                        run_feat = False
                    else:
                        print(f'{prnt_str} Incomplete analysis found, '
                              f'deleting and adding to job list...')
                        shutil.rmtree(outdir)
                        run_feat = True
                else:
                    print(f'{prnt_str} Analysis not found, adding to '
                          f'job list...')
                    run_feat = True

                # design analysis
                if run_feat:

                    # input basic parameters for higher-level analysis
                    design = ''
                    for key, val in CFG.FEAT_designs['base'].items():
                        design += f'\nset fmri({key}) {val}'
                    for key, val in CFG.FEAT_designs['subjectwise'].items():
                        design += f'\nset fmri({key}) {val}'

                    # input analysis-specific parameters
                    design += f'\nset fmri(outputdir) {op.abspath(outdir)}'
                    design += f'\nset fmri(npts) {n_runs}'
                    design += f'\nset fmri(multiple) {n_runs}'
                    for r, run in enumerate(runs):
                        design += f'\nset fmri(evg{r + 1}.1) 1.0'
                        design += f'\nset fmri(groupmem.{r + 1}) 1'
                        design += f'\nset feat_files({r + 1}) ' \
                                  f'{op.abspath(run)}'
                    design += f'\nset fmri(ncopeinputs) {n_contrasts}'
                    for c in range(n_contrasts):
                        design += f'\nset fmri(copeinput.{c + 1}) 1'

                    # write the file out with a unique filename
                    design_path = f'{design_dir}/{subject}_{label}.fsf'
                    with open(design_path, 'w') as file:
                        file.write(design)
                    file.close()

                    # store FEAT command for parallel processing
                    design_paths[f'{subject}_{task}'].append(design_path)


    # run jobs in parallel
    if n_procs is None:
        n_procs = mp.cpu_count() - 4  # spare some cpus for other processes
    total_jobs = sum([len(x) for x in design_paths.values()])
    batch_size = min(n_procs, len(design_paths))
    if total_jobs:
        print(f'Total of {total_jobs} jobs, processing in batches of {batch_size}')
        job_hashes = list(design_paths.keys())
        start_idx = 0
        while sum([len(x) for x in design_paths.values()]):
            batch = []

            # Ensure jobs from same subject/scan are processed in different
            # batches, as multiple jobs using a first-level dir causes issues/
            # This gets tricky if some subjects have more scans than others as
            # final batches may be smaller
            job_hashes_ordered = job_hashes[start_idx:] + job_hashes[:start_idx]
            job_idx = 0

            while len(batch) < batch_size and job_idx < len(job_hashes_ordered):
                jobs = design_paths[job_hashes_ordered[job_idx]]
                job_idx += 1
                if len(jobs):
                    batch.append(jobs.pop())
                    start_idx += 1

            processes = []
            for job in batch:
                print(f'{now().strftime(dtstr)} | {job}')
                p = subprocess.Popen(['feat', job])
                processes.append(p)
            for p in processes:
                p.communicate()

    # remove registration links to avoid confusion
    all_reg_dirs = glob.glob(
        f'derivatives/FEAT/sub-????/ses-*/task-*/run-*.feat/reg')
    for reg_dir in all_reg_dirs:
        os.remove(reg_dir)
    all_reg_dirs = glob.glob(
        f'derivatives/FEAT/sub-????/ses-*/task-*/run-*.feat/reg_standard')
    for reg_dir in all_reg_dirs:
        shutil.rmtree(reg_dir)



def FEAT_groupwise(n_procs=None):

    print('Running FEAT group-wise analyses...')

    design_paths = []
    exp = op.basename(os.getcwd())
    subjects = CFG.subjects_final[exp]
    n_subjects = len(subjects)

    for task in CFG.scan_params[exp]:

        design_dir = f"derivatives/FEAT/designs/groupwise/{task}"
        os.makedirs(design_dir, exist_ok=True)
        n_contrasts = len(CFG.FEAT_contrasts[exp][task])

        for c in range(n_contrasts):

            outdir = op.abspath(f"derivatives/FEAT/task-{task}/cope{c+1}.gfeat")

            # determine if this analysis needs to be (re)performed
            prnt_str = f'{task} |'
            if os.path.isdir(outdir):
                if os.path.isfile(f"{outdir}/cope1.feat/stats/"
                                  f"zstat1.nii.gz"):
                    #print(f'{prnt_str} Analysis found and appears '
                    #      f'complete, skipping...')
                    run_feat = False
                else:
                    print(f'{prnt_str} Incomplete analysis found, '
                          f'deleting and adding to job list...')
                    shutil.rmtree(outdir)
                    run_feat = True
            else:
                print(f'{prnt_str} Analysis not found, adding to '
                      f'job list...')
                run_feat = True

            # design analysis
            if run_feat:

                # input basic parameters for higher-level analysis
                design = ''
                for key, val in CFG.FEAT_designs['base'].items():
                    design += f'\nset fmri({key}) {val}'
                for key, val in CFG.FEAT_designs['groupwise'].items():
                    design += f'\nset fmri({key}) {val}'

                # input analysis-specific parameters
                design += f'\nset fmri(outputdir) {op.abspath(outdir)}'
                design += f'\nset fmri(npts) {n_subjects}'
                design += f'\nset fmri(multiple) {n_subjects}'
                for s, subject in enumerate(subjects):
                    design += f'\nset fmri(evg{s + 1}.1) 1.0'
                    design += f'\nset fmri(groupmem.{s + 1}) 1'
                    input_dir = (
                        f'derivatives/FEAT/sub-{subject}/subject-wise_space-'
                        f'standard/task-{task}_all-runs.gfeat/cope{c + 1}.feat')
                    design += f'\nset feat_files({s + 1}) ' \
                              f'{op.abspath(input_dir)}'
                design += f'\nset fmri(ncopeinputs) 1'
                design += f'\nset fmri(copeinput.1) 1'

                # write the file out with a unique filename
                design_path = f'{design_dir}/cope{c+1}.fsf'
                with open(design_path, 'w') as file:
                    file.write(design)
                file.close()

                # store FEAT command for parallel processing
                design_paths.append(design_path)

    # run jobs in parallel
    if n_procs is None:
        n_procs = mp.cpu_count() - 4  # spare some cpus for other processes
    total_jobs = len(design_paths)
    if total_jobs:
        print(f'Total of {total_jobs} jobs, '
              f'processing in batches of {n_procs}')
        for batch in range(math.ceil(len(design_paths) / n_procs)):
            processes = []
            first_job = batch * n_procs
            last_job = min(((batch + 1) * n_procs), len(design_paths))
            jobs = design_paths[first_job:last_job]
            for job in jobs:
                print(f'{now().strftime(dtstr)} | {job}')
                p = subprocess.Popen(['feat', job])
                processes.append(p)
            for p in processes:
                p.communicate()


if __name__ == "__main__":

    os.chdir(f'{PROJ_DIR}/in_vivo/fMRI/exp1')
    n_procs = 10
    for orig, input_folder in zip(['', '_orig'], ['fmriprep-23.0.2',
                                                  'preprocessing']):
        FEAT_runwise(n_procs, f'derivatives{orig}/{input_folder}',
                     f'derivatives{orig}/FEAT')
        FEAT_subjectwise(n_procs, f'derivatives{orig}/FEAT')
        FEAT_groupwise(n_procs, f'derivatives{orig}/FEAT')
