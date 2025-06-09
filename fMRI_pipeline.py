"""
controller script for running the analysis pipeline
"""
import itertools
import os
import os.path as op
import time
import glob
import json
from utils import PROJ_DIR, seconds_to_text

stop_for_checks = False

if __name__ == "__main__":

    start = time.time()
    num_procs = 8

    for exp in ['exp1', 'exp2']:

        os.chdir(exp)
        subjects = json.load(open("participants.json", "r+"))
        
        # preprocess data
        from utils import preprocess
        preprocess()

        # check anatomical segmentation quality and fix errors
        from utils import check_segmentation
        if stop_for_checks:
            check_segmentation()

        # perform registration
        from utils import registration
        overwrite = []  # ['func_anat', 'anat_std']
        registration(subjects, overwrite)

        # make ROIs
        from utils import make_ROIs
        overwrite = False
        make_ROIs(subjects, overwrite)

        # FEAT analysis for each run
        from utils import FEAT_runwise
        FEAT_runwise(num_procs, overwrite)

        # check registration quality
        if stop_for_checks:
            input('Check registrations in derivatives/FEAT/runwise_reg_plots, '
                  'then press enter to continue')

        # FEAT analysis across runs for each subject
        from utils import FEAT_subjectwise
        for space in ['func', 'standard']:
            FEAT_subjectwise(num_procs, space=space)

        # FEAT analysis across entire sample
        from utils import FEAT_groupwise
        FEAT_groupwise(num_procs)

        # check ROIs
        if stop_for_checks:
            input('Check ROI images in derivatives/ROIs/plots, then press '
                  'enter to continue')
        
        # RSA
        from utils import do_RSA
        # overwrite_analyses = ['responses', 'RSA_ROI',
        # 'RSA_searchlight', 'compare_regions']
        overwrite_analyses = []
        # overwrite_plots = ['TSNE', 'RSM', 'MDS', 'RSM_models',
        # 'compare_regions']
        overwrite_plots = []  # to overwrite specific plots
        overwrite_stats = False
        do_RSA(exp, overwrite_analyses, overwrite_plots, overwrite_stats,
               num_procs)

    # final plots
    from utils import make_final_plots
    make_final_plots()

    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')


