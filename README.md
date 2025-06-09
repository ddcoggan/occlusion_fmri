Analysis pipeline for fMRI experiment
The scripts and their dependencies are listed in order below 

    1 initialise_BIDS.py
    requires a configured participants.json file and raw data downloaded and unpacked in sourcedata

    2 make_events.py
    requires 1 and matlab logfiles in sourcedata

    3 preprocess.py
    requires 1

    4 FEAT_runwise.py
    requires 1-3 and a run-wise FEAT design configured for the first subject

    5 FEAT_subjectwise.py
    requires 1-4 and a subject-wise FEAT design configured for the first subject

    6 registration_and_ROI_masks.py
    requires 1-5

    7 RSA_calculate_RDMs.py
    requires 1-6
    
    8 RSA_contrasts.py
    requires 1-7

    9 RSA_stats.R
    requires 1-8

This repository contains two fMRI datasets and computational modeling data and analysis scripts for the associated publication. Top-level folders 'exp1' and 'exp2' contain the fMRI datasets with BIDS directory structure. The fMRI_pipeline.py script produces all derivatives of the source data, and calls python modules in the 'utils' directory. The 'control' directory contains the data and scripts for populations which can be reproduced from the raw data using the ' 'utils' contains scripts for fMRI data analysis and general functions like default plotting parameters. control' contains the pRF and computational modeling data and scripts, and
