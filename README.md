Analysis pipeline for fMRI experiments and computational control analyses for Coggan and Tong, "Evidence of strong amodal completion of occluded objects in both early and high-level visual cortex", currently under review at Cerebral Cortex. The data from Experiment 1 can be modelled computationally through the CogganTong2024 neural benchmark on the visual domain of Brain-Score (https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/coggan2024_fMRI), with responses in V4 underlying the winning submission in the BrainScore 2024 competition (https://www.brain-score.org/competition2024/).

This repo requires Python 3.8 installed with the packages listed in requirements.txt. You will also need docker, FSL, and Freesurfer installed and callable from your python environment via 'os.system' commands. For FSL and Freesurfer, this can often be achieved by activating these environments in a terminal before opening your python IDE with a further command within this terminal.

The fMRI data from the two experiments are not contained here, but are available from Dryad at DOI: 10.5061/dryad.5tb2rbpg1. This dataset has the same file structure as this github repo, with two additional top-level directories named 'exp1', 'exp2', in which the data lie. To reduce the size of this dataset (to save on costs), the FEAT analysis outputs were removed, but these can be regenerated by running 'fMRI_pipeline.py'. It shouldn't take longer than a day if you are not overwriting any existing analyses in the dataset (using overwrite flags), have a reasonably powerful computer, and make use of parallel processing through the 'num_procs' parameter in this script (set it to your number of CPU cores or slightly under). Prior to running this script, you may want to delete/correct any symlinks that are broken in exp?/derivatives/registration and exp?/derivatives/ROIs, as broken links can trip up the execution.

The outputs of all control analyses are already contained in the 'control' directory, but can be rerun if needed. The key scripts are:

Human fixation instability: control/fixation_instability/analyse_fixations.py
V1 pRF estimates: control/pRF/estimate_pRFs.py
VOneNet: control/VOneNet/VOneNet_pipeline.py

The stimuli used in both experiments are contained in 'exp1/stimuli/images'.


