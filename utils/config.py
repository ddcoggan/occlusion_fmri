# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
Contains variables and functions that are useful across fMRI experiments
"""

import numpy as np
import itertools
import os.path as op
import matplotlib
import matplotlib.colors as mcolors
import json
from copy import deepcopy


PROJ_DIR = op.expanduser('/mnt/NVMe2_1TB/Dryad_dataset')
TABCOLS = list(mcolors.TABLEAU_COLORS)
T20COLS = list(matplotlib.cm.tab20.colors)


class CFG:

    # subjects
    subjects = {exp: list(json.load(open(
        f'{PROJ_DIR}/{exp}/participants.json', 'r+'
    )).keys()) for exp in ['exp1', 'exp2']}
    subjects_surf = {'exp1': ['fsaverage'], 'exp2': ['fsaverage']}

    # design
    exemplars = ['bear', 'bison', 'elephant', 'hare',
                 'jeep', 'lamp', 'sportsCar', 'teapot']
    occluders = ['none', 'lower', 'upper']
    occluder_labels = ['complete', 'lower', 'upper']
    n_exem = len(exemplars)
    n_occ = len(occluders)
    conds_iter = list(itertools.product(*[exemplars, occluders]))
    n_img = len(conds_iter)
    cond_labels = {'exp1': [], 'exp2': []}
    for cond in conds_iter:
        cond_labels['exp1'].append(f'{cond[0]}_{cond[1]}')
        cond_labels['exp2'].append(f'attn-on_{cond[0]}_{cond[1]}')
    for cond in conds_iter:
        cond_labels['exp2'].append(f'attn-off_{cond[0]}_{cond[1]}')
    cond_labels['loc'] = ['face', 'house', 'object', 'scrambled']
    attns = ['attn-on', 'attn-off']
    attns_path = ['AttnOn', 'AttnOff']
    exps_attns = [
        ('exp1', ''),
        ('exp2', 'attn-on_'),
        ('exp2', 'attn-off_'),
        ('exp2', 'attn-on-off_')]
    exps_tasks = [
        ('exp1', 'occlusion'),
        ('exp2', 'occlusionAttnOn'),
        ('exp2', 'occlusionAttnOff')]

    # scanning parameters
    scan_params = {
        'exp1': {
            'occlusion': {
                'TR': 2,
                'dynamics': 133,
                'initial_fixation': 16,
                'final_fixation': 10,
                'block_duration': 4,
                'interblock_interval': 6,
                'conditions': {
                    'image': exemplars,
                    'occlusion': occluders},
                'block_order': 'randomised'
            }}}
    scan_params['exp2'] = {
        'occlusionAttnOn': scan_params['exp1']['occlusion'].copy(),
        'occlusionAttnOff': scan_params['exp1']['occlusion'].copy()}


    # ROIs mapped from labels to filenames
    regions = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'hV4': 'V4',
        'ventral_stream_sub_V1-V4': 'IT'}

    # list regions in desired order for each set
    region_sets = {
        'EVC_IT': ['V1', 'V2', 'V3', 'hV4', 'ventral_stream_sub_V1-V4']}

    # location of masks in std space
    mask_dir_std = f'{PROJ_DIR}/masks'

    # methods
    similarity_label = {'pearson': "correlation ($\it{r}$)"}
    off_diag_mask_flat = np.array(1 - np.eye(n_img).flatten(), dtype=bool)

    # occlusion robustness analyses
    occlusion_robustness_analyses = {
        'object_completion': {
            'conds': ['EsOs', 'EsOd', 'EdOs', 'EdOd'],
            'labels': [
                'same object, same occluder',
                'same object, different occluders',
                'different objects, same occluder',
                'different objects, different occluders'],
            'index_label': 'OCI',
            'subtypes': ['prop'],
            'colours': T20COLS[4:6] + T20COLS[2:4],
            'ylims': (0, 1.1),
            'ylabel': 'OCI',
        },
        'occlusion_invariance': {
            'conds': ['EsUb', 'EsU1', 'EdUb', 'EdU1'],
            'labels': [
                'same object, both complete',
                'same object, occluded vs. complete',
                'different object, both complete',
                'different object, occluded vs. complete'],
            'index_label': 'OII',
            'subtypes': ['prop'],
            'colours': T20COLS[:2] + T20COLS[6:8],
            'ylims': (0, 1.1),
            'ylabel': 'OII',
        },
    }

    # contrast matrices for object completion and occlusion invariance
    contrast_mats = {
        'all': np.empty((n_img, n_img)),
        'object_completion': np.zeros((n_img, n_img)),
        'occlusion_invariance': np.zeros((n_img, n_img)),
    }
    for c_a, cond_label_a in enumerate(cond_labels['exp1']):

        exem_a = cond_label_a.split('_')[0]
        occ_a = cond_label_a.split('_')[1]

        for c_b, cond_label_b in enumerate(cond_labels['exp1']):

            exem_b = cond_label_b.split('_')[0]
            occ_b = cond_label_b.split('_')[1]

            if occ_a != 'none' and occ_b != 'none':
                if exem_a == exem_b and occ_a == occ_b:
                    contrast_mats['all'][c_a, c_b] = 1
                    contrast_mats['object_completion'][c_a, c_b] = 1
                elif exem_a != exem_b and occ_a == occ_b:
                    contrast_mats['all'][c_a, c_b] = 3
                    contrast_mats['object_completion'][c_a, c_b] = 3
                elif exem_a == exem_b and occ_a != occ_b:
                    contrast_mats['all'][c_a, c_b] = 2
                    contrast_mats['object_completion'][c_a, c_b] = 2
                elif exem_a != exem_b and occ_a != occ_b:
                    contrast_mats['all'][c_a, c_b] = 4
                    contrast_mats['object_completion'][c_a, c_b] = 4

            # 2x2, exemplar by one occluded yes/no
            elif occ_a == 'none' and occ_b == 'none':  # neither occluded
                if exem_a == exem_b:
                    contrast_mats['all'][c_a, c_b] = 5
                    contrast_mats['occlusion_invariance'][c_a, c_b] = 5
                else:
                    contrast_mats['all'][c_a, c_b] = 7
                    contrast_mats['occlusion_invariance'][c_a, c_b] = 7
            elif int(occ_a == 'none') + int(occ_b == 'none') == 1:  # 1 occluded
                if exem_a == exem_b:
                    contrast_mats['all'][c_a, c_b] = 6
                    contrast_mats['occlusion_invariance'][c_a, c_b] = 6
                else:
                    contrast_mats['all'][c_a, c_b] = 8
                    contrast_mats['occlusion_invariance'][c_a, c_b] = 8


    # RSM_models

    # RSM masks
    mask_bothocc = np.ones((n_img, n_img))
    mask_lt2occ = np.ones((n_img, n_img)) * np.nan
    for c in range(n_img):
        if c % n_occ == 0:
            mask_bothocc[c, :] = np.nan
            mask_bothocc[:, c] = np.nan
            mask_lt2occ[c, :] = 1
            mask_lt2occ[:, c] = 1

    # base models (image / exemplar based)
    image_model = np.eye(n_img)
    exemplar_model = np.ones((n_img, n_img)) * 0
    for e in range(n_exem):
        image_model[(e * n_occ), (e * n_occ + 1):(e * n_occ + n_occ)] = .5
        image_model[(e * n_occ + 1):(e * n_occ + n_occ), (e * n_occ)] = .5
        exemplar_model[(e * n_occ):(e * n_occ + n_occ),
            (e * n_occ):(e * n_occ + n_occ)] = 1

    # base models (occluder based)
    occluder_presence_model = np.ones((n_img, n_img)) * 0
    occluder_cond_model = np.ones((n_img, n_img)) * 0
    for c in range(n_img):
        occluder_cond_model[c % n_occ::n_occ, c] = 1
        if c % n_occ == 0:
            occluder_presence_model[c, np.arange(0, n_img, n_occ)] = 1
        else:
            occluder_presence_model[c, np.arange(1, n_img, n_occ)] = 1
            occluder_presence_model[c, np.arange(2, n_img, n_occ)] = 1

    RSM_models = {
        'matrices': {
            'identity': np.eye(n_img),  # same exact image
            'image': image_model,  # proportion of shared, unoccluded pixels
            'exemplar': exemplar_model,
            'exemplar_bothocc': exemplar_model * mask_bothocc,
            'exemplar_lt2occ': exemplar_model * mask_lt2occ,
            'occluder_cond': occluder_cond_model,  # all three conds
            'occluder_position': occluder_cond_model * mask_bothocc,  # upper vs lower
            'occluder_presence': occluder_presence_model,
            'occluder_presence_lt2occ': occluder_presence_model * mask_lt2occ},
        'labels': ['identity', 'image', 'exemplar',
                   'exemplar (occluded only)', 'exemplar (not both occluded)',
                   'occluder condition', 'occluder position',
                   'occluder presence', 'occluder presence (not both occluded)'],
        'final_set': ['exemplar_bothocc', 'occluder_position'],
        'colours': matplotlib.colormaps['Dark2'].colors[:9],
        'ylims': (-.05, .3),
        'ylabel': r"regression coeff. ($\beta$)",
    }

    for e in range(n_exem):
        RSM_models['matrices']['image'] \
            [(e * n_occ), (e * n_occ + 1):(e * n_occ + n_occ)] = .5
        RSM_models['matrices']['image'] \
            [(e * n_occ + 1):(e * n_occ + n_occ), (e * n_occ)] = .5
        RSM_models['matrices']['exemplar'] \
            [(e * n_occ):(e * n_occ + n_occ),
            (e * n_occ):(e * n_occ + n_occ)] = 1
    for c in range(n_img):

        RSM_models['matrices']['occluder_cond'][c % n_occ::n_occ, c] = 1

        if c % n_occ == 0:
            RSM_models['matrices']['occluder_presence'] \
                [c, np.arange(0, n_img, n_occ)] = 1
            RSM_models['matrices']['occluder_presence_lt2occ'] \
                [c, np.arange(0, n_img, n_occ)] = 1
            RSM_models['matrices']['occluder_position'][c,:] = np.nan
            RSM_models['matrices']['occluder_position'][:,c] = np.nan
            RSM_models['matrices']['exemplar_bothocc'][c, :] = np.nan
            RSM_models['matrices']['exemplar_bothocc'][:, c] = np.nan
        else:
            RSM_models['matrices']['occluder_presence'] \
                [c, np.arange(1, n_img, n_occ)] = 1
            RSM_models['matrices']['occluder_presence'] \
                [c, np.arange(2, n_img, n_occ)] = 1
            RSM_models['matrices']['occluder_presence_lt2occ'] \
                [c, np.arange(1, n_img, n_occ)] = np.nan
            RSM_models['matrices']['occluder_presence_lt2occ'] \
                [c, np.arange(2, n_img, n_occ)] = np.nan
            RSM_models['matrices']['occluder_position'] \
                [c, np.arange(c % n_occ, n_img, n_occ)] = 1
            RSM_models['matrices']['exemplar_bothocc'] \
                [c, c] = 1


    FEAT_designs = {
        'base': {  # common to all FEAT analyses
            # misc
            'version': 6.00,  # FEAT version
            'inmelodic': 0,  # Are we in MELODIC?'
            'relative_yn': 0,  # Use relative filenames'
            'help_yn': 1,  # Balloon help
            'featwatcher_yn': 0,  # Run Featwatcher
            'brain_thresh': 10,  # Brain/background threshold, %
            'critical_z': 5.3,
            'noise': 0.66,  # Noise level
            'noisear': 0.34,  # Noise AR(1):
            'tagfirst': 1,
            'reginitial_highres_yn': 0,
            'init_init_highres': '\"\"',
            'overwrite_yn': 0,
        },
        'runwise': {
            # main
            'level': 1,  # First or higher-level analysis
            'analysis': 7,  # Which stages to run (7=full first level)
            # data
            'ndelete': 0,  # Delete volumes
            'dwell': 0.52,  # dwell time
            'te': 25, # echo time (ms):
            'totalVoxels': 76744192,
            # prestats
            'alternateReference_yn': 1,
            'regunwarp_yn': 0,  # B0 unwarping
            'unwarp_dir': 'y',  # B0 unwarp direction
            'filtering_yn': 1,  # Carry out pre-stats processing?
            'bet_yn': 0,  # brain extraction
            'smooth': 0,  # spatial smoothing
            'norm_yn': 0,  # Intensity normalization
            'perfsub_yn': 0,  # Perfusion subtraction
            'temphp_yn': 1,  # Highpass temporal filtering
            'paradigm_hp': 100,  # Highpass temporal filtering cutoff
            'templp_yn': 0,  # Lowpass temporal filtering
            'tagfirst': 1,  # Perfusion tag/control order
            'melodic_yn': 0,  # MELODIC ICA data exploration
            # registration
            'reginitial_highres_yn': 0,  # Reg to initial structural
            'reghighres_yn': 0,
            'regstandard_yn': 0,
            'regstandard': '"/usr/local/fsl/data/standard/MNI152_T1_2mm_brain"',
            'regstandard_search': 90,
            'regstandard_dof': 6,
            'regstandard_nonlinear_yn': 0,
            'regstandard_nonlinear_warpres': 10,
            # stats
            'stats_yn': 1,  # Carry out main stats?
            'mixed_yn': 2,  # Mixed effects/OLS
            'prewhiten_yn': 1,  # Carry out prewhitening?'
            'evs_vox': 0,  # Number of EVs
            'con_mode_old': 'orig',  # Contrast & F-tests mode
            'con_mode': 'orig',  # Contrast & F-tests mode
            'nftests_orig': 0,  # Number of F-tests
            'nftests_real': 0,  # Number of F-tests
            # post-stats
            'poststats_yn': 0,  # Carry out post-stats steps?
            'thresh': 0,
            'conmask_zerothresh_yn': 0,  # Contrast masking
            'conmask1_1': 0,  # Do contrast masking at all?
        },
        'modeling': {'exp1': {}, 'exp2': {}},
        'subjectwise': {
            # main
            'level': 2,
            'analysis': 2,
            # data
            'inputtype': 1,
            'filtering_yn': 0,
            'sscleanup_yn': 0,
            'totalVoxels': 95785984,  # MNI 2mm: 95785984
            'tr': 2.0,  # higher-level needs this for some reason
            'ndelete': 0,  # higher-level needs this for some reason
            # stats
            'mixed_yn': 3,
            'randomisePermutations': 5000,
            'evs_orig': 1,
            'evs_real': 1,
            'evs_vox': 0,
            'ncon_orig': 0,
            'ncon_real': 1,
            'nftests_orig': 0,  # Number of F-tests
            'nftests_real': 0,  # Number of F-tests
            'evtitle1': '"mean"',
            'shape1': 2,
            'convolve1': 0,
            'convolve_phase1': 0,
            'tempfilt_yn1': 0,
            'deriv_yn1': 0,
            'con_mode_old': 'real',
            'con_mode': 'real',
            'custom1': '"dummy"',
            'ortho1.0': 0,
            'ortho1.1': 0,
            'conpic_real.1': 1,
            'conname_real.1': '"mean"',
            'con_real1.1': 1,
            # post-stats
            'poststats_yn': 0,
            'threshmask': '\"\"',
            'thresh': 3,
            'prob_thresh': 0.05,
            'z_thresh': 3.1,
            'zdisplay': 0,
            'zmin': 2,
            'zmax': 8,
            'rendertype': 1,
            'bgimage': 1,
            'tsplot_yn': 1,
            'conmask_zerothresh_yn': 0,  # Contrast masking
            'conmask1_1': 0,  # Do contrast masking at all?
            # misc
            'alternative_mask': '\"\"',
            'init_highres': '\"\"',
            'init_standard': '\"\"',
        },
    }
    FEAT_designs['groupwise'] = FEAT_designs['subjectwise'].copy()
    FEAT_designs['groupwise']['mixed_yn'] = 2
    FEAT_designs['groupwise']['poststats_yn'] = 1
    FEAT_designs['groupwise']['robust_yn'] = 1


    FEAT_contrasts = {'exp1': {'occlusion': {
        **{x + 1: [cond, np.eye(24, dtype=int)[x]] for x, cond in
           enumerate(cond_labels['exp1'])},
        **{25: ['all-conds', np.repeat(1, n_img)],
           26: ['upper-gt-lower', np.tile([0, -1, 1], n_exem)],
           27: ['unocc-gt-occ', np.tile([2, -1, -1], n_exem)],
           28: ['animate-gt-inanimate-unocc', np.concatenate([
               np.tile([1, 0, 0], 4), np.tile([-1, 0, 0], 4)])]}
        }}}

    # add design lines for each task
    for task, contrasts in FEAT_contrasts['exp1'].items():

        # scanning parameters
        design_items = {
            'tr': scan_params['exp1'][task]['TR'],
            'npts': scan_params['exp1'][task]['dynamics']
        }

        # explanatory variables
        n_conds = np.prod([len(conds) for conds in scan_params[
            'exp1'][task]['conditions'].values()])
        design_items['evs_orig'] = n_conds
        design_items['evs_real'] = n_conds*2
        conds = cond_labels['exp1']
        for c, cond in enumerate(conds):
            design_items[f'evtitle{c+1}'] = f'"{cond}"'
            design_items[f'shape{c+1}'] =  3
            design_items[f'convolve{c+1}'] = 3
            design_items[f'convolve_phase{c+1}'] = 0
            design_items[f'tempfilt_yn{c+1}'] = 1
            design_items[f'deriv_yn{c+1}'] = 1
            for c2 in range(n_conds+1):
                design_items[f'ortho{c+1}.{c2}'] = 0

        # contrasts
        n_contrasts = len(contrasts)
        design_items['ncon_orig'] = n_contrasts
        design_items['ncon_real'] = n_contrasts
        for co, cont in enumerate(contrasts):
            contrast, weights = contrasts[cont]
            design_items[f'conpic_real.{cont}'] = 1
            design_items[f'conpic_orig.{cont}'] = 1
            design_items[f'conname_real.{cont}'] = f'"{contrast}"'
            design_items[f'conname_orig.{cont}'] = f'"{contrast}"'
            for cond in range(n_conds):
                weight = weights[cond]
                design_items[f'con_real{cont}.{cond*2 + 1}'] = weight
                design_items[f'con_real{cont}.{cond*2 + 2}'] = 0
                design_items[f'con_orig{cont}.{cond + 1}'] = weight
            for co2 in range(n_contrasts):
                if co2 != co:
                    design_items[f'conmask{co+1}_{co2+1}'] = 0

        # add design lines to FEAT_design
        FEAT_designs['modeling']['exp1'][task] = design_items.copy()

    # do for exp2
    FEAT_contrasts['exp2'] = {
        'occlusionAttnOn': FEAT_contrasts['exp1']['occlusion'],
        'occlusionAttnOff': FEAT_contrasts['exp1']['occlusion']}
    FEAT_designs['modeling']['exp2']['occlusionAttnOn'] = FEAT_designs[
        'modeling']['exp1']['occlusion']
    FEAT_designs['modeling']['exp2']['occlusionAttnOff'] = FEAT_designs[
        'modeling']['exp1']['occlusion']







