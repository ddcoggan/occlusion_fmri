# /usr/bin/python
# Created by David Coggan on 2023 09 28

import os
import os.path as op
from .config import CFG, PROJ_DIR

def check_segmentation():
    """
    Check segmentation quality and fix errors
    """
    exp = op.basename(os.getcwd())
    subjects = CFG.subjects[exp]
    for subject in subjects:
        while True:
            #fs_sub_dir = f'{os.environ["SUBJECTS_DIR"]}/sub-{subject}'
            fs_sub_dir = f' /home/tonglab/david/subjects/freesurfer/sub-{subject}'
            cmd = (f'freeview '
                   f'-v {fs_sub_dir}/mri/T1.mgz '
                   f'{fs_sub_dir}/mri/brainmask.mgz:visible=0 '
                   f'{fs_sub_dir}/mri/wm.mgz:colormap=heat:opacity=0.40:heatscale=100,250 '
                   f'-f {fs_sub_dir}/surf/lh.smoothwm:edgecolor=yellow '
                   f'{fs_sub_dir}/surf/rh.smoothwm:edgecolor=yellow '
                   f'{fs_sub_dir}/surf/lh.pial.T1:edgecolor=blue '
                   f'{fs_sub_dir}/surf/rh.pial.T1:edgecolor=blue')
            print(cmd)
            os.system(cmd)
            # fix any errors to wm.mgz then close freesurfer
            reconstruct = input('Rerun surface reconstruction? (y/n)')
            if reconstruct == 'y':
                os.system(f'recon-all -autorecon2-wm -autorecon3 -subjid sub-{subject}')
            else:
                break

if __name__ == "__main__":
    os.chdir(f'{PROJ_DIR}/data/in_vivo/fMRI/exp1')
    check_segmentation()