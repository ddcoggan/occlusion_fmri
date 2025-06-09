# /usr/bin/python
from .config import CFG, PROJ_DIR, TABCOLS
from .FEAT import FEAT_runwise, FEAT_subjectwise, FEAT_groupwise
from .make_ROIs import make_ROIs
from .get_wang_atlas import get_wang_atlas
from .plot_utils import make_legend, export_legend, custom_defaults
from .preprocess import preprocess
from .RSA import get_responses, RSA_dataset, line_plot, clustered_barplot, \
    compare_regions, do_RSA, RSA
from .seconds_to_text import seconds_to_text
from .initialise_BIDS import initialise_BIDS
from .make_events import make_events
from .apply_topup import apply_topup
from .check_segmentation import check_segmentation
from .registration import registration
from .final_plots import main as make_final_plots

