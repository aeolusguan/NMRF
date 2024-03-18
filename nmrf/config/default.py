from .config import CfgNode as CN

# NOTE: given the new config system
# we will stop adding new functionalities to default CfgNode

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# ------------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# ------------------------------------------------------------------------------

_CN = CN()

# The version number, to upgrade from old config to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_CN.VERSION = 2

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_CN.BACKBONE = CN()
_CN.BACKBONE.NORM_FN = "instance"
_CN.BACKBONE.OUT_CHANNELS = 256

_CN.DPN = CN()
_CN.DPN.MAX_DISP = 320
_CN.DPN.COST_GROUP = 4
_CN.DPN.NUM_PROPOSALS = 4
_CN.DPN.CONTEXT_DIM = 64

_CN.NMP = CN()
_CN.NMP.PROP_EMBED_DIM = 128
_CN.NMP.INFER_EMBED_DIM = 128
_CN.NMP.MLP_RATIO = 4
_CN.NMP.SPLIT_SIZE = 1
_CN.NMP.WINDOW_SIZE = 6
_CN.NMP.REFINE_WINDOW_SIZE = 4
_CN.NMP.PROP_N_HEADS = 4
_CN.NMP.INFER_N_HEADS = 4
_CN.NMP.NUM_PROP_LAYERS = 5
_CN.NMP.NUM_INFER_LAYERS = 5
_CN.NMP.NUM_REFINE_LAYERS = 5
_CN.NMP.RETURN_INTERMEDIATE = True
_CN.NMP.ATTN_DROP = 0.0
_CN.NMP.PROJ_DROP = 0.0
_CN.NMP.DROP_PATH = 0.0
_CN.NMP.DROPOUT = 0.0
_CN.NMP.NORMALIZE_BEFORE = True
_CN.NMP.WITH_REFINEMENT = True

# ---------------------------------------------------------------------------- #
# Dataset and data augmentation
# ---------------------------------------------------------------------------- #
_CN.DATASETS = CN()
# List of dataset name for training
_CN.DATASETS.TRAIN = ["sceneflow"]
# List of dataset name for testing
_CN.DATASETS.TEST = ["things"]
# Image gamma
_CN.DATASETS.IMG_GAMMA = None
# Color saturation
_CN.DATASETS.SATURATION_RANGE = [0, 1.4]
# Flip the images horizontally or vertically, valid choice [False, 'h', 'v']
_CN.DATASETS.DO_FLIP = False
# Re-scale the image randomly
_CN.DATASETS.SPATIAL_SCALE = [-0.2, 0.4]
# Simulate imperfect rectification
_CN.DATASETS.YJITTER = False
# Image size for training
_CN.DATASETS.CROP_SIZE = [384, 768]

# ---------------------------------------------------------------------------- #
# Dataset and data augmentation
# ---------------------------------------------------------------------------- #
_CN.DATALOADER = CN()
# Number of data loading threads
_CN.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_CN.SOLVER = CN()

_CN.SOLVER.MAX_ITER = 300000

_CN.SOLVER.BASE_LR = 0.0005
_CN.SOLVER.BASE_LR_END = 0.0

_CN.SOLVER.WEIGHT_DECAY = 0.00001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_CN.SOLVER.WEIGHT_DECAY_NORM = 0.00001

# Save a checkpoint after every this number of iterations
_CN.SOLVER.CHECKPOINT_PERIOD = 100000
_CN.SOLVER.LATEST_CHECKPOINT_PERIOD = 1000

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration). If we use 16 GPUs
# and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
_CN.SOLVER.IMS_PER_BATCH = 8

# Gradient clipping
_CN.SOLVER.GRAD_CLIP = 1.0

_CN.SOLVER.LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4, 1.6, 2.0, 2.0]

# resume from pretrain model for finetuning or resuming from terminated training
_CN.SOLVER.RESUME = None
_CN.SOLVER.STRICT_RESUME = True
_CN.SOLVER.NO_RESUME_OPTIMIZER = False

_CN.SOLVER.AUX_LOSS = True

# Maximum disparity for training,
# ground truth disparities exceed than this threshold will be ignored for loss computation
_CN.SOLVER.MAX_DISP = 192

# Loss type used in cost aggregation and refinement
_CN.SOLVER.LOSS_TYPE = "L1"

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_CN.TEST = CN()
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_CN.TEST.EVAL_PERIOD = 20000
# Threshold for metric computation for testing
_CN.TEST.EVAL_THRESH = [['1.0', '3.0']]
# Maximum disparity for metric computation mask
_CN.TEST.EVAL_MAX_DISP = [192]
# Whether use only valid pixels in evaluation
_CN.TEST.EVAL_ONLY_VALID = [True]
# Whether evaluate disparity proposal
_CN.TEST.EVAL_PROP = [True]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_CN.SEED = 326
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same of similar sizes, benchmark is often helpful.
_CN.CUDNN_BENCHMARK = True

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from nmrf.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_CN.GLOBAL = CN()
_CN.GLOBAL.HACK = 1.0