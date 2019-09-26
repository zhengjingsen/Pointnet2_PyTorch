from easydict import EasyDict as edict
import math

cfg = edict()
cfg.DATA_PATH = '/home/sasi/datavault/tools/vault1/model_data/semanticKitti/dataset/sequences'

# Training params
cfg.DEBUG = True
cfg.BATCH_SIZE = 8

cfg.MAX_GRAD_NORM = 1.0
cfg.MOVING_AVERAGE_DECAY = 0.999
cfg.BATCHNORM_DECAY = 0.999
cfg.WEIGHT_DECAY = 0.001
cfg.MOVING_AVERAGE_FOR_VAL = True

# Data trimming params
cfg.MIN_X = -1.
cfg.MAX_X = 1.
cfg.MIN_Y = -1.
cfg.MAX_Y = 1.

# Data range to be considered to normalize b/w [-1, 1] 
# eg: default [-75, 75] is normalized to [-1, 1]
cfg.MIN_RANGE_X = -75.
cfg.MAX_RANGE_X = 75.
cfg.MIN_RANGE_Y = -75.
cfg.MAX_RANGE_Y = 75.
cfg.MIN_RANGE_Z = -5.
cfg.MAX_RANGE_Z = 5.

# Grid size of each voxel
# cfg.GRID_SIZE = 1.
cfg.GRID_SIZE = 0.01 # Input is normalized to lie in [-1, 1]

# Optimization params
cfg.ITER_NUM = 300
cfg.SUB_ITER_NUM = 1000
cfg.MOMENTUM = 0.9
cfg.LEARNING_RATE = 0.001

# lr schedule
# changed to cosine with warmup

# Data Augmentation
cfg.RAND_INTENSITY_VARIATION = 0.05 # if intensity lies in [0,1]
cfg.SCALE_PROB = 0.5
cfg.MAX_SCALE_CHANGE = 0.2
cfg.MAX_ROTATION = 180
cfg.MAX_SHIFT_X = 25
cfg.MAX_SHIFT_Y = 25
cfg.MAX_Z_CHANGE = 2

# Loading multiple frames
cfg.SEQ_NUM = 1


def get_x_num(cfg):
    return int(math.floor(
        (cfg.MAX_X - cfg.MIN_X) / cfg.GRID_SIZE)) # + 1 --> why


def get_y_num(cfg):
    return int(math.floor(
        (cfg.MAX_Y - cfg.MIN_Y) / cfg.GRID_SIZE)) # + 2   # add 2 here to seperate different batches -->what's meaning?
