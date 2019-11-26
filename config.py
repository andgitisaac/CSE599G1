import os

# Enviromnet Settings
USE_CUDA = True
TRAIN_DIR = 'data'
MODEL_DIR = 'model'
SAMPLE_DIR = 'sample'
LOG_DIR = 'log'

# Network Settings
IMG_SIZE = 64
ZDIM = 100
NGF = 32
NDF = 32


# Training Settings
SEED = 6666

EPOCHS = 50
BATCH_SIZE = 64
LR = 0.0005
LOG_STEP = 50
FLIP = 0.3

NGPU = 1

