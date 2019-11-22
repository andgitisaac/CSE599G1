import os

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from config import *
from utils.fuel import ImageDataset, train_transform
from models.generator import Generator
from models.discriminator import Discriminator

cudnn.benchmark = True
device = torch.device('cuda')

generator = Generator()
generator.train()
generator.to(device)

discriminator = Discriminator()
discriminator.train()
discriminator.to(device)

train_loader = DataLoader(dataset=ImageDataset(TRAIN_DIR, train_transform(size=SIZE)),
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=WORKERS)

for epoch in range(EPOCHS):
    
    for step, (image) in enumerate(train_loader):
        print(image.size())