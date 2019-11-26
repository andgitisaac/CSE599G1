import os
import random

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import *
# from utils.fuel import ImageDataset, train_transform
from utils.helper import weights_init
from network.generator import Generator
from network.discriminator import Discriminator

from tensorboardX import SummaryWriter

writer = SummaryWriter(LOG_DIR)

cudnn.benchmark = True
if USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)

# data_loader = DataLoader(dataset=ImageDataset(TRAIN_DIR, train_transform(size=IMG_SIZE)),
#                             batch_size=BATCH_SIZE,
#                             shuffle=True
#                         )

dataset = dset.ImageFolder(root=TRAIN_DIR,
                            transform=transforms.Compose([
                                transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
data_loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=BATCH_SIZE,
                                        shuffle=True, 
                                        num_workers=4)


generator = Generator(ZDIM, NGF, NGPU)
generator.apply(weights_init)
generator.to(device)
print(generator)

discriminator = Discriminator(NDF, NGPU)
discriminator.apply(weights_init)
discriminator.to(device)
print(discriminator)


criterion = nn.BCELoss()

fixed_z = torch.FloatTensor(BATCH_SIZE, ZDIM, 1, 1).normal_(0, 1).to(device)

optimizerD = Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

steps_per_epoch = len(data_loader)

try:    
    for epoch in range(EPOCHS):
        for step, (real_image, _) in enumerate(tqdm(data_loader)):
            steps = epoch * steps_per_epoch + step

            if real_image.size()[0] != BATCH_SIZE:
                continue

            # Soft labels
            real_label = torch.FloatTensor(BATCH_SIZE, 1).uniform_(0.7, 1.0).to(device)
            fake_label = torch.FloatTensor(BATCH_SIZE, 1).uniform_(0.0, 0.3).to(device)

            ### Update Discriminator ### 

            gt_real, gt_fake = real_label, fake_label
            if steps != 0 and steps % 2 == 0:
                if random.random() < FLIP:
                    gt_real, gt_fake = gt_fake, gt_real

            # Train with real
            discriminator.zero_grad()
            real_image = real_image.to(device)
            
            pred = discriminator(real_image)
            loss_D_real = criterion(pred, gt_real)

            # train with fake
            z = torch.FloatTensor(BATCH_SIZE, ZDIM, 1, 1).normal_(0, 1).to(device)
            fake_image = generator(z)
            pred = discriminator(fake_image.detach())
            loss_D_fake = criterion(pred, gt_fake)

            loss_D = (loss_D_real + loss_D_fake)
            loss_D.backward()
        
            optimizerD.step()
            

            ### Update Generator ### 

            generator.zero_grad()
            pred = discriminator(fake_image)
            loss_G = criterion(pred, real_label)
            loss_G.backward()
            optimizerG.step()

            if steps != 0 and steps % LOG_STEP == 0:
                # print('Epoch: {:03d}, Step: {:04d} => Loss_D: {:.4f}, Loss_G: {:.4f}'
                #         .format(epoch, step, loss_D.item(), loss_G.item())
                #     )

                with torch.no_grad():
                    fake_image = generator(fixed_z)
                vutils.save_image(fake_image.data,
                        '{}/samples_epoch_{:03d}.png'.format(SAMPLE_DIR, epoch),
                        nrow=8,
                        normalize=True
                )

                writer.add_scalar('Loss_D', loss_D.item(), steps)
                writer.add_scalar('Loss_G', loss_G.item(), steps)
                writer.add_image('fake_image', vutils.make_grid(fake_image.data, nrow=8, normalize=True), steps)
        

        torch.save(generator.state_dict(), '{}/G_epoch_{:3d}.pth'.format(MODEL_DIR, epoch))
        torch.save(discriminator.state_dict(), '{}/D_epoch_{:3d}.pth'.format(MODEL_DIR, epoch))

except KeyboardInterrupt as ke:
    print('Interrupted')
except:
    import traceback
    traceback.print_exc()
finally:
    pass
