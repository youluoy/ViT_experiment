#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import modules.ViT_alta as Vit_alta


num_classes = 100
batch_size, channel, height, width = 10, 3, 32, 32
num_patch_row = 4
net = Vit_alta.Vit_alta(in_channels=channel, num_classes=num_classes, num_patch_row=num_patch_row)

