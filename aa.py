import torch
import torchvision.models as models
import thop
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import os



print(os.environ['SLURM_PROCID'])