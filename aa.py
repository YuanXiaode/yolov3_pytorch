import torch
import torchvision.models as models
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import os
import cv2 as cv

a = torch.arange(12).reshape((3,2,2))
# print(a)

index = torch.tensor([1,2])
index2 = torch.tensor([1,0])

print(a[index,index2].shape)
print(a[index,index2])