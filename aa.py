
import torch
import torchvision.models as models
import thop
import numpy as np


a = np.ones((3,10,10)) * 255
a = a.astype(np.int8)
# print(a)
print(a.astype(np.float32))