import numpy as np
import torch
import torchvision
import multiprocessing as mp
from pprint import pprint
from PIL import Image
from pathlib import Path
import re
import platform
import pkg_resources as pkg
import os
from itertools import repeat
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


iou = torch.from_numpy(np.arange(12).reshape((4,3)))
print(iou)
x = torch.where(iou > 4)
matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
m0, m1, _ = matches.transpose().astype(np.int16)
matches = matches[matches[:, 2].argsort()[::-1]]  # 按iou 大->小

array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
fig = plt.figure(figsize=(12, 9), tight_layout=True)
sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
labels = (0 < len(names) < 99) and len(names)==self.nc  # apply names to ticklabels
sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
    xticklabels=names + ['background FP'] if labels else "auto",
    yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
fig.axes[0].set_xlabel('True')
fig.axes[0].set_ylabel('Predicted')
fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)