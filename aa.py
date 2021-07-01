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
import glob
import random

x = np.array([2,4,1])
y = np.array([2,4,1])
n = 9

a = np.digitize(x,[2,3,4,5,6])
print(a)

