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

nc  = 5
map = 6
maps = np.zeros(nc) + map

print(maps)
