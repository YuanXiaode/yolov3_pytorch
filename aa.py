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

dh = 0

img_formats = ["aa","bb"]
print(f'Supported formats are:images: {img_formats}')