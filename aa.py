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

sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings

x = "\\images\\bus.jpg"

a = 'txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1))

print(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1))

