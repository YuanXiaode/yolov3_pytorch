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
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a = {"Age":12,"name":"Tom"}
b = {"Age":14,"name":"Tom2"}

c = [a,b]

for k ,v in c.it