# encoding: utf-8
"""
      author: YuanXiaode
      file: aa.py
      time: 2021/6/24 9:47
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision

import yaml  # for torch hub

cfg = "models/yolov3.yaml"

with open(cfg) as f:
    ayaml = yaml.safe_load(f)  # model dict

print(ayaml)

