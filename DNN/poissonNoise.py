import glob
import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.functional as F
from torchvision.models import alexnet, vgg19
import torchvision.transforms as transforms
from PIL import Image
import json
import pickle


class poissonNoise(nn.Module):
    def __init__(self):
        super(poissonNoise, self).__init__()

    def forward(self, input):
        return torch.poisson(input * 100 + 2) / 100
