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


def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img = std*img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
