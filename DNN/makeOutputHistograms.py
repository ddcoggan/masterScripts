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


def makeOutputHistograms(activationFile, imageName ='sample'):
    activation = pickle.load(open(activationFile, 'rb'))
    for l in range(len(activation.keys())):
        a = activation[l].cpu()
        a = a.reshape(-1)
        n, bins, patches = plt.hist(x=a, bins=25, color='#0504aa',
                                    alpha=1, rwidth=0.85)
        plt.grid(axis='y', alpha=1)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'layer {l:02d}')
        outDir = f'{os.path.dirname(activationFile)}/histograms'
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        filename = f'{outDir}/layer{l:02d}.png'
        plt.savefig(filename)
        plt.close()
