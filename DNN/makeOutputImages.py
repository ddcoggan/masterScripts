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

def makeOutputImages(activationFile, imageName ='sample'):
    activation = pickle.load(open(activationFile, 'rb'))
    for l in range(len(activation.keys())):
        a = activation[l]
        if len(a.shape) > 3:
            a = F.einsum('akli->kali', a)
            a = torch.nn.functional.interpolate(input=a, size=(224,224), align_corners=True, mode='bilinear')
        outDir = f'{os.path.dirname(activationFile)}/featureMaps'
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        filename = f'{outDir}/layer{l:02d}.png'
        save_image(tensor=a, fp=filename, normalize=True, range=(0, 1))
