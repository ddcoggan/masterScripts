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



modelLayerLabels = {'alexnet': ['conv1', 'relu1', 'maxpool1',
		                 'conv2', 'relu2', 'maxpool2',
		                 'conv3', 'relu3',
		                 'conv4', 'relu4',
		                 'conv5', 'relu5', 'maxpool3',
		                 'avgpool',
		                 'dropout6', 'fc6', 'relu6',
		                 'dropout7', 'fc7', 'relu7',
		                 'fc8'],
                    'vgg19': ['conv01', 'relu01',
                              'conv02', 'relu02', 'maxpool02',
                              'conv03', 'relu03',
                              'conv04', 'relu04', 'maxpool04',
                              'conv05', 'relu05',
                              'conv06', 'relu06',
                              'conv07', 'relu07', 
                              'conv08', 'relu08', 'maxpool08',
                              'conv09', 'relu09',
                              'conv10', 'relu10',
                              'conv11', 'relu11', 
                              'conv12', 'relu12', 'maxpool12',
                              'conv13', 'relu13',
                              'conv14', 'relu14',
                              'conv15', 'relu15',
                              'conv16', 'relu16', 'maxpool16',
                              'avgpool',
                              'fc17', 'relu17', 'dropout17',
                              'fc18', 'relu18', 'dropout18',
                              'fc19']}
