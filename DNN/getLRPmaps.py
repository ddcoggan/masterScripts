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
import sys
sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
import zoo
from LRP import lrp
from matplotlib.colors import ListedColormap


def getLRPmaps(modelName='alexnet',dataset='imagenet1000',paramsPath=None,imagePath=None,outPath=None,classID=None,returnMap=False):

    if outPath == None:
        saveFigure = False
    else:
        saveFigure = True
        # set up outdir
        outDir = os.path.dirname(outPath)
        os.makedirs(outDir, exist_ok=True)

    # open image and apply transform
    image = Image.open(imagePath)
    if len(image.size) < 3:
        image = image.convert('RGB')
    # create image preprocessor
    transformSequence = [
            transforms.Resize(224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(), # transform PIL image to Tensor.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] # Only normalize on Tensor datatype.
    transform = transforms.Compose(transformSequence)
    image = transform(image)

    if paramsPath == None:
        pretrained = True
    else:
        pretrained = False

    # get model
    model = getattr(zoo, modelName)()

    # make sure last layer is adapted to the number of classes
    trainPath = os.path.join('/home/dave/Datasets', dataset, 'train')
    nClasses = len(glob.glob(os.path.join(trainPath, '*')))
    try:
        inFeatures = model.classifier[-1].in_features
        model.classifier.add_module(name=str(len(model.classifier) - 1), module=nn.Linear(inFeatures, nClasses, True))
    except:
        inFeatures = model.decoder[-2].in_features
        model.decoder.add_module(name='linear', module=nn.Linear(inFeatures, nClasses, True))

    # Load weights from file
    if not pretrained:

        # load network weights
        resume_weight = torch.load(paramsPath)
        if modelName in ['inception_v3', 'resnet152']:
            newDict = {}
            for key in resume_weight['model']:
                newDict[f'module.{key}'] = resume_weight['model'][key]
            model.load_state_dict(newDict)
        else:
            try:
                model.load_state_dict(resume_weight['model'])
            except: # remove 'module.' from start of each key
                newDict = {}
                for key in resume_weight['model']:
                    newDict[f'{key[7:]}'] = resume_weight['model'][key]
                model.load_state_dict(newDict)

    model.cuda()
    model.eval()

    inputs = image.cuda()
    inputs = inputs.unsqueeze_(dim=0)

    lrp_model = lrp.LRP(model)

    R_targets = torch.zeros_like(model(inputs))
    R_targets[0,classID] = 1.
    relevance = lrp_model.relprop(inputs, R_targets)
    relevance = np.nan_to_num(relevance, nan=0)
    relevance = np.sum(relevance, axis=1)  # sum
    relevance[relevance < 0] = 0  # only positive
    relevanceScaled = relevance / np.max(relevance, axis=(1, 2))[:, np.newaxis, np.newaxis] * 255  # probability multiplied by 255
    relevanceScaled = relevanceScaled.squeeze()
    b = 10 * ((np.abs(relevanceScaled) ** 3.0).mean() ** (1.0 / 3))
    #my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    #my_cmap[:, 0:3] *= 0.85
    #my_cmap = ListedColormap(my_cmap)
    if saveFigure:
        plt.figure(figsize=(3.5, 3.5))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.imshow(relevanceScaled, cmap='rainbow', vmin=0, vmax=b, interpolation='nearest')
        plt.savefig(outPath)
        plt.show()
    if returnMap:
        return relevance

