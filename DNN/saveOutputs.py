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
from modelLayerLabels import modelLayerLabels
import zoo
from zoo.prednet import *
sys.path.append('/mnt/HDD12TB/masterScripts/DNN/zoo/CORnet_master')
import cornet

def saveOutputs(modelName='alexnet',dataset='imagenet1000',paramsPath=None,imagePath=None,outPath=None,transform=None, cycles=3):

    def get_activation(idx):
        def hook(model, input, output):
            activation[idx] = output.detach()
        return hook

    # set up outdir
    outDir = os.path.dirname(outPath)
    os.makedirs(outDir, exist_ok=True)

    # paths to dataset
    datasetPath = f'/home/dave/Datasets/{dataset}'
    if not os.path.isdir(datasetPath):
        datasetPath = f'/media/dave/HotProjects/Datasets/{dataset}'
    trainPath = os.path.join(datasetPath, 'train')

    # open image and apply transform and save out
    image = Image.open(imagePath)
    if len(image.size) < 3:
        image = image.convert('RGB')
    # create image preprocessor
    if transform == None:
        transformSequence = [
                transforms.Resize(224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(), # transform PIL image to Tensor.
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] # Only normalize on Tensor datatype.
        transform = transforms.Compose(transformSequence)
    image = transform(image)
    input = image.cuda()
    input = input.unsqueeze_(dim=0)

    if paramsPath == None:
        pretrained = True
    else:
        pretrained = False

    # get model
    if modelName.startswith('cornet'):
        model = getattr(cornet, modelName)
    elif modelName.startswith('PredNet'):
        model = PredNetImageNet_detailedOutput(cls=cycles) # need detailed output version for saving outputs
        convparams = [p for p in model.baseconv.parameters()] + \
                     [p for p in model.FFconv.parameters()] + \
                     [p for p in model.FBconv.parameters()] + \
                     [p for p in model.linear.parameters()] + \
                     [p for p in model.GN.parameters()]
        rateparams = [p for p in model.a0.parameters()] + \
                     [p for p in model.b0.parameters()]
    else:
        model = getattr(zoo, modelName)
    if modelName not in ['PredNetImageNet']:
        model = model(pretrained=pretrained)

    # make sure last layer is adapted to the number of classes
    nClasses = len(glob.glob(os.path.join(trainPath, '*')))
    if nClasses != 1000:
        if modelName in ['alexnet', 'vgg19']:
            inFeatures = model.classifier[-1].in_features
            model.classifier.add_module(name=str(len(model.classifier) - 1), module=nn.Linear(inFeatures, nClasses, True))
        elif modelName in ['cornet_s']:
            inFeatures = model.module.decoder[-2].in_features
            model.module.decoder.add_module(name='linear', module=nn.Linear(inFeatures, nClasses, True))
        elif modelName in ['inception_v3'] or modelName.startswith('resnet'):
            inFeatures = model.fc.in_features
            model.add_module(name='fc', module=nn.Linear(inFeatures, nClasses, True))
            model.aux_logits = False
        elif modelName.startswith('PredNet'):
            inFeatures = model.linear.in_features
            model.add_module(name='linear', module=nn.Linear(inFeatures, nClasses, True))

    # Load weights from file
    if not pretrained:

        # load network weights
        resume_weight = torch.load(paramsPath)
        try:  # try to load model as is
            model.load_state_dict(resume_weight['model'])
        except:
            try:  # remove 'module.' from start of each key
                newDict = {}
                for key in resume_weight['model']:
                    newDict[f'{key[7:]}'] = resume_weight['model'][key]
                model.load_state_dict(newDict)
            except:  # adding 'module.' prefix to keys
                newDict = {}
                for key in resume_weight['model']:
                    newDict[f'module.{key}'] = resume_weight['model'][key]
                model.load_state_dict(newDict)

    activation = {'input': image}

    if modelName in ['alexnet','vgg19']:
        layers = modelLayerLabels[modelName]
        # set up activations dict
        # cycle through model registering forward hook
        layerCounter = 0
        for l, layer in enumerate(model.features):
            model.features[l].register_forward_hook(get_activation(layers[layerCounter]))
            layerCounter += 1
        model.avgpool.register_forward_hook(get_activation(layers[layerCounter]))
        layerCounter += 1
        for l, layer in enumerate(model.classifier):
            model.classifier[l].register_forward_hook(get_activation(layers[layerCounter]))
            layerCounter += 1
        model.cuda()
        model.eval()
        output = model(input)

    elif modelName.startswith('PredNet'):

        model.cuda()
        model.eval()
        output, (x_ff_Ctr, x_fb_Ctr, x_pred_Ctr, x_err_Ctr, x_ff_beforeGN_Ctr, x_fb_beforeGN_Ctr) = model(input)
        activation['feedforward'] = x_ff_Ctr
        activation['feedback'] = x_fb_Ctr
        activation['prediction'] = x_pred_Ctr
        activation['error'] = x_err_Ctr
        activation['feedforward_preGroupNorm'] = x_ff_beforeGN_Ctr
        activation['feedback_preGroupNorm'] = x_fb_beforeGN_Ctr
        
    elif modelName in ['cornet_s']:
        
        model.module.V1.register_forward_hook(get_activation('V1'))
        model.module.V2.register_forward_hook(get_activation('V2'))
        model.module.V4.register_forward_hook(get_activation('V4'))
        model.module.IT.register_forward_hook(get_activation('IT'))
        model.module.decoder.register_forward_hook(get_activation('decoder'))
        
        model.cuda()
        model.eval()
        output = model(input)
        
    elif modelName.startswith('resnet'):

        model.relu.register_forward_hook(get_activation('relu'))
        model.layer1.register_forward_hook(get_activation('layer1'))
        model.layer2.register_forward_hook(get_activation('layer2'))
        model.layer3.register_forward_hook(get_activation('layer3'))
        model.layer4.register_forward_hook(get_activation('layer4'))
        model.fc.register_forward_hook(get_activation('fc'))

        model.cuda()
        model.eval()
        output = model(input)
        
    elif modelName in ['inception_v3']:

        model.Conv2d_1a_3x3.register_forward_hook(get_activation('Conv2d_1a_3x3'))
        model.Conv2d_4a_3x3.register_forward_hook(get_activation('Conv2d_4a_3x3'))
        model.Mixed_5b.register_forward_hook(get_activation('Mixed_5b'))
        model.Mixed_5c.register_forward_hook(get_activation('Mixed_5c'))
        model.Mixed_5d.register_forward_hook(get_activation('Mixed_5d'))
        model.Mixed_6a.register_forward_hook(get_activation('Mixed_6a'))
        model.Mixed_6b.register_forward_hook(get_activation('Mixed_6b'))
        model.Mixed_6c.register_forward_hook(get_activation('Mixed_6c'))
        model.Mixed_6d.register_forward_hook(get_activation('Mixed_6d'))
        model.Mixed_6e.register_forward_hook(get_activation('Mixed_6e'))
        model.Mixed_7a.register_forward_hook(get_activation('Mixed_7a'))
        model.Mixed_7b.register_forward_hook(get_activation('Mixed_7b'))
        model.Mixed_7c.register_forward_hook(get_activation('Mixed_7c'))
        model.fc.register_forward_hook(get_activation('fc'))

        model.cuda()
        model.eval()
        output = model(input)

    with open(outPath, 'wb') as f:
        pickle.dump(activation, f, pickle.HIGHEST_PROTOCOL)
    f.close()



