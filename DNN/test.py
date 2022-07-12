import os
import glob
import sys
import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from accuracy import accuracy
from alterImages import occludeImages, addNoise, blurImages
import zoo
from zoo.prednet import *

sys.path.append('/mnt/HDD12TB/masterScripts/DNN/zoo/CORnet_master')
import cornet

def test(modelName='alexnet', datasetPath = '/home/dave/Datasets', batchSize =256, weightsPath=None,
         workers=6, occlusionMethod='crossBars', coverage=.4,  propOccluded=1.0, colour=(0,0,0), invert=False, cycles=3,
         blur=False, blurSigmas=None, blurWeights=None, noise=False, noiseLevels=None):

    ### IMAGE HANDLING ###

    # paths to dataset
    if not os.path.isdir(datasetPath):
        datasetPath = f'/home/dave/Datasets/{datasetPath}'
    valPath = os.path.join(datasetPath, 'val')

    # create image preprocessor (normalization happens later with occlusion)
    if modelName in ['inception_v3']:
        imageSizePre = 500
        imageSize = 299
    else:
        imageSizePre = 300
        imageSize = 224

    transformSequence = [
        transforms.Resize(imageSizePre),  # resize (smallest edge becomes this length)
        transforms.CenterCrop([imageSizePre, imageSizePre]),  # make square
        transforms.RandomRotation(10),  # rotate
        transforms.ToTensor()]  # transform PIL image to Tensor.
    transform = transforms.Compose(transformSequence)

    # create image loaders
    valData = ImageFolder(valPath, transform=transform)
    valLoader = DataLoader(valData, batch_size=batchSize, shuffle=True, num_workers=workers)

    ### MODEL HANDLING ###

    if weightsPath == None:
        pretrained = True
    else:
        pretrained = False

    # get model
    if modelName.startswith('cornet'):
        model = getattr(cornet, modelName)
    elif modelName.startswith('PredNet'):
        model = PredNetImageNet(cls=cycles)
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
    nClasses = len(glob.glob(os.path.join(valPath, '*')))
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

    # get GPU info
    #print(f'Using {torch.cuda.device_count()} GPUs')
    if torch.cuda.device_count() > 1 and modelName not in ['cornet_s', 'PredNetImageNet']: # cornet does not currently work with multiple GPUs
        model = nn.DataParallel(model)
    model.cuda()  # put model on cuda

    # set starting epoch
    starting_epoch = 0
    if weightsPath is not None:

        # load network weights
        #print("Loading weights at %s" %(weightsPath))
        resume_weight = torch.load(weightsPath)
        try: # try to load model as is
            model.load_state_dict(resume_weight['model'])
        except:
            try: # remove 'module.' from start of each key
                newDict = {}
                for key in resume_weight['model']:
                    newDict[f'{key[7:]}'] = resume_weight['model'][key]
                model.load_state_dict(newDict)
            except: # adding 'module.' prefix to keys
                newDict = {}
                for key in resume_weight['model']:
                    newDict[f'module.{key}'] = resume_weight['model'][key]
                model.load_state_dict(newDict)

    model.eval()
    loader = valLoader
    logString = 'Evaluating'

    # track accuracy and loss across entire epoch
    acc1 = 0.0
    acc5 = 0.0

    for batch, (inputs, targets) in enumerate(loader):

        # alter images
        if blur == True:
            inputs = blurImages(inputs, blurSigmas, blurWeights)
        if noise == True:
            inputs = addNoise(inputs, noiseLevels)
        if occlusionMethod != 'unoccluded':
            inputs = occludeImages(inputs, occlusionMethod, coverage, colour, invert, propOccluded)
        inputs = transforms.RandomCrop(imageSize)(inputs)
        inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs)

        inputs, targets = inputs.cuda(), targets.cuda()
        output = model(inputs)

        '''
        # save some input images
        if batch == 0:
            sampleInputDir = f'{os.path.dirname(os.path.dirname(weightsPath))}/sampleTestInputs'
            os.makedirs(sampleInputDir, exist_ok=True)
            for i in range(inputs.size(0)):  # (min(inputs.size(0), 32)):
                image = inputs[i, :, :, :]
                imageArray = np.array(torch.Tensor.cpu(image.permute(1, 2, 0)))
                imagePos = imageArray - imageArray.min()
                imageScaled = imagePos * (255.0 / imagePos.max())
                imagePIL = Image.fromarray(imageScaled.astype(np.uint8))
                imagePIL.save(f'{sampleInputDir}/{i:04}.png')
        '''

        # get accuracy and loss for this batch
        acc1batch = accuracy(output, targets, (1,))[0].detach().cpu().item()
        acc5batch = accuracy(output, targets, (5,))[0].detach().cpu().item()

        # compute mean accuracy and loss for this epoch so far
        if batch == 0:
            acc1 = acc1batch
            acc5 = acc5batch
        elif batch > 0:
            acc1 = ((acc1 * batch) + acc1batch) / (batch + 1)
            acc5 = ((acc5 * batch) + acc5batch) / (batch + 1)


        #print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}. {logString}...\t\tBatch: {batch}, Top 1 accuracy (batch): {acc1batch}, Top 1 accuracy (total): {acc1}')

    if type(acc1) is not float:
        Exception('acc1 is of length > 1')
        
    return acc1, acc5

    print('Done')
