import sys
import pickle
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader # Takes ImageFolder as a parameter
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision.datasets import ImageFolder
import numpy as np
import glob
import os
from PIL import Image
import datetime
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
if os.uname().nodename == 'finn':
    sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
elif os.uname().nodename == 'u110380':
    sys.path.append('/home/dave/masterScripts/DNN')
from alterImages import occludeImages, addNoise, blurImages
from accuracy import accuracy
import zoo
from zoo.prednet import *

if os.uname().nodename == 'finn':
    sys.path.append('/mnt/HDD12TB/masterScripts/DNN/CORnet_master')
elif os.uname().nodename == 'u110380':
    sys.path.append('/home/dave/masterScripts/DNN/CORnet_master')
import cornet

def train(modelName='alexnet', model=None, datasetPath='/home/dave/Datasets/imagenet16', pretrained=True, learningRate=.01,
          optimizerName='SGD', batchSize=256, nEpochs=25, restartFrom=None, workers=6, outDir=None, returnModel=False, nGPUs=-1, GPUids=None, skipZeroth=False, occlude=False,
          occlusionMethod=None, coverage=.4, propOccluded=1.0, colour=(0, 0, 0), invert=False, cycles=3, momentum=.9,
          weight_decay=1e-4, printOut={}, blur=False, blurSigmas=None, blurWeights=None, noise=False, noiseLevels=None,times=[2,2,4,2]):

    ### CONFIGURATION
    os.makedirs(outDir, exist_ok=True) # make output directory

    # record configuration
    config = {'model': modelName, 'datasetPath': datasetPath, 'pretrained': pretrained,
              'learningRate': learningRate, 'optimizer': optimizerName, 'batchSize': batchSize,
              'nEpochs': nEpochs, 'outDir': outDir, 'occlusionMethod': occlusionMethod,
              'coverage': coverage, 'colour': colour, 'invert': invert, 'cycles': cycles, 'times': times}
    configFile = os.path.join(outDir, 'config.pkl')
    pickle.dump(config, open(configFile, 'wb'))

    # record weights and log data
    modelSavePath = os.path.join(outDir, 'params')
    os.makedirs(modelSavePath, exist_ok=True)
    logFile = os.path.join(outDir, 'log.csv')

    # record plot data
    plotDir = os.path.join(outDir, 'plots')
    os.makedirs(plotDir, exist_ok=True)
    plotStatsFile = os.path.join(outDir, 'plotStats.pkl')


    ### IMAGE HANDLING ###

    # paths to dataset
    if not os.path.isdir(datasetPath):
        datasetPath = f'/home/dave/Datasets/{datasetPath}'
    trainPath = os.path.join(datasetPath, 'train')
    valPath = os.path.join(datasetPath, 'val')

    # create image preprocessor (normalization happens later with occlusion)
    if modelName in ['inception_v3']:
        imageSizePre = 500
        imageSize = 299
    else:
        imageSizePre = 300
        imageSize = 224

    transformSequence = [
            transforms.Resize(imageSizePre), # resize (smallest edge becomes this length)
            transforms.CenterCrop([imageSizePre, imageSizePre]),  # make square
            transforms.RandomRotation(10), # rotate
            transforms.ToTensor()] # transform PIL image to Tensor.
    transform = transforms.Compose(transformSequence)

    # create image loaders
    trainData = ImageFolder(trainPath, transform=transform)
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=workers)
    valData = ImageFolder(valPath, transform=transform)
    valLoader = DataLoader(valData, batch_size=batchSize, shuffle=True, num_workers=workers)

    ### MODEL HANDLING ###

    if model is None:

        # get model
        if modelName.endswith('varRec'):
            model = getattr(cornet, modelName)
            model = model(pretrained=False, times=times)
        elif modelName.startswith('cornet'):
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
        if modelName not in ['PredNetImageNet', 'cornet_s_varRec']:
            model = model(pretrained=pretrained)


        # make sure last layer is adapted to the number of classes
        nClasses = len(glob.glob(os.path.join(trainPath, '*')))
        if nClasses != 1000:
            if modelName in ['alexnet', 'vgg19']:
                inFeatures = model.classifier[-1].in_features
                model.classifier.add_module(name=str(len(model.classifier)-1), module=nn.Linear(inFeatures, nClasses, True))
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

    # show model architecture
    print(model)

    # set optimizer
    if modelName.startswith('PredNet'):
        if optimizerName == 'SGD':
            optimizer = torch.optim.SGD([{'params': convparams}, {'params': rateparams, 'weight_decay': 0}],
                                        lr=learningRate,
                                        momentum=momentum,
                                        weight_decay=weight_decay)
    else:
        if optimizerName == 'SGD':
            optimizer = optim.SGD(params=model.parameters(), lr=learningRate)
        elif optimizerName == 'Adam':
            optimizer = optim.Adam(params=model.parameters(), lr=learningRate)

    # get GPU info
    if nGPUs == -1:
        GPUids=None
        print(f'Using all available GPUs ({torch.cuda.device_count()})')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()  # put model on cuda
    elif nGPUs == 1:
        print(f'Using 1 GPU (#{GPUids})')
        model.cuda()

    printString = f'Started at {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} |'
    for p in printOut.keys():
        printString += f' {p}: {printOut[p]} |'
    print(printString)

    # set starting epoch
    starting_epoch = 0
    if skipZeroth:
        starting_epoch=1

    if restartFrom is not None:

        # load network weights
        print("Resuming from weights at %s" %(restartFrom))
        resume_weight = torch.load(restartFrom)
        try:
            model.load_state_dict(resume_weight['model'])
        except: # remove 'module.' from start of each key
            newDict = {}
            for key in resume_weight['model']:
                newDict[f'{key[7:]}'] = resume_weight['model'][key]
            model.load_state_dict(newDict)
        optimizer.load_state_dict(resume_weight['optimizer'])
        starting_epoch = int(restartFrom.split('/')[-1].split('.')[0]) + 1

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100)
    loss_fn = nn.CrossEntropyLoss().cuda()


    for epoch in range(starting_epoch, nEpochs+1):

        trainEvals = ['train', 'eval']

        if epoch == 0 or (epoch == 1 and skipZeroth):
            log = pd.DataFrame(
                columns=['epoch', 'batch', 'trainEval', 'acc1batch', 'acc5batch', 'lossBatch', 'cumAcc1epoch',
                         'cumAcc5epoch', 'cumLossEpoch'])
            plotStats = {'train': {'acc1': [], 'acc5': [], 'loss': []},
                         'eval': {'acc1': [], 'acc5': [], 'loss': []}}
        else:

            log = pd.read_csv(logFile)
            plotStats = pickle.load(open(plotStatsFile, 'rb'))

            # continue from eval if interrupted during eval
            if len(plotStats['train']['acc1']) > len(plotStats['eval']['loss']):
                trainEvals = ['eval']

        for trainEval in ['train','eval']:

            # train/eval specific settings
            if trainEval == 'train':

                # for zeroth epoch, just measure performance, no training
                if epoch == 0:
                    model.eval()
                else:
                    model.train()

                loader = trainLoader
                logString = 'Training'.ljust(10)

            elif trainEval == 'eval':

                model.eval()
                loader = valLoader
                logString = 'Evaluating'

            # track accuracy and loss across entire epoch
            acc1epoch = 0.0
            acc5epoch = 0.0
            lossEpoch = 0.0

            with tqdm(loader, unit="batch") as tepoch:

                for batch, (inputs, targets) in enumerate(tepoch):
                    tepoch.set_description(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {logString} | Epoch {epoch}/{nEpochs}')

                    if trainEval == 'train' and epoch > 0:
                        optimizer.zero_grad()
                        inputs, targets = Variable(inputs), Variable(targets)

                    # alter images
                    if blur:
                        inputs = blurImages(inputs, blurSigmas, blurWeights)
                    if noise:
                        inputs = addNoise(inputs, noiseLevels)
                    if occlude:
                        if occlusionMethod != 'unoccluded':
                            inputs = occludeImages(inputs, occlusionMethod, coverage, colour, invert, propOccluded)
                    inputs = transforms.RandomCrop(imageSize)(inputs)
                    inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs)

                    # save some input images
                    if epoch+batch == 0:
                        sampleInputDir = f'{outDir}/sampleInputs'
                        os.makedirs(sampleInputDir, exist_ok=True)
                        for i in range(min(inputs.size(0), 32)):
                            image = inputs[i, :, :, :]
                            imageArray = np.array(image.permute(1, 2, 0))
                            imagePos = imageArray - imageArray.min()
                            imageScaled = imagePos * (255.0 / imagePos.max())
                            imagePIL = Image.fromarray(imageScaled.astype(np.uint8))
                            imagePIL.save(f'{sampleInputDir}/{i:04}.png')

                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                    # get accuracy for this batch
                    acc1batch = accuracy(outputs, targets, (1,))[0].detach().cpu().item()
                    acc5batch = accuracy(outputs, targets, (5,))[0].detach().cpu().item()

                    # compute mean accuracy and loss for this epoch so far
                    if batch == 0:
                        acc1epoch = acc1batch
                        acc5epoch = acc5batch
                        lossEpoch = loss.item()
                    elif batch > 0:
                        acc1epoch = ((acc1epoch * batch) + acc1batch) / (batch + 1)
                        acc5epoch = ((acc5epoch * batch) + acc5batch) / (batch + 1)
                        lossEpoch = ((lossEpoch * batch) + loss.item()) / (batch + 1)

                    if trainEval == 'train' and epoch > 0:
                        loss.backward()
                        optimizer.step()

                    tepoch.set_postfix(acc1batch=f'{acc1batch:.2%}'.zfill(6),acc1epoch=f'{acc1epoch:.2%}'.zfill(6),lossBatch=f'{loss.item():.3f}')

                    # log
                    logNewRow = pd.DataFrame({'epoch': epoch,
                                          'batch': batch,
                                          'trainEval': trainEval,
                                          'acc1batch': acc1batch,
                                          'acc5batch': acc5batch,
                                          'lossBatch': loss.item(),
                                          'cumAcc1epoch': acc1epoch,
                                          'cumAcc5epoch': acc5epoch,
                                          'cumLossEpoch': lossEpoch}, index=[0])
                    log = pd.concat([log, logNewRow])

            # save plot stats
            plotStats[trainEval]['acc1'].append(acc1epoch)
            plotStats[trainEval]['acc5'].append(acc5epoch)
            plotStats[trainEval]['loss'].append(lossEpoch)
            pickle.dump(plotStats, open(plotStatsFile, 'wb'))

            # save parameters for this epoch
            epochSavePath = os.path.join(modelSavePath, '%03i.pt' % (epoch))
            params = {'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
            torch.save(params, epochSavePath)
            torch.cuda.empty_cache()

        # save log file
        log.to_csv(logFile, index=False)

        # make plots
        epochs = list(range(int(skipZeroth),epoch+1))
        for plotType in ['acc1', 'acc5', 'loss']:

            plt.plot(epochs, plotStats['train'][plotType][:epoch+1], label='train')
            plt.plot(epochs, plotStats['eval'][plotType][:epoch+1], label='eval')
            plt.xlabel('epoch')
            plt.ylabel(plotType)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plotDir, f'{plotType}.png'))
            plt.show()
            plt.close()

        # adapt learning rate
        if epoch > 0:
            scheduler.step()
            
    if returnModel:
        return model

    print('Done')

if __name__ == '__main__':
    train(modelName='alexnet',
          model=None,
          datasetPath='/home/dave/Datasets/imagenet16',
          pretrained=True,
          learningRate=.01,
          optimizerName='SGD',
          batchSize=256,
          nEpochs=25,
          restartFrom=None,
          workers=6,
          outDir=None,
          returnModel=False,
          nGPUs=-1,
          GPUids=None,
          skipZeroth=False,
          occlude=False,
          occlusionMethod=None,
          coverage=.4,
          propOccluded=1.0,
          colour=(0, 0, 0),
          invert=False,
          cycles=3,
          momentum=.9,
          weight_decay=1e-4,
          printOut={},
          blur=False,
          blurSigmas=None,
          blurWeights=None,
          noise=False,
          noiseLevels=None,
          times=[2,2,4,2])