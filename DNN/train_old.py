import os
import glob
import sys
import datetime
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader # Takes ImageFolder as a parameter
import torch.optim as optim
from torch.autograd.variable import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from accuracy import accuracy
import zoo

def train(modelName='alexnet', datasetPath = '/home/dave/Datasets', pretrained=True, learningRate=.01, optimizer='SGD', batchSize=256, nEpochs=25, restartFrom=None, workers=6, outDir=None, transform=None):

    ### CONFIGURATION
    os.makedirs(outDir, exist_ok=True) # make output directory

    # record configuration
    config = {'model': modelName, 'datasetPath': datasetPath, 'pretrained': pretrained,
              'learningRate': learningRate, 'optimizer': optimizer, 'batchSize': batchSize,
              'nEpochs': nEpochs, 'outDir': outDir, 'transform': transform}
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

    # create image preprocessor
    if transform == None:
        transformSequence = [
                transforms.Resize(224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(), # transform PIL image to Tensor.
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] # Only normalize on Tensor datatype.
        transform = transforms.Compose(transformSequence)

    # create image loaders
    trainData = ImageFolder(trainPath, transform=transform)
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=workers)
    valData = ImageFolder(valPath, transform=transform)
    valLoader = DataLoader(valData, batch_size=batchSize, shuffle=True, num_workers=workers)

    ### MODEL HANDLING ###

    # get model
    model = getattr(zoo, modelName)(pretrained=pretrained)

    # make sure last layer is adapted to the number of classes
    nClasses = len(glob.glob(os.path.join(trainPath, '*')))
    try:
        inFeatures = model.classifier[-1].in_features
        model.classifier.add_module(name=str(len(model.classifier)-1), module=nn.Linear(inFeatures, nClasses, True))
    except:
        inFeatures = model.decoder[-2].in_features
        model.decoder.add_module(name='linear', module=nn.Linear(inFeatures, nClasses, True))
    
    # show model architecture
    print(model)

    # set optimizer
    if optimizer == 'SGD':
        optimizer = optim.SGD(params=model.parameters(), lr=learningRate)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=learningRate)

    # get GPU info
    print(f'Using {torch.cuda.device_count()} GPUs')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()  # put model on cuda

    # set starting epoch
    starting_epoch = 0
    if restartFrom is not None:

        # load network weights
        print("Resuming from weights at %s" %(restartFrom))
        resume_weight = torch.load(restartFrom)
        model.load_state_dict(resume_weight['model'])
        optimizer.load_state_dict(resume_weight['optimizer'])
        starting_epoch = int(restartFrom.split('/')[-1].split('.')[0]) + 1

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8)
    loss_fn = nn.CrossEntropyLoss().cuda()


    for epoch in range(starting_epoch, nEpochs+1):

        if epoch == 0:
            log = pd.DataFrame(
                columns=['epoch', 'batch', 'trainEval', 'acc1batch', 'acc5batch', 'lossBatch', 'cumAcc1epoch',
                         'cumAcc5epoch', 'cumLossEpoch'])
            plotStats = {'train': {'acc1': [], 'acc5': [], 'loss': []},
                         'eval': {'acc1': [], 'acc5': [], 'loss': []}}
        else:

            log = pd.read_csv(logFile)
            plotStats = pickle.load(open(plotStatsFile, 'rb'))

        for trainEval in ['train','eval']:

            # train/eval specific settings
            if trainEval == 'train':

                # for zeroth epoch, just measure performance, no training
                if epoch == 0:
                    model.eval()
                else:
                    model.train()

                loader = trainLoader
                logString = 'Training'

            elif trainEval == 'eval':

                model.eval()
                loader = valLoader
                logString = 'Evaluating'

            # track accuracy and loss across entire epoch
            acc1epoch = 0.0
            acc5epoch = 0.0
            lossEpoch = 0.0

            for batch, (inputs, targets) in enumerate(loader):

                if trainEval == 'train' and epoch > 0:
                    optimizer.zero_grad()
                    inputs, targets = Variable(inputs), Variable(targets)

                inputs, targets = inputs.cuda(), targets.cuda()
                output = model(inputs)

                # get accuracy and loss for this batch
                acc1batch = accuracy(output, targets, (1,))[0].detach().cpu().item()
                acc5batch = accuracy(output, targets, (5,))[0].detach().cpu().item()
                loss = loss_fn(output, targets)

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

                print(
                    f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}. {logString}...\t\tEpoch: {epoch}, Batch: {batch}, Top 1 accuracy (batch): {acc1batch}, Top 1 accuracy (epoch): {acc1epoch}')

                # log
                log = log.append({'epoch': epoch,
                                      'batch': batch,
                                      'trainEval': trainEval,
                                      'acc1batch': acc1batch,
                                      'acc5batch': acc5batch,
                                      'lossBatch': loss.item(),
                                      'cumAcc1epoch': acc1epoch,
                                      'cumAcc5epoch': acc5epoch,
                                      'cumLossEpoch': lossEpoch},
                                     ignore_index=True)

            # save log file
            log.to_csv(logFile, index=False)

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

        # make plots
        epochs = list(range(epoch+1))
        for plotType in ['acc1', 'acc5', 'loss']:
            plt.plot(epochs, plotStats['train'][plotType], label='train')
            plt.plot(epochs, plotStats['eval'][plotType], label='test')
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

    print('Done')
