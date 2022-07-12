import os
import glob
import sys
#sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
sys.path.append('/USers/david/Library/Mobile Documents/com~apple~CloudDocs/Work/masterScripts/DNN')
from train import train
import time
#time.sleep(18000)

overwrite = False
noise=False
indCoverages = [.1,.2,.4,.8]
occluders = []
for x in sorted(glob.glob('DNN/images/occluders/*')):
        occluders.append(os.path.basename(x))
occluders.append('unoccluded')
occludersFMRI = ['barHorz04','barVert12','barHorz08']
occludersNoLevels = ['unoccluded','naturalTextured','naturalTextured2']
occludersBehavioural = ['barHorz04', 'barVert04', 'barObl04', 'mudSplash', 'polkadot','polkasquare','crossBarOblique','crossBarCardinal', 'naturalUntexturedCropped2']
occludersWithLevels = occluders
for o in occludersNoLevels:
    occludersWithLevels.remove(o)

config = {'''
          'allAlexnet': {'alexnet': {'imagenet16': {'occluders': occluders, 'coverages': indCoverages}}},
          'mixedTypesMixedLevels': {'vgg19': {'imagenet16': {'occluders': [occludersBehavioural], 'coverages': [indCoverages]}}},
          'mixedLevelsMixedBlur': {'alexnet': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}}},
          'mixedLevels': {'resnet18': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'resnet34': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'resnet50': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'resnet101': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'resnet152': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'vgg19': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'cornet_s': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'PredNetImageNet': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'inception_v3': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}},
                          'alexnet': {'imagenet16': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}}},
                                      #'imagenet1000': {'occluders': occludersWithLevels, 'coverages': [indCoverages]}}},
          'fMRIandNatural': {'alexnet': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'cornet_s': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'resnet18': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'resnet34': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'resnet50': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'resnet101': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'resnet152': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'vgg19': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'inception_v3': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}},
                             'PredNetImageNet': {'imagenet16': {'occluders': occludersFMRI + occludersNoLevels, 'coverages': [.5]}}},
                             '''
          'cornet_s_varRec_2_2_4_2': {'cornet_s_varRec': {'imagenet1000': {'occluders': ['barHorz08'], 'coverages': [.5], 'times': [2,2,4,2]}}},
          'cornet_s_varRec_3_3_6_3': {'cornet_s_varRec': {'imagenet1000': {'occluders': ['barHorz08'], 'coverages': [.5], 'times': [2,2,4,2]}}},
          'cornet_s_varRec_4_4_6_4': {'cornet_s_varRec': {'imagenet1000': {'occluders': ['barHorz08'], 'coverages': [.5], 'times': [2,2,4,2]}}},
          'cornet_s_varRec_5_5_10_5': {'cornet_s_varRec': {'imagenet1000': {'occluders': ['barHorz08'], 'coverages': [.5], 'times': [2,2,4,2]}}}}

learningRate = .001
optimizerName = 'SGD'
batchSizes = {'alexnet': 1024,
              'vgg19': 128,
              'cornet_s': 256,
              'cornet_s_varRec': 128,
              'resnet18': 512,
              'resnet34': 256,
              'resnet50': 64,
              'resnet101': 64,
              'resnet152': 64,
              'inception_v3': 32,
              'PredNetImageNet': 8}
nEpochs = 32
workers = 8
pretrained = True
colours = [(0,0,0),(127,127,127),(255,255,255)]
invert=False
propOccluded = 0.8

# currently just for prednet
cycles=3
momentum=.9
weight_decay=1e-4

for analysis in config:

    for modelName in config[analysis]:

        batchSize = batchSizes[modelName]

        for dataset in config[analysis][modelName]:

            datasetPath = f'/home/dave/Datasets/{dataset}'

            for occluder in config[analysis][modelName][dataset]['occluders']:

                if type(occluder) is list:
                    occluderString = 'mixedOccluders'
                else:
                    occluderString = occluder

                for coverage in config[analysis][modelName][dataset]['coverages']:

                    if type(coverage) is list:
                        coverageString = 'mixedLevels'
                    else:
                       coverageString = str(int(coverage*100))

                    if occluder in occludersNoLevels:
                        outDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occluder)
                    else:
                        outDir = os.path.join('DNN/data', modelName, dataset, 'fromPretrained', occluderString, coverageString)

                    outDir = ['model']

                    if analysis.endswith('MixedBlur'):
                        outDir += 'MixedBlur'
                        blur = True
                    else:
                        blur = False

                    if modelName == 'cornet_s_varRec':
                        times = config[analysis][modelName][dataset]['times']

                    # get restart from file if necessary
                    weightFiles = sorted(glob.glob(os.path.join(outDir, 'params/*.pt')))
                    if 0 < len(weightFiles) and overwrite == False:
                        restartFrom = weightFiles[-1]
                    else:
                        restartFrom = None

                    # print out these values during training
                    printOut = {'model': modelName,
                                'dataset': dataset,
                                'occluder': occluder,
                                'coverage': str(coverage)}

                    # call script
                    if len(weightFiles) < nEpochs+1 or overwrite:
                        train(modelName=modelName,
                              model=None,
                              datasetPath=datasetPath,
                              pretrained=pretrained,
                              learningRate=learningRate,
                              optimizerName=optimizerName,
                              batchSize=batchSize,
                              nEpochs=nEpochs,
                              restartFrom = restartFrom,
                              workers=workers,
                              outDir=outDir,
                              occlude=True,
                              occlusionMethod=occluder,
                              coverage=coverage,
                              propOccluded=propOccluded,
                              colour=colours,
                              invert=invert,
                              cycles=cycles,
                              momentum=momentum,
                              weight_decay=weight_decay,
                              printOut=printOut,
                              blur=blur,
                              blurSigmas = [0,1,2,4,8],
                              blurWeights=[.2,.2,.2,.2,.2],
                              noise=noise,
                              noiseLevels=[1,.8,.4,.2,.1],
                              times=times
                              )

