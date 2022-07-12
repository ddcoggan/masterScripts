#!/usr/bin/python

import sys
from tqdm import tqdm
import os
import glob
import pickle
import itertools
import numpy as np
from sklearn import linear_model
from scipy.io import loadmat
from argparse import Namespace
import matplotlib.pyplot as plt
import nibabel as nib
import datetime
import shutil

sys.path.append('/mnt/HDD12TB/masterScripts')
from preprocess import preprocess
from fMRIhelperFunctions import hrf, do_GLM

expFile = 'analysis/scripts/experiment.pkl'
experiment = pickle.load(open(expFile, 'rb'))

preprocess(experiment)
overwrite = False
# first level analysis
for subject in experiment['scanInfo'].keys():
    for session in experiment['scanInfo'][subject].keys():
        for scan in experiment['design'].keys():
            for run_num, run in enumerate(experiment['scanInfo'][subject][session]['funcScans'][scan]):

                '''
                # for debugging purposes
                subject = list(experiment['scanInfo'].keys())[0]
                session = list(experiment['scanInfo'][subject].keys())[0]
                scan = list(experiment['design'].keys())[0]
                r, run = [0, experiment['scanInfo'][subject]['funcScans'][scan][0]]
                '''

                print('\n\nSubject: %s, Session: %s, Scan: %s, Run: %02i' %(subject, session, scan, run_num + 1))

                # get design
                print('Loading Design...')
                params = Namespace(**experiment['design'][scan]['params'])
                dynamics = np.arange(0,params.nDynamics*params.TR, params.TR)
                nBlocks = int(((params.nDynamics * params.TR) - (params.initialFixation + params.finalFixation)) / (params.blockDuration + params.IBI))
                variables = list(experiment['design'][scan]['conditions'].keys())
                levels = [experiment['design'][scan]['conditions'][variable] for variable in variables]
                conds = list(itertools.product(*levels))
                nConds = len(conds)
                nReps = int(nBlocks/nConds)
                condNames = []
                for cond in conds:
                    if scan == 'figureGround':
                        condNames.append(cond[0])
                    elif scan == 'figureGround_loc':
                        condNames = ['stimulus']
                contrastsNoNuisance = np.array(experiment['design'][scan]['contrasts']['matrix'])
                contrastNames = experiment['design'][scan]['contrasts']['names']


                # get event data
                if nConds > 1:
                    eventFile = glob.glob('data/fMRI/eventFiles/%s/%s/%s/*_rn%i_*' %(subject, session, scan, run_num+1))[0]
                    eventData = loadmat(eventFile)
                    eventData = eventData['experiment']['conditions']
                    blockOrder = []
                    for b in range(nBlocks):
                        thisCond = []
                        item = eventData[0,0][0][b] # matlab structures have horrendous indexing when importing to python
                        if type(item) is np.ndarray:
                            item = item[0] # strip array structure if necessary
                        thisCond.append(item)
                        blockOrder.append(conds.index(tuple(thisCond)))
                else:
                    blockOrder = np.zeros(nReps, dtype=int)

                # make boxcar models
                boxCars = np.zeros(shape=[nConds + 1, params.nDynamics])
                boxCars[0, :] = 1
                for b in range(nBlocks):
                    c = blockOrder[b]
                    startTR = int((params.initialFixation + b * (params.blockDuration + params.IBI)) / params.TR)
                    boxCars[c + 1, startTR:int(startTR + (params.blockDuration / params.TR))] = 1

                # see if there are more levels of directories, e.g. with and without topup
                preprocDir = os.path.join('data/fMRI/individual', subject, session, 'preprocessing', scan, 'run%02i' % (run_num + 1))
                inDirs = list()
                for root, dirs, files in os.walk(preprocDir):
                    if not dirs:
                        inDirs.append(root)

                # run GLM
                for inDir in inDirs:

                    fMRIpath = None

                    for HRFmodel in ['single','double']:

                        # set output directory
                        extraLevels = inDir.replace(preprocDir, '')
                        outDir = os.path.join(
                            'data/fMRI/individual/%s/%s/firstLevel/%s/run%02i%s' % (subject, session, scan, run_num + 1, extraLevels),
                            '%sGamma' % HRFmodel)
                        if not os.path.exists(outDir):
                            os.makedirs(outDir)
                        print('Directory: %s, Date/Time: %s' %(outDir, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

                        GLMfile = outDir + '/GLMresults.pkl'

                        if not os.path.exists(GLMfile) or overwrite == True:

                            print('Analysis found: %r, Overwrite: %r. Running...' %(os.path.exists(GLMfile), overwrite))

                            # load fMRI data
                            if fMRIpath is None:
                                print('Loading fMRI timeseries...')
                                fMRIpath = sorted(glob.glob(os.path.join(inDir, '*')))[-1]
                                fMRIdataNifti = nib.load(fMRIpath)
                                fMRIdata = fMRIdataNifti.get_fdata()
                                X, Y, Z, T = fMRIdata.shape
                                df = T-1

                                # convert to percent signal change
                                meanVolume = np.mean(fMRIdata, axis=3)
                                meanVolume4D = np.repeat(meanVolume[:, :, :, np.newaxis], T, axis=3)
                                fMRIdataPSC = (fMRIdata - meanVolume4D)
                                for vt in itertools.product(list(range(X)), list(range(Y)), list(range(Z)), list(range(T))):  # have to loop over voxels to avoid dividing by zero on out of head voxels
                                    if meanVolume[vt[0:3]] != 0:
                                        fMRIdataPSC[vt] = (fMRIdataPSC[vt] / meanVolume[vt[0:3]]) * 100

                            # make mean functional image for plotting purposes
                            print('Calculating mean functional image...')
                            os.system('fslmaths %s -Tmean %s/meanFunc.nii.gz' % (fMRIpath, outDir))
                            os.system('bet %s/meanFunc.nii.gz %s/meanFuncBet.nii.gz' % (outDir, outDir))
                            os.system('fslmaths %s/meanFuncBet.nii.gz -bin %s/meanFuncMask.nii.gz' % (outDir, outDir))
                            meanFuncNifti = nib.load('%s/meanFuncMask.nii.gz' % outDir)
                            meanFuncMask = meanFuncNifti.get_fdata()

                            # convolve box cars with hrf and add derivatives
                            print('Creating regressors...')
                            thisHRF = hrf(np.array(list(range(0, 30, params.TR))), gamma=HRFmodel)
                            designMatrix = np.zeros(shape = [T, nConds*2+1])
                            designMatrix[:,1] = 1
                            for c in range(nConds):
                                # standard regressors
                                prediction = np.convolve(boxCars[c+1,:], thisHRF)
                                designMatrix[:,c+1] = prediction[0:T]/np.max(prediction[0:T])
                                # derivatives (nuisance regressors)
                                derivative = np.diff(prediction)
                                designMatrix[:, nConds+c+1] = derivative[0:T]/np.max(derivative[0:T])
                            # motion parameters (nuisance regressors)
                            motionFile = glob.glob(os.path.join(inDir, '3_motionCorrected.par'))[0]
                            motion = np.genfromtxt(motionFile, delimiter='  ')
                            designMatrix = np.concatenate((designMatrix, motion), axis = 1)

                            # plot design matrix
                            plt.figure(figsize=(10, 3))
                            plt.imshow(designMatrix, cmap = 'gray')
                            plt.axes().set_aspect('auto')
                            plt.colorbar()
                            plt.xlabel('contrast (predictors - derivatives - motion params)')
                            plt.ylabel('time (TR)')
                            plt.title('design matrix')
                            plt.savefig(outDir + '/designMatrix.png')
                            plt.show()
                            plt.close()

                            # specify contrasts from experiment file
                            if len(contrastsNoNuisance) > 0:
                                if len(contrastsNoNuisance.shape) < 2:
                                    nContrasts=1
                                    contrasts = np.zeros(designMatrix.shape[1])
                                    contrasts[1:(nConds+1)] = contrastsNoNuisance
                                else:
                                    nContrasts = contrastsNoNuisance.shape[0]
                                    contrasts = np.zeros(nContrasts, designMatrix.shape[1])
                                    contrasts[:, 1:(nConds+1)] = contrastsNoNuisance
                            else:
                                contrasts=None

                            # run GLM
                            print('Running GLM...' )
                            beta, model, residual, R, MSE, se_estimate, contrast_beta, se_contrast_beta, contrast_t = do_GLM(designMatrix, fMRIdataPSC, contrasts)

                            # save out analysis (apart from beta and t which are saved as NIFTIs
                            GLMresults = {'model': model, 'residual': residual, 'R': R,'MSE': MSE, 'se_estimate': se_estimate, 'se_contrast_beta': se_contrast_beta}
                            pickle.dump(GLMresults, open(GLMfile, 'wb'))

                            # save out activation maps as NIFTIs
                            print('Making activation maps...')
                            actDir = os.path.join(outDir, 'activationMaps')
                            if not os.path.exists(actDir):
                                os.makedirs(actDir)

                            # beta map of original conditions
                            for c, cond in enumerate(condNames):

                                activationMap = beta[:, :, :, c + 1]
                                img = nib.Nifti1Image(activationMap, meanFuncNifti.affine,
                                                      meanFuncNifti.header)  # copy affine and header from meanFunc
                                nib.save(img, os.path.join(actDir, 'pe%02i_%s_beta.nii.gz' % (c+1, cond)))

                             # beta and t maps for contrasts
                            if len(contrasts) > 0:
                                for c, name in enumerate(contrastNames):

                                    # beta
                                    activationMap = contrast_beta[:, :, :, c]
                                    img = nib.Nifti1Image(activationMap, meanFuncNifti.affine,
                                                          meanFuncNifti.header)  # copy affine and header from meanFunc
                                    nib.save(img, os.path.join(actDir, 'cope%02i_%s_beta.nii.gz' %(c+1, name)))

                                    # t
                                    activationMap = contrast_t[:, :, :, c]
                                    img = nib.Nifti1Image(activationMap, meanFuncNifti.affine,
                                                          meanFuncNifti.header)  # copy affine and header from meanFunc
                                    nib.save(img, os.path.join(actDir, 'cope%02i_%s_t.nii.gz' %(c+1, name)))

                            # plot model fit for voxel with largest beta values (L2 norm)
                            print('Plotting peak voxel GLM...')
                            maxCoords = np.unravel_index(np.nanargmax(GLMresults['R']),[X, Y, Z])
                            print('Coordinates of max product of beta values: %i %i %i\n' % (maxCoords))
                            plt.figure(figsize=(20, 3))
                            plt.plot(dynamics, fMRIdataPSC[maxCoords], 'k', label='data')
                            plt.plot(dynamics, GLMresults['model'][maxCoords], 'r', label='model')
                            '''
                            include models * respective betas in plot
                            # fitted_models = np.transpose([designMatrix[:, c] * beta_hat[c] for c in range(1, nConds + 1)])
                            # plt.plot(dynamics, fitted_models, 'b', label = 'regressors') # in case you want to
                            '''
                            plt.xlabel('time (s)')
                            plt.ylabel('percent signal change')
                            plt.title('voxel location: %s' % (str(maxCoords)))
                            plt.savefig(outDir + '/peakVoxelFit.png')
                            plt.show()
                            plt.close()

                        else:
                            print('Analysis found: %r, Overwrite: %r. Skipping Analysis.\n' % (os.path.exists(GLMfile), overwrite))

    # Combine results across runs for each subject
    print('Combining across runs/sessions...')
    for distCor in ['noDistCor', 'TopUp', 'b0']:

        print('\nDistortion correction type: %s' %distCor)
        print('Coregistering individual maps...')

        # all data is registered to the reg run of the first session
        regSession = list(experiment['scanInfo'][subject].keys())[0]
        regScan = list(experiment['scanInfo'][subject][regSession]['funcScans'].keys())[0]
        runs = experiment['scanInfo'][subject][regSession]['funcScans'][regScan]
        regRun = runs[0]
        run = sorted(runs).index(regRun) + 1
        targetImage = glob.glob(os.path.join('data/fMRI/individual', subject, regSession, 'preprocessing', regScan, 'run%02i' %run, distCor, '*Tmean*'))[0]

        outDir = os.path.join('data/fMRI/individual', subject, 'allSessions', distCor)
        os.makedirs(os.path.join(outDir, 'reg'), exist_ok=True)

        # copy over a mean functional image
        shutil.copy(os.path.join('data/fMRI/individual', subject, regSession, 'firstLevel', regScan, 'run%02i' %run, distCor, 'singleGamma/meanFunc.nii.gz'), os.path.join(outDir, 'reg/meanFunc.nii.gz'))

        for s, session in enumerate(experiment['scanInfo'][subject].keys()):

            # make transformation matrices
            print('Making transformation matrices...')
            if s > 0: # skip the first session as these are already in the reg space

                # find the reg run used for this session
                thisRegScan = list(experiment['scanInfo'][subject][session]['funcScans'].keys())[0]
                theseRuns = experiment['scanInfo'][subject][session]['funcScans'][thisRegScan]
                thisRegRun = theseRuns[0]
                thisRun = sorted(theseRuns).index(thisRegRun) + 1
                inputImage = glob.glob(os.path.join('data/fMRI/individual', subject, session, 'preprocessing', thisRegScan, 'run%02i' % thisRun, distCor, '*Tmean*'))[0]

                # create mapping between this session and first session
                transMat = os.path.join(outDir, 'reg/%s_to_%s.mat' %(session, regSession))
                if not os.path.isfile(transMat):
                    os.system('flirt -in %s -ref %s -omat %s' % (inputImage, targetImage, transMat))

            print('Transforming parameter estimates...')
            for scan in experiment['design'].keys():
                runDirs = sorted(glob.glob(os.path.join('data/fMRI/individual', subject, session, 'firstLevel', scan, 'run*')))
                for r, runDir in enumerate(runDirs):
                    for HRFmodel in ['single', 'double']:

                        thisOutDir = os.path.join(outDir, '%sGamma' %HRFmodel, scan, 'singleRunMaps')
                        os.makedirs(thisOutDir, exist_ok=True)

                        maps = sorted(glob.glob(os.path.join(runDir, distCor, '%sGamma' %HRFmodel, 'activationMaps/*beta*')))
                        for map in maps:
                            mapName = os.path.basename(map)
                            outFile = os.path.join(thisOutDir, '%s_run%02i_%s' %(session, r+1, mapName))

                            if not os.path.isfile(outFile):

                                # if first session, just copy over
                                if s == 0:
                                    shutil.copy(map, outFile)

                                # maps from other sessions need to be registered to first session
                                else:
                                    os.system('flirt -applyxfm -in %s -ref %s -out %s -init %s' % (map, targetImage, outFile, transMat))

        # average across all sessions
        print('Averaging across all sessions...')
        for HRFmodel in ['single', 'double']:
            for scan in experiment['design'].keys():
                inDir = os.path.join(outDir, '%sGamma' %HRFmodel, scan, 'singleRunMaps')

                # parameter estimates
                for imageType in ['pe', 'cope']:
                    imageCounter = 1
                    while len(glob.glob(os.path.join(inDir, '*_%s%02i*' %(imageType, imageCounter)))) > 0:
                        indMaps = glob.glob(os.path.join(inDir, '*_%s%02i*' %(imageType, imageCounter)))
                        condInfo = indMaps[0].split(sep='_')
                        if scan == 'figureGround':
                            condName = '%s_%s' % (condInfo[2], condInfo[3])
                        elif scan == 'figureGround_loc':
                            condName = '%s_%s' %(condInfo[3], condInfo[4])


                        # average the beta maps
                        outFile = os.path.join(outDir, '%sGamma' % HRFmodel, scan, '%s_beta.nii.gz' % condName)
                        if not os.path.isfile(outFile):
                            fslCommand = 'fslmaths %s' % indMaps[0]
                            for indMap in indMaps[1:]:
                                fslCommand += ' -add %s' %indMap
                            fslCommand += ' -div %i %s' %(len(indMaps), outFile)
                            os.system(fslCommand)

                        # calculate t maps
                        sampleMean = outFile
                        outFile = os.path.join(outDir, '%sGamma' % HRFmodel, scan, '%s_t.nii.gz' % condName)
                        if not os.path.isfile(outFile):
                            tempDir = os.path.join(outDir, '%sGamma' % HRFmodel, scan, 'temp')
                            os.makedirs(tempDir, exist_ok=True)

                            # standard deviation
                            for indMap in indMaps:
                                # subtract mean
                                indMapDiff = os.path.join(tempDir, '%s_meanCentre.nii.gz' %(os.path.basename(indMap)[:-7]))
                                os.system('fslmaths %s -sub %s %s' %(indMap, sampleMean, indMapDiff))

                                # squared error
                                indMapSqErr = os.path.join(tempDir, '%s_SqErr.nii.gz' %(os.path.basename(indMap)[:-7]))
                                os.system('fslmaths %s -sqr %s' %(indMapDiff, indMapSqErr))

                            # sum squared errors and divid by n-1
                            fslCommand = 'fslmaths %s' % os.path.join(tempDir, '%s_SqErr.nii.gz' %(os.path.basename(indMap)[:-7]))
                            for indMap in indMaps[1:]:
                                indMapSqErr = os.path.join(tempDir, '%s_SqErr.nii.gz' % (os.path.basename(indMap)[:-7]))
                                fslCommand += ' -add %s' %  indMapSqErr
                            fslCommand += ' -div %i -sqrt %s' % (len(indMaps)*-1, os.path.join(tempDir, '%s_standardDeviation.nii.gz' %condName))
                            os.system(fslCommand)

                            # t map
                            os.system('fslmaths %s -div %f %s' %(os.path.join(tempDir, '%s_standardDeviation.nii.gz' %condName), np.sqrt(len(indMaps)), os.path.join(tempDir, '%s_standardError.nii.gz' %condName)))
                            os.system('fslmaths %s -div %s %s' %(sampleMean, os.path.join(tempDir, '%s_standardError.nii.gz' %condName), outFile))

                        imageCounter += 1

print('Done.')
