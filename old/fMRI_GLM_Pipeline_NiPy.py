#!/usr/bin/python

import sys
from tqdm import tqdm
import os
import glob
import pickle
import itertools
import numpy as np
from scipy.io import loadmat
from argparse import Namespace
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel import save
import datetime
import shutil
from nipy.modalities.fmri.glm import FMRILinearModel

sys.path.append('/mnt/HDD12TB/masterScripts')
from preprocess import preprocess
from fMRIhelperFunctions import hrf

expFile = 'analysis/scripts/experiment.pkl'
experiment = pickle.load(open(expFile, 'rb'))

preprocess(experiment)
overwrite = True
# first level analysis
for subject in experiment['scanInfo'].keys():
    for session in experiment['scanInfo'][subject].keys():
        for scan in experiment['design'].keys():

            # configure dictionary to store fMRI paths and designs for analysis across all runs
            allRuns_paths = {'noDistCor': {'singleGamma': [], 'doubleGamma': []},
                           'TopUp': {'singleGamma': [], 'doubleGamma': []},
                           'b0': {'singleGamma': [], 'doubleGamma': []}}
            allRuns_designs = {'noDistCor': {'singleGamma': [], 'doubleGamma': []},
                           'TopUp': {'singleGamma': [], 'doubleGamma': []},
                           'b0': {'singleGamma': [], 'doubleGamma': []}}

            nRuns = len(list(experiment['scanInfo'][subject][session]['funcScans'][scan]))

            # collate design matrices and data paths
            for run_num, run in enumerate(experiment['scanInfo'][subject][session]['funcScans'][scan]):

                '''
                # for debugging purposes
                subject = list(experiment['scanInfo'].keys())[0]
                session = list(experiment['scanInfo'][subject].keys())[0]
                scan = list(experiment['design'].keys())[0]
                r, run = [0, experiment['scanInfo'][subject]['funcScans'][scan][0]]
                '''

                print('\nCollating paths and designs. Subject: %s, Session: %s, Scan: %s, Run: %02i' %(subject, session, scan, run_num + 1))

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
                contrasts = experiment['design'][scan]['contrasts']


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

                # run GLM
                for distCor in ['noDistCor', 'TopUp', 'b0']:

                    fMRIpath = os.path.join(preprocDir, distCor, '6_spatiallyFiltered.nii.gz')
                    fMRIpathTmean = fMRIpath[:-7] + '_Tmean.nii.gz'
                    fMRIpathPSC = fMRIpath[:-7] + '_PSC.nii.gz'
                    if not os.path.isfile(fMRIpathPSC):
                        os.system('fslmaths %s -Tmean %s' % (fMRIpath, fMRIpathTmean))
                        os.system('fslmaths %s -sub %s -div %s -mul 100 %s' % (
                        fMRIpath, fMRIpathTmean, fMRIpathTmean, fMRIpathPSC))

                    # add copies of TMean and other useful images to outDir
                    outDir = os.path.join('data/fMRI/individual/%s/%s/firstLevel/%s/run%02i/%s' % (subject, session, scan, run_num + 1, distCor))
                    os.makedirs(outDir, exist_ok=True)
                    if not os.path.isfile('%s/meanFunc.nii.gz' % (outDir)):
                        os.system('fslmaths %s -Tmean %s/meanFunc.nii.gz' % (fMRIpath, outDir))
                        os.system('bet %s/meanFunc.nii.gz %s/meanFuncBet.nii.gz' % (outDir, outDir))
                        os.system('fslmaths %s/meanFuncBet.nii.gz -bin %s/meanFuncMask.nii.gz' % (outDir, outDir))
                        meanFuncNifti = nib.load('%s/meanFuncMask.nii.gz' % outDir)

                    for HRFmodel in ['single','double']:

                        outDir = os.path.join('data/fMRI/individual/%s/%s/firstLevel/%s/run%02i/%s/%sGamma' % (
                        subject, session, scan, run_num + 1, distCor, HRFmodel))
                        os.makedirs(outDir, exist_ok=True)

                        # convolve box cars with hrf and add derivatives
                        thisHRF = hrf(np.array(list(range(0, 30, params.TR))), gamma=HRFmodel)
                        designMatrix = np.zeros(shape = [params.nDynamics, nConds*2+1])
                        designMatrix[:,0] = 1
                        for c in range(nConds):
                            # standard regressors
                            prediction = np.convolve(boxCars[c+1,:], thisHRF)
                            designMatrix[:,c+1] = prediction[0:params.nDynamics]/np.max(prediction[0:params.nDynamics])
                            # derivatives (nuisance regressors)
                            derivative = np.diff(prediction)
                            designMatrix[:, nConds+c+1] = derivative[0:params.nDynamics]/np.max(derivative[0:params.nDynamics])
                        # motion parameters (nuisance regressors)
                        motionFile = glob.glob(os.path.join(os.path.dirname(fMRIpathPSC), '3_motionCorrected.par'))[0]
                        motion = np.genfromtxt(motionFile, delimiter='  ')
                        # mean center and normalise
                        for c in range(6):
                            motion[:,c] = (motion[:,c] - np.mean(motion[:,c])) /np.max(motion[:,c])
                        designMatrix = np.concatenate((designMatrix, motion), axis = 1)

                        # make list of regressors
                        regressors = list(['intercept'])
                        for i in condNames:
                            regressors.append(i)
                        for i in condNames:
                            regressors.append('%s_deriv' %i)
                        motionNames = ['rot_X', 'rot_Y', 'rot_Z', 'trans_X', 'trans_Y', 'trans_Z']
                        for i in motionNames:
                            regressors.append(i)

                        # plot design matrix
                        plt.figure(figsize=(10, 3))
                        plt.imshow(designMatrix, cmap = 'gray')
                        plt.axes().set_aspect('auto')
                        plt.colorbar()
                        plt.xlabel('regressors')
                        plt.xticks(range(len(regressors)), regressors, rotation = 45)
                        plt.ylabel('time (TR)')
                        plt.title('design matrix')
                        plt.savefig(outDir + '/designMatrix.png')
                        plt.show()
                        plt.close()

                        # add paths and designs to dictionary
                        allRuns_designs[distCor]['%sGamma'%HRFmodel].append(designMatrix)
                        allRuns_paths[distCor]['%sGamma'%HRFmodel].append(fMRIpathPSC)

            # run GLM
            for distCor in ['noDistCor', 'TopUp', 'b0']:
                for HRFmodel in ['single', 'double']:

                    # analyse each run individually
                    for run_num, run in enumerate(experiment['scanInfo'][subject][session]['funcScans'][scan]):

                        print('Subject: %s, Session: %s, Scan: %s, Run: %02i, distCor: %s, HRFmodel: %s' % (
                        subject, session, scan, run_num + 1, distCor, '%sGamma'%HRFmodel))
                        outDir = os.path.join('data/fMRI/individual', subject, session, 'firstLevel', scan,
                                              'run%02i' % (run_num + 1), distCor, '%sGamma'%HRFmodel)
                        os.makedirs(outDir + '/activationMaps', exist_ok=True)

                        GLMfile = os.path.join(outDir, 'GLMfile.pkl')

                        if not os.path.isfile(GLMfile) or overwrite:

                            print('Analysis found: %r, overwrite: %r. Running GLM Analysis...' %(os.path.isfile(GLMfile), overwrite))
                            fmri_glm = FMRILinearModel(allRuns_paths[distCor]['%sGamma'%HRFmodel][run_num], allRuns_designs[distCor]['%sGamma'%HRFmodel][run_num])
                            fmri_glm.fit(do_scaling=True, model='ar1')
                            for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):

                                # add nuisance regressors to contrast_val
                                contrast_val_full = np.zeros(len(contrast_val)*2 +7)
                                contrast_val_full[1:(len(contrast_val)+1)] = contrast_val # intercept before, derivatives and motion params after
                                print('  Contrast %02i out of %02i: %s' %
                                      (index + 1, len(contrasts), contrast_id))

                                # save the z_image and b_image
                                z_map, t_map, = fmri_glm.contrast(contrast_val_full, con_id=contrast_id, output_z=True, output_stat=True)
                                image_path = os.path.join(outDir, 'activationMaps/%s_z_map.nii' % contrast_id)
                                save(z_map, image_path)
                                image_path = os.path.join(outDir, 'activationMaps/%s_t_map.nii' % contrast_id)
                                save(t_map, image_path)

                            pickle.dump(fmri_glm, open(GLMfile, 'wb'))

                        else:
                            print('Analysis found: %r, Overwrite: %r. Skipping GLM Analysis.\n' % (os.path.exists(GLMfile), overwrite))

                    # now perform across all runs
                    print('Subject: %s, Session: %s, Scan: %s, Run: allRuns, distCor: %s, HRFmodel: %s' % (
                        subject, session, scan, distCor, '%sGamma'%HRFmodel))

                    outDir = os.path.join('data/fMRI/individual', subject, session, 'firstLevel', scan, 'allRuns', distCor, '%sGamma'%HRFmodel)
                    os.makedirs(outDir + '/activationMaps', exist_ok=True)
                    GLMfile = os.path.join(outDir, 'GLMfile.pkl')

                    if not os.path.isfile(GLMfile) or overwrite:

                        print('Analysis found: %r, overwrite: %r. Running GLM Analysis...' % (
                        os.path.isfile(GLMfile), overwrite))
                        fmri_glm = FMRILinearModel(allRuns_paths[distCor]['%sGamma'%HRFmodel],
                                                   allRuns_designs[distCor]['%sGamma'%HRFmodel])
                        fmri_glm.fit(do_scaling=True, model='ar1')
                        for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
                            # add nuisance regressors to contrast_val
                            contrast_val_full = np.zeros(len(contrast_val) * 2 + 7)
                            contrast_val_full[1:(len(
                                contrast_val) + 1)] = contrast_val  # intercept before, derivatives and motion params after
                            print('  Contrast %02i out of %02i: %s' %
                                  (index + 1, len(contrasts), contrast_id))

                            contrast_val_full = list([contrast_val_full,]*nRuns)
                            # save the z_image and t_image
                            t_map, z_map, = fmri_glm.contrast(contrast_val_full, con_id=contrast_id, output_z=True,
                                                              output_stat=True)
                            image_path = os.path.join(outDir, 'activationMaps/%s_z_map.nii' % contrast_id)
                            save(z_map, image_path)
                            image_path = os.path.join(outDir, 'activationMaps/%s_t_map.nii' % contrast_id)
                            save(t_map, image_path)

                        pickle.dump(fmri_glm, open(GLMfile, 'wb'))

                    else:
                        print('Analysis found: %r, Overwrite: %r. Skipping GLM Analysis.\n' % (
                        os.path.exists(GLMfile), overwrite))
print('Done')