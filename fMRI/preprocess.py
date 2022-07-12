#!/usr/bin/python
'''
Master script for preprocessing fMRI data
author: DDC 2019/12/18

Pass a dictionary with the experiment parameters, example below.

dataDir = '/mnt/HDD12TB/projects/p012_surroundSuppressionLGN/data/fMRI'
freesurferDir = os.path.join(dataDir, 'freesurfer')
TR = 2
lowF = {'figureGround': 304,
        'figureGround_loc': 32,
        'restingState': 120}  # inverse of lower threshold for high-pass temporal filtering. Keys must match 'scanInfo' below.
smooth = 2  # std of spatial filtering Gaussian kernel (mm)

# scan info. WARNING! Make sure to construct these such that the registration scan is the first run of the first func scan type.
scanInfo = {'M012': {'191217': {'sessID': 339374,
                           'funcScans': {'figureGround': [18, 12, 14, 16, 22, 24, 26, 28],
                                         'figureGround_loc': [10, 20],      
                                         'restingState': [8]},
                           'anatScan': 4}}}
'''

import os, glob, shutil
import pandas as pd
import matplotlib.pyplot as plt

def preprocess(data):

    dataDir = data['dataDir']
    freesurferDir = data['freesurferDir']
    lowF = data['lowF']
    smooth = data['smooth']
    scanInfo = data['scanInfo']

    if not os.path.isdir(freesurferDir):
        os.makedirs(freesurferDir)

    subjects = list(scanInfo.keys())
    for subject in subjects:
        for session in scanInfo[subject].keys():
            sessID = scanInfo[subject][session]['sessID']
            for funcScanType in list(scanInfo[subject][session]['funcScans'].keys()):
                for scan in scanInfo[subject][session]['funcScans'][funcScanType]:

                    # get run (based on order in scan session, not order in scanInfo)
                    runs = sorted(scanInfo[subject][session]['funcScans'][funcScanType])
                    run = runs.index(scan) + 1

                    for distCor in ['noDistCor', 'TopUp', 'b0']:

                        print(
                            'SUBJECT: %s, SESSION: %s, SCAN NAME: %s, SCAN: %s, RUN: %s, DISTORTION CORRECTION: %s' % (
                            subject, session, funcScanType, scan, run, distCor))


                        inDir = os.path.join(dataDir, 'individual', subject, session, 'rawData')
                        outDir = os.path.join(dataDir, 'individual', subject, session, 'functional', funcScanType, 'run%02d/%s/preprocessing' % (run, distCor))
                        os.makedirs(outDir, exist_ok=True)

                        # make a copy of raw data in the output directory
                        inFile = glob.glob(os.path.join(inDir, '*%d.%02d*.nii' % (sessID, scan)))[0]
                        outFile = os.path.join(outDir, '1_rawData.nii')
                        if not os.path.exists(outFile):
                            shutil.copyfile(inFile, outFile)

                        # distortion correction
                        if distCor == 'TopUp':

                            topupFile = glob.glob(os.path.join(inDir, '*%d.%02d*.nii' % (sessID, scan + 1)))[0]

                            inFile = outFile  # new inFile is the output of previous operation
                            outFile = os.path.join(outDir, '2_distortionCorrected.nii.gz')  # set final filename for top up output

                            # if this file doesn't exist then run top up
                            if not os.path.isfile(outFile):
                                print('RUNNING DISTORTION CORRECTION (TOP UP)...')
                                os.system('python2 %s %s %s 90 270' % ('/mnt/HDD12TB/masterScripts/fMRI/fsl_TOPUP_call.py', inFile, topupFile))

                                # rename top-up corrected image
                                inFile = ('%s_topup_nsl46_TUcorrected.nii.gz' % inFile[:-4])
                                os.rename(inFile, outFile)

                            # delete unwanted top up output from inDir, outDir and master scripts dir
                            os.system('rm -rf *topup*')
                            os.system('rm -rf %s/*topup*' % inDir)
                            os.system('rm -rf %s/*topup*' % outDir)
                            os.system('rm -rf /mnt/HDD12TB/masterScripts/*topup*')
                            os.system('rm -rf /home/dave/*topup*')

                        elif distCor == 'b0':

                            if 'b0Scan' in scanInfo[subject][session].keys(): # only run if b0 map exists

                                inFile = glob.glob(os.path.join(inDir, '*%d.%02d*.nii' % (sessID, scan)))[0]
                                outFile = os.path.join(outDir, '2_distortionCorrected.nii.gz')  # set final filename for b0 output

                                if not os.path.isfile(outFile):

                                    # Preprocessing real field map image
                                    realFile = glob.glob(os.path.join(inDir, '*%d.%02d*e2*.nii' % (sessID, scanInfo[subject][session]['b0Scan'])))[0]
                                    os.system('fslmaths %s -div 1500 -mul 3.14 %s_rads.nii.gz' %(realFile, realFile[:-7]))
                                    os.system('fugue --loadfmap=%s_rads.nii.gz -m --savefmap=%s_rads_reg.nii.gz'  %(realFile[:-7], realFile[:-7])) # regularization

                                    # Preprocessing magnitude image
                                    magnitudeFile = glob.glob(os.path.join(inDir, '*%d.%02d*e1.nii' % (sessID, scanInfo[subject][session]['b0Scan'])))[0]
                                    os.system('bet %s %s_brain.nii.gz' %(magnitudeFile, magnitudeFile[:-7])) # brain extraction
                                    os.system('fslmaths %s_brain.nii.gz -ero %s_brain_ero.nii.gz' %(magnitudeFile[:-7], magnitudeFile[:-7])) # erode/remove one voxel from all edges

                                    # Fugue processing
                                    te = 0.00226 # echo time
                                    wfs = 40.284 # water fat shift
                                    acc = 1 # acceleration factor
                                    npe = 160 # phase encoding steps
                                    fstrength = 7 # field strength
                                    wfd_ppm = 3.4
                                    g_ratio_mhz_t = 42.57
                                    etl = npe/acc
                                    wfs_hz = fstrength * wfd_ppm * g_ratio_mhz_t
                                    ees = wfs / (wfs_hz * etl)

                                    os.system('prelude -a %s_brain_ero.nii.gz -p %s_rads_reg.nii.gz -u %s_rads_reg_unwrapped.nii.gz' %(magnitudeFile[:-7], realFile[:-7], realFile[:-7]))
                                    os.system('fslmaths %s_rads_reg_unwrapped.nii.gz -mul 1500 %s_rads_reg_unwrapped.nii.gz' %(realFile[:-7], realFile[:-7]))
                                    os.system('fugue -i %s --dwell=%f --loadfmap=%s_rads_reg_unwrapped.nii.gz --unwarpdir=y- --asym=%f --despike -u %s' %(inFile, ees, realFile[:-7], te, outFile))

                            else:
                                os.system('rm -rf %s' %outDir) # delete the new b0 outdir if b0 map not available

                        # make volume against which to register other scans
                        regScanType = list(scanInfo[subject][session]['funcScans'].keys())[0]
                        reg = scanInfo[subject][session]['funcScans'][regScanType][0]
                        regScans = sorted(glob.glob(os.path.join(dataDir, 'individual', subject, session, 'functional', regScanType, 'run%02d/%s/preprocessing/*.nii*' % (sorted(scanInfo[subject][session]['funcScans'][regScanType]).index(reg) + 1,distCor))))
                        if len(regScans) > 1:
                           regScan = regScans[1] # try and get distortion corrected
                           regScanTmean = '%s_Tmean.nii.gz' % regScan[:-7]
                        else:
                           regScan = regScans[0] # if no distortion corrected nifti available, use raw data
                           regScanTmean = '%s_Tmean.nii.gz' % regScan[:-4]
                        if not os.path.isfile(regScanTmean):
                            print('MAKING REGISTRATION VOLUME...')
                            os.system('mcflirt -in %s -out %s_tempRegRun.nii.gz' % (regScan, regScan[:-7]))
                            os.system('fslmaths %s_tempRegRun.nii.gz -Tmean %s' % (regScan[:-7], regScanTmean))
                            os.remove('%s_tempRegRun.nii.gz' % regScan[:-7])

                        # motion correction
                        inFile = os.path.join(outDir, '2_distortionCorrected.nii.gz')
                        if not os.path.isfile(inFile):
                            inFile = os.path.join(outDir, '1_rawData.nii')
                        outFile = os.path.join(outDir, '3_motionCorrected.nii.gz')
                        if not os.path.isfile(outFile):
                            print('RUNNING MOTION CORRECTION...')
                            os.system('mcflirt -plots -in %s -reffile %s -out %s' % (inFile, regScanTmean, outFile[:-7]))

                            # make plot
                            mcf = pd.read_table('%s.par' %(outFile[:-7]), sep = '  ', header=None, engine = 'python')

                            mcfRotation = mcf[[0, 1, 2]]
                            mcfRotation.columns = ['X', 'Y', 'Z']
                            rotationPlot = mcfRotation.plot.line()
                            rotationPlot.get_figure().savefig('%s_rotation.png' %(outFile[:-7]))

                            mcfTranslation = mcf[[3, 4, 5]]
                            mcfTranslation.columns = ['X', 'Y', 'Z']
                            translationPlot = mcfTranslation.plot.line()
                            translationPlot.get_figure().savefig('%s_translation.png' %(outFile[:-7]))

                            plt.close('all')

                        # motion outliers
                        molFile = inFile[:-7] + '_mol'
                        if not os.path.isfile(molFile):
                            print('IDENTIFYING MOTION OUTLIERS...')
                            os.system('fsl_motion_outliers -i %s -o %s --dvars' % (inFile, molFile))

                        # slice timing correction
                        inFile = outFile
                        outFile = os.path.join(outDir, '4_sliceTimingCorrected.nii.gz')
                        if not os.path.isfile(outFile):
                            print('RUNNING SLICE TIMING CORRECTION...')
                            os.system('slicetimer -i %s -o %s -r 2 -d 2' % (inFile, outFile))

                        # temporal filtering
                        inFile = outFile
                        outFile = os.path.join(outDir, '5_temporallyFiltered.nii.gz')
                        if not os.path.isfile(outFile):
                            print('RUNNING TEMPORAL FILTERING...')  # adapted for fsl 5.07 or later, which demeans the data when filtering
                            os.system('fslmaths %s -Tmean %s_Tmean.nii.gz' % (inFile, inFile[:-7]))
                            os.system('fslmaths %s -bptf %i -1 %s_filtnomean.nii.gz' % (inFile, lowF[funcScanType], inFile[:-7]))
                            os.system('fslmaths %s_Tmean.nii.gz -add %s_filtnomean.nii.gz %s' % (inFile[:-7], inFile[:-7], outFile))
                            os.remove('%s_filtnomean.nii.gz' % (inFile[:-7]))
                            os.remove('%s_Tmean.nii.gz' % (inFile[:-7]))

                        # spatial smoothing
                        inFile = outFile
                        outFile = os.path.join(outDir, '6_spatiallyFiltered.nii.gz')
                        if not os.path.isfile(outFile):
                            print('RUNNING SPATIAL FILTERING...')
                            os.system('fslmaths %s -s %f %s' % (inFile, smooth, outFile))

            # anatomical scans
            if not os.path.exists(os.path.join(dataDir, 'individual', subject, session, 'anatomical')):
                os.makedirs(os.path.join(dataDir, 'individual', subject, session, 'anatomical'))

            # brain extraction
            inFile = glob.glob(os.path.join(inDir, '*%d.%02d*.nii' % (sessID, scanInfo[subject][session]['anatScan'])))[0]
            outFileOriginal = os.path.join(dataDir, 'individual', subject, session, 'anatomical', 'anatomical.nii')
            outFileBET = os.path.join(dataDir, 'individual', subject, session, 'anatomical', 'anatomical_brain.nii.gz')
            if not os.path.exists(outFileBET):
                print('RUNNING ANATOMICAL BRAIN EXTRACTION...')
                shutil.copyfile(inFile, outFileOriginal)
                os.system('bet %s %s' %(outFileOriginal, outFileBET))


            # surface segmentation and registration
            if not os.path.isdir(os.path.join(freesurferDir, '%s%s' %(session,subject))):
                print('RUNNING SURFACE SEGMENTATION AND REGISTRATION...')
                os.system('bash /mnt/HDD12TB/masterScripts/callReconAll_bbregister.sh -s %s%s -a %s -d %s -f %s' %(session, subject, outFileOriginal, freesurferDir, regScanTmean))

    print('DONE.')
