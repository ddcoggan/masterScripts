from __future__ import division
from scipy import signal
import scipy.stats as sps
import numpy as np
import os, glob
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
from scipy.optimize import fmin

def doGLM(designMatrix=None, fMRIdata=None, contrasts=None):
    # Make sure the design matrix has the right orientation
    if designMatrix.shape[1] > designMatrix.shape[0]:
        designMatrix = designMatrix.transpose()

    # get timeseries dimensions
    if fMRIdata.ndim > 2:
        X, Y, Z, T = fMRIdata.shape
    else:
        T = fMRIdata.shape[1]

    # flatten timeseries
    timeseries = fMRIdata.reshape((-1, T))
    nVoxels = timeseries.shape[0]
    C = designMatrix.shape[1]

    # degress of freedom is the number of timepoints minus the number of regressors
    df = T - C

    # Calculate the dot product of the transposed design matrix
    # and the design matrix and invert the resulting matrix.
    tmp1 = np.linalg.pinv(designMatrix.T.dot(designMatrix))

    # Now calculate the dot product of the above result and the
    # transposed design matrix
    tmp2 = tmp1.dot(designMatrix.T)

    # Pre-allocate variables
    beta = np.zeros((nVoxels, C))
    residual = np.zeros((nVoxels, T))
    model = np.zeros((nVoxels, T))
    R = np.zeros(nVoxels)
    MSE = np.zeros(nVoxels)
    se_estimate = np.zeros(nVoxels)
    contrast_beta = []
    se_contrast_beta = []
    contrast_t = []

    if contrasts is not None:
        if type(contrasts) is list:
            nContrasts = 1
            contrast_beta = np.zeros(nVoxels)
            se_contrast_beta = np.zeros(nVoxels)
            contrast_t = np.zeros(nVoxels)
        else:
            nContrasts = contrasts.shape[0]
            contrast_beta = np.zeros((nVoxels, nContrasts))
            se_contrast_beta = np.zeros((nVoxels, nContrasts))
            contrast_t = np.zeros((nVoxels, nContrasts))

    # run voxel-wise analysis
    for v in range(nVoxels):

        beta[v] = np.dot(tmp2, timeseries[v].T)
        model[v] = np.dot(designMatrix, beta[v])
        residual[v] = (timeseries[v] - model[v])
        R[v] = np.sqrt(model[v].var() / timeseries[v].var())
        MSE[v] = sum(np.power(residual[v], 2)) / df
        if contrasts is not None:
            contrast_beta[v] = np.dot(beta[v], contrasts)
            se_contrast_beta[v] = np.sqrt(MSE[v] * contrasts.dot(tmp1).dot(contrasts.T))
            contrast_t[v] = contrast_beta[v] / se_contrast_beta[v]

    if fMRIdata.ndim > 2:
        beta = beta.reshape((X, Y, Z, C))
        residual = residual.reshape((X, Y, Z, T))
        model = model.reshape((X, Y, Z, T))
        R = R.reshape((X, Y, Z))
        MSE = MSE.reshape((X, Y, Z))
        se_estimate = se_estimate.reshape((X, Y, Z))

        if contrasts is not None:
            contrast_beta = contrast_beta.reshape((X, Y, Z, nContrasts))
            se_contrast_beta = se_contrast_beta.reshape((X, Y, Z, nContrasts))
            contrast_t = contrast_t.reshape((X, Y, Z, nContrasts))

    return beta, model, residual, R, MSE, se_estimate, contrast_beta, se_contrast_beta, contrast_t
