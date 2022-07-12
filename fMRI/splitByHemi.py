import os, glob
import nibabel as nib
import numpy as np

def splitByHemi(input='None'):
    inputDir = os.path.dirname(input)
    inputName = os.path.basename(input).split(sep='.')[0]

    nii = nib.load(input)
    dat = nii.get_fdata()
    affine = nii.affine
    header = nii.get_header()
    xdim = dat.shape[0]


    if xdim % 2 == 1:
        xdim = xdim - 1

    # Split off left
    left_dat = np.zeros(dat.shape)

    left_dat[int(xdim / 2):, :, :] = dat[int(xdim / 2):, :, :]

    # Split off right
    right_dat = np.zeros(dat.shape)
    right_dat[:int(xdim / 2), :, :] = dat[:int(xdim / 2), :, :]

    # Save
    nib.Nifti1Image(left_dat, affine, header).to_filename(os.path.join(inputDir, inputName + '_lh.nii.gz'))
    nib.Nifti1Image(right_dat, affine, header).to_filename(os.path.join(inputDir, inputName + '_rh.nii.gz'))

