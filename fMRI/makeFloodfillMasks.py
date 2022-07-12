from __future__ import division
from scipy import signal
import scipy.stats as sps
import numpy as np
import os, glob
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
from scipy.optimize import fmin

def makeFloodfillMasks(infile, nvox, coords, minthr, outfile, mask=None):
    """
    Script uses flood fill algorithm to generate masks comprising a specified number
    of spatially contiguous voxels clustered around a given seed point.

    Specify the 3D input statistical volume using the --infile flag.

    Specify the desired number of voxels in your mask using the --nvox flag.

    The --coord flag is used to specify the x y z voxel (not mm) co-ordinates of the
    desired seed point (e.g. co-ordinate of the peak voxel in your desired cluster).

    The script saves out a mask of the identified cluster to the filepath specified
    with the --outfile flag.  In addition, the script will print to the command
    window details of the desired and actual size of the mask and the value it is
    thresholded at.  Desired and actual mask sizes should be the same if the
    algorithm was able to find an optimal solution.

    You may optionally provide a mask with the --mask flag to be applied to the
    statistical volume before performing the clustering.  For instance, you could
    use this to prevent the algorithm growing clusters outside of a desired region.

    You may optionally specify a minimum acceptable value to threshold the cluster
    at using the --thr flag (default is Z = 1.8); if the algorithm identifies a
    threshold for the cluster below this, the threshold is set to this value
    regardless of the number of voxels requested.

    Example usage:
    python defineROICluster_signed.py --infile myZstat.nii.gz --nvox 100 --coord 10 20 30
    --outfile myClusterMask.nii.gz --mask myInitialMask.nii.gz --thr 2.3
    """

    ### Custom function definitions
    def fminerr(thr, vol, nvox, x, y, z):
        """ Thresholds <vol> at <thr>.  Then uses scipy.ndimage.label to label
        structures (i.e. voxel clusters) in thresholded vol.
        Calculates the number of voxels in cluster that contains coord
        @ <x,y,z>, returns squared error between this and desired cluster size <nvox>
        """
        thr_vol = vol.copy()
        thr_vol[thr_vol < thr] = 0  # threshold
        labelled_array, n_features = label(thr_vol, structure=generate_binary_structure(vol.ndim, vol.ndim))  # label
        # (Uses custom scructuring element that considers all diagonal elements to be neighbours)
        return ((labelled_array == labelled_array[x, y, z]).sum() - nvox) ** 2  # return error

    ### New fminerr by SJ ###
    ## NB: THIS IS A SIGNED fminerr FUNCTION: +/-
    def fminerr2(thr, vol, nvox, x, y, z):
        """ Thresholds <vol> at <thr>.  Then uses scipy.ndimage.label to label
        structures (i.e. voxel clusters) in thresholded vol.
        Calculates the number of voxels in cluster that contains coord
        @ <x,y,z>, returns squared error between this and desired cluster size <nvox>
        """
        thr_vol = vol.copy()
        thr_vol[thr_vol < thr] = 0  # threshold
        labelled_array, n_features = label(thr_vol, structure=generate_binary_structure(vol.ndim, vol.ndim))  # label
        # (Uses custom scructuring element that considers all diagonal elements to be neighbours)
        return (nvox - (labelled_array == labelled_array[x, y, z]).sum())  # return error

    ### Begin main script
    x, y, z = coords

    # Load infile
    img = nib.load(infile)
    vol = img.get_fdata()
    hdr = img.header

    # Apply mask if provided
    if mask is not None:
        maskvol = nib.load(mask).get_fdata().astype(bool)  # load into boolean mask
        vol *= maskvol  # apply mask

    # Determine value at seed voxel
    x0 = vol[x, y, z]

    # Get a temp, sorted list of the values in the volume for a list of thresholds to check...
    tmpvol = vol[vol > 0]
    tmpvol.sort()
    tmpvol = tmpvol[::-1]

    # Initialise a value that we know is too big
    old_out = nvox
    # Set a flag to check whether we find a threshold
    success = False
    # Start at the (nvox-1)-th value, any less and the cluster will be too small


    for i in range(nvox - 1, tmpvol.shape[0]):
        # Check the difference in cluster size NB: THIS IS SIGNED +/-
        out = fminerr2(tmpvol[i], vol, nvox, x, y, z)
        # If we've gone too far, check which is nearest to the desired cluster size:
        if out < 0:
            if -out > old_out:
                new_thresh = tmpvol[i - 1]
            else:
                new_thresh = tmpvol[i]
            # We don't need to check any more, so break out, after setting the  success flag
            success = True
            break
        # Keep a record of the previous error
        old_out = out

    # Assign our new threshold value to the right variable
    if success:
        xopt = new_thresh
    else:
        # Set the threshold to the smallest value in the mask volume
        xopt = tmpvol[-1]

    # Check that xopt is not below acceptable threshold
    # If it is, set xopt to acceptable threshold
    if xopt < minthr:
        xopt = minthr
        import warnings
        warnmsg = 'Thr too low, setting to %f' % minthr
        warnings.warn(warnmsg, UserWarning)

    # Run labelling algorithm once more with xopt to get final cluster
    thr_vol = vol.copy()
    thr_vol[thr_vol < xopt] = 0.0
    labelled_array, n_features = label(thr_vol, structure=generate_binary_structure(vol.ndim, vol.ndim))
    clustersize = (labelled_array == labelled_array[x, y, z]).sum()  # actual clustersize
    """ clustersize should == nvox if fmin has found an optimal xopt.  If not,
    fmin was unable to find an optimal solution - this is worth knowing!"""

    # Map back to volume and save out
    maskim = labelled_array == labelled_array[x, y, z]  # cluster as mask
    hdr['cal_min'] = 0
    hdr['cal_max'] = 1
    nii2save = nib.Nifti1Image(dataobj=maskim, affine=None, header=hdr)
    nib.save(nii2save, outfile)

    # Print results out to text file
    return clustersize, xopt



