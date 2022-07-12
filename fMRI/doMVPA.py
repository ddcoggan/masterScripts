import numpy as np
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets.miscfx import remove_invariant_features
from scipy.stats import pearsonr


### Custom function definitions
def doMVPA(samples, mask):
    """ Main function, performs MVPA pipeline on filepaths list <samples>
    Assumes 1st fold of crossvalidation in 1st half of samples list, 2nd in 2nd
    """
    # Load dataset
    ds = fmri_dataset(samples = samples, mask = mask)
    ds = remove_invariant_features(ds)
    nConds = len(ds.samples)/2 # assuming split-half x-validation

    # Do correlations
    corrs = np.zeros([nConds, nConds])
    for i, s1 in enumerate(ds.samples[:nConds]):
        for j, s2 in enumerate(ds.samples[nConds:]):
            corrs[i,j] = pearsonr(s1,s2)[0]

    # Flatten corrs
    corrs = corrs.flatten()

    # Return
    return corrs
