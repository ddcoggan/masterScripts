from __future__ import division
from scipy import signal
import scipy.stats as sps
import numpy as np
import os, glob

def hrf(t,
        gamma='double',
        peak_delay=6,
        under_delay=16,
        peak_disp=1,
        under_disp=1,
        p_u_ratio=6,
        normalize=True,
        ):
    """ SPM HRF function from sum of two gamma PDFs

    Modified by DDC to include option for removing undershoot (single gamma)

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.

    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
    `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
    and dispersion `under_disp`, and divided by the `p_u_ratio`).

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF
    gamma : string, optional
        double gamma includes undershoot, single does not
    peak_delay : float, optional
        delay of peak
    peak_disp : float, optional
        width (dispersion) of peak
    under_delay : float, optional
        delay of undershoot
    under_disp : float, optional
        width (dispersion) of undershoot
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.  SPM does this
        by default.

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    See ``spm_hrf.m`` in the SPM distribution.
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    if gamma == 'double':
        undershoot = sps.gamma.pdf(pos_t,
                                   under_delay / under_disp,
                                   loc=0,
                                   scale=under_disp)
        hrf[t > 0] = peak - undershoot / p_u_ratio
    else:
        hrf[t > 0] = peak
    if not normalize:
        return hrf
    return hrf / np.max(hrf)


