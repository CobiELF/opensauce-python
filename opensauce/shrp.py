from __future__ import division

import numpy as np

"""
SHRP - a pitch determination algorithm based on Subharmonic-to-Harmonic Ratio.

Based on the shrp.m from voicesauce v1.25.  Per that file, for details of the
algorithm, please see Sun, X.,"Pitch determination and voice quality analysis
using subharmonic-to-harmonic ratio" To appear in the Proc. of ICASSP2002,
Orlando, Florida, May 13 -17, 2002.  For update information, please check
http://mel.speech.nwu.edu/sunxj/pda.htm.

This is not a full re-implementation of shrp.m: it only implements those
functions actually used by the voicesauce func_GetSHRP function.

"""

# XXX This is a work in progress, working from the bottommost functions up.


def zcr(x, dur):
    # func_Get_SHRP does not use this, because CHECK_VOICING is always 0.
    raise NotImplemented


def _pi_arange(width):
    return 2*np.pi*np.arange(width)/(width-1)

def _not_implemented():
    raise NotImplementedError

def _triangular(n):
    m = (n-1)/2
    res = np.arange(np.floor(m+1))/m
    return np.append(res, res[np.ceil(m)-1::-1])

window_funcs = dict(
    rect = lambda n: np.ones(n),
    tria = _triangular,
    hann = lambda n: 0.5*(1 - np.cos(_pi_arange(n))),
    hamm = lambda n: 0.54 - 0.46*np.cos(_pi_arange(n)),
    blac = lambda n: (0.42 - 0.5*np.cos(_pi_arange(n))
                      + 0.08*np.cos(2*_pi_arange(n))),
    kais = lambda n: _not_implemented(),
    )

def window(width, window_type, beta=None):
    """Generate a window function (1 dim ndarray) of length width.

    Given a window_type from the list 'rectangular', 'triangular', 'hanning',
    'hamming', 'blackman', 'kaiser', or at least the first four characters of
    one of those strings, return a 1 dimensional ndarray of floats expressing a
    window function of length 'width' using the 'window_type'.  'beta' is an
    additional input for the kaiser algorithm.  (XXX: kaiser is not currently
    implemented.)

    """
    algo = window_funcs.get(window_type[:4])
    if algo is None:
        raise ValueError(
            "Unknown window algorithm type {!r}".format(window_type))
    return algo(width)
