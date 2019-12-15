"""Harmonic amplitude estimation
"""

# Licensed under Apache v2 (see LICENSE)

from __future__ import division

import numpy as np
from scipy import fftpack
from .optimization import fminsearchbnd
from .textgrid import TextGrid


def get_A1_A2_A3(y, fs, F0, F1, F2, F3, variables, textgridfile=None):
    N_periods = variables["Nperiods"]
    sampleshift = fs/1000 * variables["frameshift"]

    A1 = np.zeros(len(F0, 1)) * np.nan
    A2 = np.zeros(len(F0, 1)) * np.nan
    A3 = np.zeros(len(F0, 1)) * np.nan

    if textgridfile == None: # if no textgrid
        for k in range(len(F0)):
            ks = round(k * sampleshift)

            if (ks <= 0 or ks > len(y)): continue

            F0_curr = F0(k)

            if (np.isnan(F0_curr) or F0_curr == 0): continue

            N0_curr = 1 / F0_curr * fs

            ystart = round(ks - N_periods/2*N0_curr)
            yend = round(ks + N_periods/2*N0_curr) - 1

            if ystart <= 0 or yend > len(y): continue

            yseg = y[ystart:yend]

            if (np.isnan(F1[k]) or np.isnan(F2[k]) or np.isnan(F3[k]) or np.isnan(N0_curr)):
                A1_, fmax = ana_get_magnitude_max(yseg, F1[k], fs, 8192) #fmax unused?
                A2_, fmax = ana_get_magnitude_max(yseg, F2[k], fs, 8192)
                A3_, fmax = ana_get_magnitude_max(yseg, F3[k], fs, 8192)

                A1[k] = A1_
                A2[k] = A2_
                A3[k] = A3_

        return A1, A2, A3
    else:
        tbuffer = variables["tbuffer"]
        fp = open(variables["TextgridIgnoreList"])
        ignorelabels = np.loadtxt(fp, '\s', delimiter=",")
        ignorelabels = ignorelabels[1]
        fp.close()

        tg = TextGrid()
        tg.read(textgridfile)
        pass



def ana_get_magnitude_max(x, Fx, fs, fftlen):
    if np.isnan(Fx):
        return np.nan, np.nan

    else:
        length = len(x)
        hamlen = min(fftlen, length) #unused?
        factor = 1
        X = fftpack.fft(x, fftlen)
        for idx in range(len(X)):
            if X[idx] == 0:
                X[idx] = 0.000000001 # guard against log(0)

        X = 20*np.log10(factor*abs(X[1:fftlen/2, :]))
        fstep = fs/fftlen
        lowf = Fx - 0.1*Fx
        if lowf < 0: lowf = 0
        highf = Fx + 0.1*Fx
        if highf > fs/2 - fstep: highf = fs/2-fstep

        M = np.zeros(len(Fx))
        fmax = np.zeros(len(Fx))
        for cnt in range(len(Fx)):
            m, pos = max(X[1+round(lowf[cnt]/fstep):1+round(highf[cnt]/fstep), :])
            fmax[cnt] = (pos-1+round(lowf(cnt)/fstep))*fstep
            M[cnt] = m

        return 


# get maximal spectral magnitude of a signal x around position Fx in dB
def get_harmonics(data, f_est, fs):
    df = 0.1
    df_range = round(f_est * df)

    f_min = f_est - df_range
    f_max = f_est + df_range

    f = lambda x: est_max_val(x, data, fs)

    x, val, _, _ = fminsearchbnd(f, f_est, f_min, f_max)

    h = -val
    fh = x
    return h, fh

def est_max_val(x, data, fs):
    n = list(range(len(data)))
    v = np.exp(-1.j * 2 * np.pi * x * n/fs)
    val = -1 * 20 * np.log10(abs(data * np.transpose(v)))
    return val

def correction_iseli_i(f, F_i, B_i, fs):
    """Return the i-th correction (dB) to the harmonic amplitude using the
       algorithm developed by Iseli and Alwan. Note this correction should be
       *subtracted* from the amplitude. The total correction is computed by
       subtracting all of the i-th corrections from the amplitude.

       Reference -- M. Iseli and A. Alwan, An improved correction formula for
       the estimation of harmonic magnitudes and its application to open
       quotient estimation.

    Args:
        f      - frequency/harmonic to be corrected (Hz) [NumPy vector]
        F_i    - i-th formant frequency (Hz) [NumPy vector]
        B_i    - i-th formant bandwidth (Hz) [NumPy vector]
        fs     - sampling frequency (Hz)
    Returns:
        corr_i - i-th correction to harmonic amplitude in dB [NumPy vector]
    """
    # These variable names are from the Iseli-Alwan paper
    # Normalize frequencies to sampling frequency
    r_i = np.exp(- np.pi * B_i / fs)
    omega_i = 2 * np.pi * F_i / fs
    omega  = 2 * np.pi * f / fs

    # Factors needed to compute correction
    numerator_sqrt = r_i**2 + 1 - 2 * r_i * np.cos(omega_i)
    denom_factor1 = r_i**2 + 1 - 2 * r_i * np.cos(omega_i + omega)
    denom_factor2 = r_i**2 + 1 - 2 * r_i * np.cos(omega_i - omega)

    # Correction in the z-domain
    # corr = 10 * log10(numerator_sqrt**2 / (denom_factor1 * denom_factor2))
    # Formula simplifies due to logarithm arithmetic
    corr_i = 20 * np.log10(numerator_sqrt) - 10 * np.log10(denom_factor1) - 10 * np.log10(denom_factor2)

    return corr_i

def bandwidth_hawks_miller(F_i, F0):
    """Return formant bandwidth estimated from the formant frequency and the
       fundamental frequency

       For each formant frequency, estimate the bandwidth from a 5th order
       power series with coefficients C1 or C2 depending on whether the
       frequency is less or greater than 500 Hz, then scale by a factor that
       depends on the fundamental frequency.

       Reference -- J.W. Hawks and J.D. Miller, A formant bandwidth estimation
       procedure for vowel synthesis, JASA, Vol. 97, No. 2, 1995

    Args:
        F_i - i-th formant frequency (Hz) [NumPy vector]
        F0  - Fundamental frequency (Hz) [NumPy vector]
    Returns:
        B_i - Bandwidth corresponding to i-th formant (Hz) [NumPy vector]
    """
    # Bandwidth scaling factor as a function of F0,
    # to accommodate the wider bandwidths of female speech
    S = 1 + 0.25 * (F0 - 132) / 88

    # Coefficients C1 (for F_i < 500 Hz) and C2 (F_i >= 500 Hz)
    #
    # There are 6 coefficients for each term in a 5th order power series
    C1 = np.array([165.327516, -6.73636734e-1, 1.80874446e-3, -4.52201682e-6, 7.49514000e-9, -4.70219241e-12])
    C2 = np.array([15.8146139, 8.10159009e-2, -9.79728215e-5, 5.28725064e-8, -1.07099364e-11, 7.91528509e-16])

    # Construct matrix that is a 5th order power series
    # of the formant frequency
    F_i_mat = np.vstack((F_i**0, F_i**1, F_i**2, F_i**3, F_i**4, F_i**5))

    # Construct mask for formant frequency < 500 Hz
    #
    # Set NaN values in F_i to 0, so that when we do the boolean operation
    # F_i < 500, it doesn't throw a runtime error about trying to do boolean
    # operations on NaN, which is an invalid value.
    # It doesn't matter what value we replace NaN with, because regardless of
    # the values in the Boolean mask corresponding to F_i = NaN, these will
    # get multiplied by NaN again in the formant bandwidth estimation below
    # and NaN * False = NaN * True = NaN. Any arithmetic operation on NaN
    # results in another NaN value.
    F_i_dummy = F_i.copy()
    F_i_dummy[np.isnan(F_i_dummy)] = 0
    # Tile/repeat the mask for each of the 6 terms in the power series
    mask_less_500 = np.tile(F_i_dummy < 500, (len(C1), 1))

    # Formant bandwidth estimation
    #
    # For each formant frequency, estimate the bandwidth from a 5th order power
    # series with coefficients C1 or C2 depending on whether the frequency is
    # less or greater than 500 Hz, then scale by a factor that depends on the
    # fundamental frequency
    B_i = S * (np.dot(C1, F_i_mat * mask_less_500) + np.dot(C2, F_i_mat * np.logical_not(mask_less_500)))

    return B_i
