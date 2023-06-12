"""
LPC encoding
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from scipy.signal import resample
from scipy.signal.windows import hann
from scipy.linalg import solve_toeplitz, toeplitz

from math import *


# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------


def blocks_decomposition(x, w, R=0.5):

    """
    Performs the windowing of the signal

    Parameters
    ----------

    x: numpy array
      single channel signal
    w: numpy array
      window
    R: float (default: 0.5)
      overlapping between subsequent windows

    Return
    ------

    out: numpy array
      block decomposition of the signal
    """
    T = len(w)
    S = 1 - R
    nb_blocks = int(np.ceil(len(x) / (T * S))) - 1
    blocks = np.zeros((nb_blocks, T))
    x_completed = np.concatenate((x, np.zeros(T - len(x) % T)))
    for i in range(nb_blocks):
        blocks[i] = x_completed[int(i * T * S) : int(i * T * S + T)] * w
        # print(blocks[i])
    return blocks


def blocks_reconstruction(blocks, w, signal_size, R=0.5):

    """
    Reconstruct a signal from overlapping blocks

    Parameters
    ----------

    blocks: numpy array
      signal segments. blocks[i,:] contains the i-th windowed
      segment of the speech signal
    w: numpy array
      window
    signal_size: int
      size of the original signal
    R: float (default: 0.5)
      overlapping between subsequent windows

    Return
    ------

    out: numpy array
      reconstructed signal
    """
    w[0] = 1e-10
    T = len(w)
    S = 1 - R
    nb_blocks = blocks.shape[0]
    signal = np.zeros(int(nb_blocks * T * S + T))
    passages = np.zeros(int(nb_blocks * T * S + T))
    for i in range(nb_blocks):
        signal[int(i * T * S) : int(i * T * S + T)] += blocks[i] * 1 / w
        passages[int(i * T * S) : int(i * T * S + T)] += 1
    signal = signal[:signal_size]
    passages = passages[:signal_size]
    signal = signal / passages
    return signal


# -----------------------------------------------------------------------------
# Linear Predictive coding
# -----------------------------------------------------------------------------


def autocovariance(x, k):

    """
    Estimates the autocovariance C[k] of signal x

    Parameters
    ----------

    x: numpy array
      speech segment to be encoded
    k: int
      covariance index
    """

    res = 0
    mean = np.mean(x)
    for i in range(len(x) - k):
        res += (x[i] - mean) * (x[i + k] - mean)
    # return res / (len(x) - k)
    return res  # les r_s sont en fait non normalisés (voir démo) !


def lpc_encode(x, p):

    """
    Linear predictive coding

    Predicts the coefficient of the linear filter used to describe the
    vocal track

    Parameters
    ----------

    x: numpy array
      segment of the speech signal
    p: int
      number of coefficients in the filter

    Returns
    -------

    out: tuple (coef, e, g)
      coefs: numpy array
        filter coefficients
      prediction: numpy array
        lpc prediction
    """
    r = np.array([autocovariance(x, k) for k in range(p)])
    b = np.array([autocovariance(x, k) for k in range(1, p + 1)])
    coef = solve_toeplitz(r, b)

    pred = np.zeros(len(x))
    for i in range(len(x)):
        s = 0
        for k in range(1, p + 1):
            s += coef[k - 1] * x[i - k]  # double boucle mais on peut faire mieux
        pred[i] = s

    e = np.linalg.norm(x - pred) ** 2

    return coef, pred


def lpc_decode(coefs, source):

    """
    Synthesizes a speech segment using the LPC filter and an excitation source

    Parameters
    ----------

    coefs: numpy array
      filter coefficients

    source: numpy array
      excitation signal

    Returns
    -------

    out: numpy array
      synthesized segment
    """
    p = len(coefs)
    N = len(source)
    signal = np.zeros(N + p)
    for i in range(p, N + p):
        signal[i] = np.dot(coefs, signal[i : i - p : -1]) + source[i - p]
    return signal[p:]  # on enlève les p premiers termes


# -----------------------------------------------------------------------------
# Pitch detection
# -----------------------------------------------------------------------------


def compute_cepstrum(x):

    """
    Computes the cepstrum of the signal

    Parameters
    ----------

    x: numpy array
      signal

    Return
    ------

    out: nunmpy array
      signal cepstrum
    """

    cepstrum = np.fft.ifft(np.log(np.abs(np.fft.fft(x))))
    return np.abs(cepstrum)


def cepstrum_pitch_detection(cepstrum, threshold, max_rate, sample_rate):

    """
    Cepstrum based pitch detection

    Parameters
    ----------

    cepstrum: numpy array
      cepstrum of the signal segment
    threshold: float
      threshold used to distinguish between voiced/unvoiced segments
    max_rate: float
      maximal pitch frequency
    sample_rate: float
      sample rate of the signal

    Return
    ------

    out: int
      estimated pitch. For an unvoiced segment, the pitch is set to zero
    """

    min_index = int(sample_rate / max_rate)
    quef_max = np.max(cepstrum[min_index:])
    quef_max_index = np.argmax(cepstrum[min_index:])
    mean = np.mean(cepstrum[min_index:])
    # print(min_index, mean, quef_max, quef_max_index)
    if mean < threshold * quef_max:
        pitch = sample_rate / quef_max_index
        return pitch
    else:
        return 0
