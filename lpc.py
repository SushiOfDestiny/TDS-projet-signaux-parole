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

import utils


# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------

fs = 8e3 # sampling frequency

def make_window(T,dt):
    
    """
    Returns the Hamming window

    Parameters
    ----------
    
    T: window duration
    dt: duration of a sample

    Return
    ------


    """

    size = int(T/dt)
    times = np.arange(size) * dt
    window = 0.54 - 0.46 * np.cos( 2*np.pi/T * times)
    return times, window

# times,window=make_window(0.02,1/(8e3))
# utils.plot_signal(window,8e3)

def blocks_decomposition(x, w, R = 0.5):

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
      unité ? %
    
    Return
    ------
    
    out: numpy array
      block decomposition of the signal
    """

    window = np.concatenate( ( w, np.zeros(x.size - w.size) ) )
    blocks = np.zeros( shape=( int((2 * x.size) / w.size - 1) , x.size) )

    for k in range(blocks.shape[0]):
        blocks[k, :] = x * window
        window = np.roll( window, int(w.size * R) )
    
    return blocks

# Exemple de fenêtrage de Hamming d'un signal audio constant
T=1
dt=1/fs
size=int(T/dt)

wintimes,win =make_window( 0.2, dt )
# print(win.shape)

f_init = np.ones( size ,dtype=float )
# utils.plot_signal(f_init,fs)

blocks = blocks_decomposition( f_init, win)
    
    
    
def blocks_reconstruction(blocks, w, signal_size, R = 0.5):

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

    window = np.concatenate( ( w, np.ones(signal_size - w.size) ) ) # fenêtre étendue par des uns pour pouvoir les signaux fenêtrés par ses valeurs
    f = np.zeros(signal_size, dtype=float)

    for k in range(blocks.shape[0]-1):
        portion = blocks[k, :] / window
        # utils.plot_signal(portion,fs)
        ind_fin_portion = (k+1) * int(w.size * R)
        f[:ind_fin_portion] = f[:ind_fin_portion] + portion[:ind_fin_portion]
        window = np.roll(window, int(w.size * R))
    
    portion = blocks[-1, :] / window
    f = f + portion
    
    return f

#Exemple de reconstruction
f_init_ext = np.pad(f_init, (f_init.size, f_init.size))
blocks = blocks_decomposition(f_init_ext, win)
utils.plot_signal(f_init_ext,fs)
f_rec=blocks_reconstruction(blocks, win, f_init_ext.size)
utils.plot_signal(f_rec,fs)
    
    

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
    
    # TODO
        
    
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
    
    # TODO
    
    
    
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

    # TODO



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

    # TODO



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

    # TODO

