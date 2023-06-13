"""
Some functions used to plot the results
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *



# ------------------------------------------------------------------------------
# Display
# ------------------------------------------------------------------------------

def plot_signal(x, fs):

    """
    Plot the specified signal
    
    Parameters
    ----------
    
    x: numpy array
      signal
    fs: int
      sampling rate
    """

    t = 1e3*np.arange(x.size)/fs
    plt.plot(t, x)
    plt.xlabel('t [ms]')
    # plt.show()
    
    
def plot_spectrum(xhat, fs):

    """
    Plot the specified spectrum
    
    Parameters
    ----------
    
    xhat: numpy array
      Fourier transform of signal x
    fs: int
      sampling rate
    """

    n = xhat.size
    freq = np.fft.fftfreq(n, 1/fs)/1e3
    plt.plot(freq[:n//2], np.abs(xhat[:n//2]))
    plt.xlabel('f [kHz]')
    # plt.show()

def plot_cepstrum(cepstrum, fs):

    """
    Plot the specified cepstrum
    
    Parameters
    ----------
    
    cepstrum: numpy array
      cepstrum of signal x
    fs: int
      sampling rate
    """
    offset = 0.8 # correspond plutôt à un threshold qu'un offset

    n = cepstrum.size
    t = np.arange(n)/fs # quéfrences
    mask = (t[:n//2] >= offset)
    plt.plot(1e3*t[:n//2][mask], cepstrum[:n//2][mask])
    plt.xlabel('t [ms]')
    # plt.show()
    
    
def plot_reconstruction(signal, reconstruction, fs):

    """
    Plot a signal and its reconstruction on the same graph.
    
    Parameters
    ----------
    
    signal: numpy array
      original signal
    reconstruction: numpy array
      reconstruction of the signal
    fs: int
      sampling rate
    """

    t = 1e3*np.arange(signal.size)/fs
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].plot(t, signal, label='original')
    ax[0].plot(t, reconstruction, '--', label='reconstruction')
    ax[0].legend()
    ax[0].set_xlabel('t [ms]')
    
    err = signal - reconstruction
    ax[1].plot(t, err)
    ax[1].set_xlabel('t [ms]')
    
    plt.show()
    

    

