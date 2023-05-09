"""
LPC encoding
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

import scipy
from scipy.signal import resample
from scipy.signal.windows import hann 
from scipy.linalg import solve_toeplitz, toeplitz

from math import *

import utils


# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------

fs = 8e3 # sampling frequency

def make_window(L,fs):
    
    """
    Returns the Hamming window

    Parameters
    ----------
    
    L: window duration (s)
    fs: sample rate

    Return
    ------


    """

    size = int(L*fs)
    times = np.arange(size) / fs
    window = 0.54 - 0.46 * np.cos( 2*np.pi/L * times)
    return times, window

# times,window=make_window(0.02,1/(8e3))
# # utils.plot_signal(window,8e3)

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

    # window = np.concatenate( ( w, np.zeros(x.size - w.size) ) )
    # blocks = np.zeros( shape=( int((2 * x.size) / w.size - 1) , x.size) )

    # for k in range(blocks.shape[0]):
    #     blocks[k, :] = x * window
    #     window = np.roll( window, int(w.size * R) )

    win_extended = np.concatenate( ( w, np.zeros(x.size - w.size) ) ) # fenêtre allongée par des zéros pour faire mm taille que signal x
    interval = int(w.size * R) # nombre d'instants séparant le début de 2 blocs consécutifs, correspondant à la moitié de la taille de la fenêtre si R=0.5
    blocks = np.zeros( shape=( x.size // interval, w.size) )
    

    for k in range(blocks.shape[0]):
        big_block = x * win_extended
        blocks[k, :] = big_block[ k *  interval : k * interval + w.size ]
        win_extended = np.roll( win_extended, interval )


    
    return blocks

# Exemple de fenêtrage de Hamming d'un signal audio constant
L=1
size=int(L * fs)
win_dur = 0.2
wintimes,win =make_window( win_dur, fs )
# print(win.shape)

f_init = np.ones( size ,dtype=float )
# # utils.plot_signal(f_init,fs)

blocks = blocks_decomposition( f_init, win)

# for k in range(blocks[0].size):
#     utils.plot_signal(blocks[k, :], fs)
    
    
    
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

    # window = np.concatenate( ( w, np.ones(signal_size - w.size) ) ) # fenêtre étendue par des uns pour pouvoir les signaux fenêtrés par ses valeurs
    # f = np.zeros(signal_size, dtype=float)

    # for k in range(blocks.shape[0]-1):
    #     portion = blocks[k, :] / window
    #     # # utils.plot_signal(portion,fs)
    #     ind_fin_portion = (k+1) * int(w.size * R)
    #     f[:ind_fin_portion] = f[:ind_fin_portion] + portion[:ind_fin_portion]
    #     window = np.roll(window, int(w.size * R))
    
    # portion = blocks[-1, :] / window
    # f = f + portion

    f_pad = np.zeros(signal_size + w.size, dtype=float) # étendu avec des 0 pour fenêtrer la fin du signal réel

    for k in range(blocks.shape[0]):
        portion = blocks[k, :] / w
        interval = int(w.size * R)
        f_pad[k*interval : (k+1)*interval] = f_ext[k*interval : (k+1)*interval] + portion
    
    f = f_pad[:signal_size] # on enlève les 0 ajoutés au bord
    f[interval : ] = f[interval : ] * 0.5 # on moyenne les intervalles d'instant où on a ajouté 2 portions

    return f

#Exemple de reconstruction
f_init_ext = np.pad(f_init, (f_init.size, f_init.size))
blocks = blocks_decomposition(f_init_ext, win)
# utils.plot_signal(f_init_ext,fs)
f_rec=blocks_reconstruction(blocks, win, f_init_ext.size)
# utils.plot_signal(f_rec,fs)
    
    

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
    x_acov = np.zeros(x.size - k, dtype=float)
    x_mean = x.mean()
    for u in range(x_acov.size):
        x_acov[u] = (x[u] - x_mean) * (x[u + k] - x_mean)
    
    return x_acov.mean()
        
def convolve(a,b,n):
    """
    evaluate convolution of a and b in n
    a: numpy array
    b: numpy array
      assumed that b.size >= a.size
    """
    # conv = np.zeros(b.size, dtype=float) # estimations du signal non fenêtré
    # conv[0] = b[0]
    # for n in range(1, b.size):
    #     conv[n] = np.array([a[k] * b[n - 1 - k] for k in range(min(a.size, n))]).sum()
    
    # return conv
    return np.array([a[k] * b[n - 1 - k] for k in range(min(a.size, n))]).sum()

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
      musts be < x.size ?
      
    Returns
    -------
    
    out: tuple (coef, e, g)
      coefs: numpy array
        filter coefficients
      prediction: numpy array
        lpc prediction
    """
    v_acov = np.vectorize(lambda k: autocovariance(x, k))

    alphas = solve_toeplitz(c_or_cr=v_acov(np.arange(p)), b=v_acov(np.arange(1, p+1)))
     
    # x_estim=convolve(alphas, x) # estimations du signal non fenêtré
    x_estim = np.zeros(x.size, dtype=float) # estimations du signal non fenêtré
    x_estim[0] = x[0]
    for n in range(1, x.size):
        x_estim[n] = convolve(alphas,x,n)
        
    return alphas, x_estim

# test
alphas, x_estim = lpc_encode(np.array([0.,1.,2.,3.,4.,5.]), 3)
    
    
    
    
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
    s = np.zeros(source.size, dtype=float)
    s[0] = source[0]
    for n in range(1, s.size):
        s[n] = source[n] + convolve(coefs, s, n)
    
    return s

# test
lpc_decode(alphas, np.ones(5, dtype=float))
    



# -----------------------------------------------------------------------------
# Pitch detection
# -----------------------------------------------------------------------------

v_norm = np.vectorize(lambda z: np.linalg.norm(z)) # give array of norms from a array of complex

def compute_cepstrum(x):

    """
    Computes the cepstrum of the signal
    
    Parameters
    ----------
    
    x: numpy array
      signal
      
    Return
    ------
    
    out: numpy array
      signal cepstrum
    """    

    log_norm_dft_s = np.log(  v_norm( np.fft.fft( x ) ) )
    x_cepstrum = np.fft.ifft( log_norm_dft_s )

    return x_cepstrum

# test
# times = np.linspace(0,1,8000)
# x = np.cos( 2*np.pi*1e3 * times) + np.sin( 2*np.pi*3e3 * times)

# utils.plot_signal(x,fs)
# utils.plot_spectrum( np.fft.fft( x ), fs )
# utils.plot_cepstrum( compute_cepstrum( x ), fs )



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
    init_time = 4e-3 # we skip the cepstrum content before this value (in ms)
    init_ind = int(init_time * sample_rate)
    seg_ceps = cepstrum[init_ind:] # truncated cepstrum
  
    if  np.max(seg_ceps)  / np.mean(seg_ceps) > threshold:
        pitch_estim = np.argmax(seg_ceps)
    else:
        pitch_estim = 0.
    
    return pitch_estim

# def create_impulse_train(M: int, T: float) -> np.array:
#     """
#     create train of M impulsions with a pitch of T (s)"""
#     size = int( M * T )
#     e = np.zeros(size, dtype=float)
#     for k in range(M):
#         e = e + scipy.signal.unit_impulse(size, int( k * T ) )
    
#     return e

def create_impulse_train(sample_rate, size: int, T: float) -> np.array:
    """
    create train of impulsions with a size pf size and a pitch of T (s)"""
    M = int( size / sample_rate * T )
    e = np.zeros(size, dtype=float)
    for k in range(M):
        e = e + scipy.signal.unit_impulse(size, int( k * T ) )
    
    return e

# test
# T=5.
# e = create_impulse_train(10, T)
# utils.plot_signal(e, fs)
        

    

