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

fs = 8e3  # sampling frequency


def make_window(L, fs):
    """
    Returns the Hamming window

    Parameters
    ----------

    L: window duration (s)
    fs: sample rate

    Return
    ------
    """

    size = int(L*fs)  # number of elements of the window
    times = np.linspace(0, L, size)
    window = 0.54 - 0.46 * np.cos(2*np.pi/L * times)

    return window


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
      unité ? %

    Return
    ------

    out: numpy array
      block decomposition of the signal
    """

    # nombre d'instants séparant le début de 2 blocs consécutifs,
    interval = int(w.size * R)
    # correspondant à la moitié de la taille de la fenêtre si R=0.5

    # signal étendu pour fenêtrer le dernier morceau plus court que la fenêtre
    x_ext = np.pad(x, (0, interval))

    blocks = np.zeros(shape=(x.size // interval, w.size))

    for k in range(blocks.shape[0]):
        blocks[k, :] = x_ext[k * interval: k * interval + w.size] * w
        # utils.plot_signal(blocks[k, :],fs)

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

    interval = int(w.size * R)
    # étendu avec des 0 pour fenêtrer la fin du signal réel
    f_ext = np.zeros(signal_size + w.size, dtype=float)

    for k in range(blocks.shape[0]):
        decoded_block = blocks[k, :] / w
        f_ext[k*interval: k*interval + w.size] += decoded_block
        # utils.plot_signal(f_ext,fs)

    f = f_ext[:signal_size]  # on enlève les 0 ajoutés au bord
    # on moyenne les intervalles d'instants où il y a overlap
    f[interval: blocks.shape[0] * interval] *= 0.5

    return f


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


def convolve(a, b, n):
    """
    evaluate convolution of a and b in n
    a: numpy array
    b: numpy array
      assumed that b.size >= a.size
    """

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

    out: tuple (coef, prediction)
      coefs: numpy array
        filter coefficients
      prediction: numpy array
        lpc prediction
    """
    v_acov = np.vectorize(lambda k: autocovariance(x, k))

    alphas = solve_toeplitz(c_or_cr=v_acov(
        np.arange(p)), b=v_acov(np.arange(1, p+1)))

    x_estim = np.zeros(x.size, dtype=float)  # estimations du signal fenêtré
    x_estim[0] = x[0]
    for n in range(1, x.size):
        x_estim[n] = convolve(alphas, x, n)

    return alphas, x_estim


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

    out: numpy array
      signal cepstrum
    """

    # abs est vectorisée
    log_norm_dft_s = np.log(abs(np.fft.fft(x)))
    x_cepstrum = np.fft.ifft(log_norm_dft_s)

    return x_cepstrum


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
    init_time = 4e-3  # we skip the cepstrum content before this value (in ms)
    init_ind = int(init_time * sample_rate)
    seg_ceps = cepstrum[init_ind:]  # segmented cepstrum

    if np.max(seg_ceps) / np.mean(seg_ceps) > threshold:
        pitch_estim = np.argmax(seg_ceps) / sample_rate
    else:
        pitch_estim = 0.

    return pitch_estim


def approx_dirac(dir_size:int):
    times = np.linspace(-np.pi, np.pi, dir_size)
    return np.sinc(times)

def create_impulse_train(sample_rate, L: float, T: float) -> np.array:
    """
    create train of M dirac impulsions of duration L sampled at the frequency sample_rate and with a pitch of T (s)
    """
    size = int(L * sample_rate)  # nb of elts in array
    M = int(L / T)
    ind_bet_pitches = int(T * fs)  # nb indices entre 2 émissions de pitch
    e = np.zeros(size, dtype=float)
    for k in range(M):
        e[k * ind_bet_pitches] = 1.

    return e

def create_impulse_train_approx(sample_rate, L: float, T: float, dir_size:int) -> np.array:
    """
    create train of M sinc impulsions of duration L sampled at the frequency sample_rate and with a pitch of T (s)
    - dir_size is the dirac's 'width' in terms of number of array elements
    """
    size = int(L * sample_rate)  # nb of elts in array
    M = int(L / T)
    ind_bet_pitches = int(T * fs)  # nb indices entre 2 émissions de pitch
    e = np.zeros(size, dtype=float)
    for k in range(M):
        start_index = k * ind_bet_pitches # index of beginning of the dirac
        ind_left = size - start_index # nb of elements left
        dir_size_modif = min(ind_left, dir_size) # modified dirac size so it fits in the remaining elements 
        e[ start_index : start_index + dir_size_modif] = approx_dirac(dir_size)[: dir_size_modif]

    return e

def compute_cepstrum_dirac_impulse_train_theoric(fs, duration:float, T:float):
    """
    compute theorical cepstrum of impulse train of pitch T
    """ 
    size = int( duration * fs )
    M = int(duration / T)
    ind_bet_pitches = int(T * fs)  # nb indices entre 2 émissions de pitch
    e_cepst = np.zeros(size, dtype=float)
    for k in range(M):
        e_cepst[k * ind_bet_pitches] = 1 / ( k + 1 )

    return e_cepst


################################
# TESTS
################################

# test make_window OK
# window=make_window(0.02,fs)
# utils.plot_signal(window,fs)

# test block_decomposition OK
# Exemple de fenêtrage de Hamming d'un signal audio constant
# L = 1
# size = int(L * fs)
# win_dur = 0.2
# win = make_window(win_dur, fs)
# utils.plot_signal(win, fs)
# # print(win.shape)
# # f_init = np.ones( size ,dtype=float )
# f_init = np.linspace(0, L, size, dtype=float)
# plt.figure()
# utils.plot_signal(f_init, fs)
# blocks = blocks_decomposition(f_init, win)
# for i in range(10):
#     utils.plot_signal(blocks[i], fs)
# plt.show()

# test block_reconstruction OK
# f_rec = blocks_reconstruction(blocks, win, f_init.size)
# utils.plot_signal(f_rec,fs)

# test lpc_encode avec sinusoïde OK
# times = np.linspace(0,1,8000)
# x = np.cos(2*np.pi * 10 * times)
# utils.plot_signal(x, fs)
# alphas, x_estim = lpc_encode(x, 32)
# utils.plot_signal(x_estim, fs)

# test compute_cepstrum avec sinusoides OK
# times = np.linspace(0,1,8000)
# x = np.cos( 2*np.pi*1e3 * times) + np.sin( 2*np.pi*3e3 * times)
# utils.plot_signal(x,fs)
# utils.plot_spectrum( np.fft.fft( x ), fs )
# utils.plot_cepstrum( compute_cepstrum( x ), fs )

# test create_impulse_train OK
# e = create_impulse_train(fs, 10., 0.2)
# utils.plot_signal(e, fs)

# test lpc_encode avec train impulsions NOT OK
# e = create_impulse_train(fs, 10., 0.2)
# utils.plot_signal(e, fs)
# alphas, e_estim = lpc_encode(e, 320)
# utils.plot_signal(e_estim, fs)

# test lpc_encode avec train impulsions * hamming NOT OK
# w = make_window(1.,fs)
# utils.plot_signal(w,fs)
# e = create_impulse_train(fs, 1., 0.2)
# utils.plot_signal(e, fs)
# utils.plot_signal(w * e, fs)
# alphas, we_estim = lpc_encode(w * e, 32)
# utils.plot_signal(we_estim, fs)

# test lpc_encode avec sinc NOT BAD COULD BE BETTER
# fs2 = 160.  # si fs2 < p=32 => erreur 'negative dimension not allowed'
# inst = 16 # nb instants du sinc
# times = np.linspace(-np.pi, np.pi, inst)
# # x = np.pad(np.sinc(times), (0,int(fs2) - inst))
# x = np.pad(approx_dirac(inst), (0,int(fs2) - inst))
# utils.plot_signal(x, fs2)
# alphas, x_estim = lpc_encode(x, 32) # augmenter le nombre de coefficients diminue la qualité de la prédiction
# utils.plot_signal(x_estim, fs2)

# test lpc_encode avec modified dirac OK
# train_length = 160
# T = 0.015875
# d = create_impulse_train_approx(fs, train_length / fs, 20 / fs, 16)
# utils.plot_signal(d, fs)
# alphas, x_estim = lpc_encode(d, 32) 
# utils.plot_signal(x_estim, fs)

# test compute_cepstrum avec exemple internet 1 OK
# lien: https://support.ptc.com/help/mathcad/r9.0/fr/index.html#page/PTC_Mathcad_Help/example_cepstrum_and_complex_cepstrum.html
# times = np.linspace(0.,1.,int(fs))
# indexes = np.arange(500)
# fun = np.vectorize( lambda i: 100 / ( i + 100 ) * np.sin( i / 5 ) )
# x = fun(indexes)
# utils.plot_signal(x, fs)
# cepstr = compute_cepstrum(x)
# utils.plot_cepstrum(cepstr, fs)

# test compute_cepstrum avec exemple internet 2 OK
# lien: https://www.mathworks.com/help/signal/ug/cepstrum-analysis.html
# times = np.linspace(0.,0.1,int(fs))
# x = np.sin(2*np.pi*45.*times)
# utils.plot_signal(x,fs)
# y = x
# y[int(fs) // 2:] += x[int(fs) // 2:]
# utils.plot_signal(y,fs)
# cepstr = compute_cepstrum(y)
# utils.plot_cepstrum(cepstr, fs)

# test compute_cepstrum_dirac_impulse_train_theoric OK
# L=1.
# T=0.2
# e = create_impulse_train(fs, L,T)
# e_cepst = compute_cepstrum_dirac_impulse_train_theoric(fs, L,T)
# utils.plot_signal(e_cepst, fs)
# utils.plot_cepstrum(e_cepst, fs)

# test compute_cepstrum_dirac_impulse_train_theoric de taille d'1 fenêtre OK
# T= 0.015875
# e_cepst = compute_cepstrum_dirac_impulse_train_theoric(fs, 160 / fs, T)
# utils.plot_signal(e_cepst, fs)
# utils.plot_cepstrum(e_cepst, fs)

# test compute_cepstrum avec train impulsions dirac NOT OK
# L=1.
# T=0.2
# e = create_impulse_train(fs, L, T)
# utils.plot_signal(e,fs)
# e_cepst = compute_cepstrum(e)
# utils.plot_cepstrum(e_cepst, fs)

# test compute cepstrum avec modified impulse train et valeur expérimentale du pitch OK
# train_length = 160
# dir_size = 16
# L = train_length / fs
# # T = 50 / fs # marche pas avec 20 ni 18
# T= 0.015875 # marche avec vraie valeur du pitch
# d = create_impulse_train_approx(fs, L, T , dir_size)
# utils.plot_signal(d, fs)
# # utils.plot_spectrum(np.fft.fft(d),fs)
# d_cepstr = compute_cepstrum(d)
# utils.plot_cepstrum(d_cepstr, fs)
# d_cepstr_th = compute_cepstrum_dirac_impulse_train_theoric(fs, L, T)
# utils.plot_cepstrum(d_cepstr_th, fs)

# test pitch detection avec cepstre théorique d'un train d'impulsions OK
# L=10.
# T=0.3
# e = create_impulse_train(fs, L, T)
# utils.plot_signal(e,fs)
# e_cepst_th = compute_cepstrum_dirac_impulse_train_theoric(fs, L , T)
# T_found = cepstrum_pitch_detection(e_cepst_th, 0.8, 1000, fs)
# print(T, T_found)

# test pitch detection avec cepstre d'une sinusoide
# L=10.
# T=0.3
# size=int(L*fs)
# times = np.linspace(0,L,size)
# s = np.sin(2*np.pi / T * times)
# utils.plot_signal(s,fs)
# s_cepstr = compute_cepstrum(s)
# T_found = cepstrum_pitch_detection(s_cepstr, 0.8, 10000, fs)
# print(T, T_found)

# test pitch detection avec cepstre théorique d'un train d'impulsions et taille fenêtre et valeur exp du pitch NOT OK
# size=160
# L= size / fs
# T=0.015875
# # T= 0.01
# e = create_impulse_train(fs, L, T)
# utils.plot_signal(e,fs)
# e_cepst_th = compute_cepstrum_dirac_impulse_train_theoric(fs, L , T)
# T_found = cepstrum_pitch_detection(e_cepst_th, 0.0, 1000, fs)
# print(T, T_found)

# test pitch detection avec cepstre théorique d'un train d'impulsions et plus grande taille fenêtre et valeur exp du pitch OK
# size=200
# L= size / fs
# T= 0.001
# e = create_impulse_train(fs, L, T)
# utils.plot_signal(e,fs)
# e_cepst_th = compute_cepstrum_dirac_impulse_train_theoric(fs, L , T)
# T_found = cepstrum_pitch_detection(e_cepst_th, 0.8, 1000, fs)
# print(T, T_found)