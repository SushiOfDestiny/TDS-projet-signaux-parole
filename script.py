"""
LPC encoding
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from scipy.signal import resample, lfilter
from scipy.signal.windows import hann 

from scipy.linalg import solve_toeplitz, toeplitz
 
from math import *
from utils import *
from lpc import *



if __name__ == '__main__':


    [sample_rate, speech] = wavfile.read('./audio/speech.wav')
    speech = np.array(speech)

    # normalize the speech
    speech = 0.9*speech/max(abs(speech))

    # resampling to 8kHz
    target_sample_rate = 8000
    target_size = int(len(speech)*target_sample_rate/sample_rate)
    speech = resample(speech, target_size)
    sample_rate = target_sample_rate
    
    # Record resampled signal
    wavfile.write("./results/speech_resampled.wav", sample_rate, speech)

    #  window
    w = hann(floor(0.02*sample_rate), False)
    # test OK
    # utils.plot_signal(w, sample_rate)
    
    # Block decomposition
    blocks = blocks_decomposition(speech, w, R = 0.5)
    n_blocks, block_size = blocks.shape
    
    # --------------------
    # Encoding
    # --------------------
    
    p = 32 # LPC filter: number of coefficients
    threshold = 0.8 # Parameters for pitch detection
    max_rate = 200 # Maximal pitch frequency [Hz]
        
    lpc_coefs, pitches, gain, errors = [], [], [], []
    for block in blocks:

        # Linear predictive coding
        coefs, prediction = lpc_encode(block, p)
        error = block - prediction
        
        # Pitch detection
        cepstrum = compute_cepstrum(block)
        # plot_cepstrum(cepstrum, sample_rate)
        pitch = cepstrum_pitch_detection(cepstrum, threshold, max_rate, 
         sample_rate)
        
        # Update
        lpc_coefs.append(coefs)
        pitches.append(pitch)
        gain.append(np.std(error))
        errors.append(error)
        
    # --------------------
    # Decoding
    # --------------------
    
    blocks_decoded = []
    for coefs, pitch, g in zip(lpc_coefs, pitches, gain):
    
        # Creates an excitation signal for a non-voiced speech
        noise = g*np.random.randn(block_size)

        if(pitch != 0):
        
            # create an excitation signal based upon a train of
            # impulses of the same length as the current block

            # nb_impulses = int( block_size / T )
            # source = g * create_impulse_train(nb_impulses, T)
            source = g * create_impulse_train(sample_rate, block_size / sample_rate, pitch)
            # utils.plot_signal(source, sample_rate)
            
            
        else:
            source = noise
    
            block_decoded = lpc_decode(coefs, w*source)
        blocks_decoded.append(block_decoded)
        
    blocks_decoded = np.array(blocks_decoded) # problème ici
    decoded_speech = blocks_reconstruction(blocks_decoded, w, speech.size, 
      R = 0.5)
    
    np.nan_to_num(decoded_speech, copy=False) # effacement des Nan
    
    wavfile.write("./results/decoded_speech.wav", sample_rate, decoded_speech)



