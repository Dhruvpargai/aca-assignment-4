import os
import math
import soundfile as sf

import numpy as np
import scipy as sp
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

REPLACE_L2_BY_L1 = True  # using L1 distance gives better results than L2 distance in key detection


def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

def compute_spectrogram(xb,fs):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(sp.fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) #let's be pedantic about normalization

    f = np.arange(0, X.shape[0])*fs/(xb.shape[1])
    
    return (X,f)

##############
### PART A ###
##############

def get_spectral_peaks(X, fs):
    top_twenty_freq_bins = np.zeros((X.shape[0], 20))
    for i, block in enumerate(X):
        peak_indices, _ = find_peaks(block)
        num_peaks = min(len(peak_indices), 20) # If 20 peaks don't exist, return as many as possible

        top_twenty_freq_bins[i] = peak_indices[np.argpartition(block[peak_indices], -num_peaks)[-num_peaks:]]
        top_twenty_freq_bins[i] *= fs / X.shape[1]

    return top_twenty_freq_bins
    
def estimate_tuning_freq(x, blockSize, hopSize, fs):
    equal_temperament_pitches = np.zeros(48) 
    for i in range(48):
        equal_temperament_pitches[i] = 220.0 * 2 ** (i / 12.0) # A3 -> A7

    xb, time_in_sec = block_audio(x, blockSize, hopSize, fs)
    X, fInHz = compute_spectrogram(xb, fs)
    top_twenty_freq_bins = get_spectral_peaks(X, fs)
    error_in_cents = np.zeros(top_twenty_freq_bins.flatten().shape)
    
    for i, freq in enumerate(top_twenty_freq_bins.flatten()):
        if freq > 0:
            nearest_pitch = np.argmin(np.abs(freq - equal_temperament_pitches))
            error_in_cents[i] = 1200 * np.log2(freq / equal_temperament_pitches[nearest_pitch])

    hist, bin_edges = np.histogram(error_in_cents, 100, (-50, 50))
    mode_of_delta_c = bin_edges[np.argmax(hist)]
    return 2 ** (mode_of_delta_c / 1200) * 440

def extract_pitch_chroma(X: np.ndarray, fs: int, tfInHz: float = 440.0):
    """
    Compute the pitch chroma from spectrogram
    
    Parameters
    ----------
    X: np.ndarray
        (freq_bins, num_blocks)
    fs: int
        sampling rate
    tfInHz: float
        tuning frequency
        
    Return
    ----------
    np.ndarray
        (12, num_blocks)
    """
    
    # initialization according to tuning frequency
    f_C5 = tfInHz * 2 ** (3 / 12)
    f_C3 = f_C5 / 4
    
    iNumPitchesPerOctave = 12
    iNumOctaves = 3
    
    num_freq_bins = X.shape[0]
    H = np.zeros([iNumPitchesPerOctave, num_freq_bins])
    
    # for each pitch class i create weighting factors in each octave j
    f_mid = f_C3
    for i in range(0, iNumPitchesPerOctave):
        afBounds = np.array([2**(-1 / (2 * iNumPitchesPerOctave)), 2**(1 / (2 * iNumPitchesPerOctave))]) * f_mid * 2 * (num_freq_bins - 1) / fs
        for j in range(0, iNumOctaves):
            iBounds = np.array([math.ceil(2**j * afBounds[0]), math.ceil(2**j * afBounds[1])])
            H[i, range(iBounds[0], iBounds[1])] = 1 / (iBounds[1] - iBounds[0] + 1)

        # increment to next semi-tone
        f_mid = f_mid * 2**(1 / iNumPitchesPerOctave)
        
    pitch_chroma = H.dot(X ** 2)
    
    norm = pitch_chroma.sum(axis=0, keepdims=True) if REPLACE_L2_BY_L1 else (pitch_chroma ** 2).sum(axis=0, keepdims=True) ** 0.5
    norm[norm == 0] = 1
    pitch_chroma = pitch_chroma / norm
        
    return pitch_chroma

def detect_key(x, blockSize, hopSize, fs, bTune=False):

    # key names
    cKeyNames = np.array(['C Maj', 'C# Maj', 'D Maj', 'D# Maj', 'E Maj', 'F Maj', 'F# Maj', 'G Maj', 'G# Maj', 'A Maj', 'A# Maj', 'B Maj',
                          'c min', 'c# min', 'd min', 'd# min', 'e min', 'f min', 'f# min', 'g min', 'g# min', 'a min', 'a# min', 'b min'])
    
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                     [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    t_pc = t_pc / t_pc.sum(axis=1, keepdims=True) if REPLACE_L2_BY_L1 else t_pc / (t_pc ** 2).sum(axis=1, keepdims=True) ** 0.5
    
    
    xb, t = block_audio(x=x, blockSize=blockSize, hopSize=hopSize, fs=fs)
    X, f = compute_spectrogram(xb=xb, fs=fs)
    
    tfInHz = estimate_tuning_freq(x=x, blockSize=blockSize, hopSize=hopSize, fs=fs) if bTune else 440.0
    
    pitch_chroma = extract_pitch_chroma(X=X, fs=fs, tfInHz=tfInHz)
    pitch_chroma = pitch_chroma.mean(axis=1)
    
    # compute manhattan distances for modes (major and minor)
    d = np.zeros(t_pc.shape)
    v_pc = np.vstack((pitch_chroma, pitch_chroma))
    for i in range(0, 12):
        d[:, i] = np.sum(np.abs(v_pc - np.roll(t_pc, i, axis=1)), axis=1) if REPLACE_L2_BY_L1 else np.sum(np.square(v_pc - np.roll(t_pc, i, axis=1)), axis=1) ** 0.5

    # get unwrapped key index
    iKeyIdx = d.argmin()

    cKey = cKeyNames[iKeyIdx]

    return cKey

def eval_tfe(pathToAudio, pathToGT):
    
    filenames = [filename.replace(".wav", "") for filename in os.listdir(pathToAudio)]
    
    blockSize = 4096
    hopSize = 2048
    results = []
    for filename in filenames:
        x, fs = sf.read(os.path.join(pathToAudio, "{}.wav".format(filename)))
        tfInHz = estimate_tuning_freq(x=x, blockSize=blockSize, hopSize=hopSize, fs=fs)
        
        with open(os.path.join(pathToGT, "{}.txt".format(filename))) as f:
            gt = float(f.readline().strip())         
        results.append(1200 * np.log2(tfInHz / gt))
#         print(filename, tfInHz, gt)
        
    return sum(results) / len(results)


def eval_key_detection(pathToAudio, pathToGT):
    filenames = [filename.replace(".wav", "") for filename in os.listdir(pathToAudio)]
    
    # key names
    cKeyNames = np.array(['A Maj', 'A# Maj', 'B Maj', 'C Maj', 'C# Maj', 'D Maj', 'D# Maj', 'E Maj', 'F Maj', 'F# Maj', 'G Maj', 'G# Maj',
                          'a min', 'a# min', 'b min', 'c min', 'c# min', 'd min', 'd# min', 'e min', 'f min', 'f# min', 'g min', 'g# min'])
    blockSize = 4096
    hopSize = 2048
    results_with_est_tf = []
    results_without_est_tf = []
    
    for filename in filenames:
        x, fs = sf.read(os.path.join(pathToAudio, "{}.wav".format(filename)))
        tfInHz_with_est_tf = detect_key(x=x, blockSize=blockSize, hopSize=hopSize, fs=fs, bTune=True)
        tfInHz_without_est_tf = detect_key(x=x, blockSize=blockSize, hopSize=hopSize, fs=fs, bTune=False)
        
        with open(os.path.join(pathToGT, "{}.txt".format(filename))) as f:
            gt = int(f.readline().strip())
            gt = cKeyNames[gt]
#         print(filename, tfInHz_with_est_tf, tfInHz_without_est_tf, gt)

        results_with_est_tf.append(tfInHz_with_est_tf == gt)
        results_without_est_tf.append(tfInHz_without_est_tf == gt)

    return np.array([
        sum(results_with_est_tf) / len(results_with_est_tf),
        sum(results_without_est_tf) / len(results_without_est_tf)
    ])

def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):
    err_tf = eval_tfe(pathToAudioTf, pathToGTTf)
    err_key = eval_key_detection(pathToAudioKey, pathToGTKey)
    return err_tf, err_key

if __name__ == "__main__":
    # Error of tuning frequency estimation: 6.0 in cents
    # Accuracy of key detection:
    # 0.6, 0.9 with L2 distance
    # 0.8, 1.0 with L1 distance
    evaluate("./key_tf/key_eval/audio/", "./key_tf/key_eval/GT/", "./key_tf/tuning_eval/audio/", "./key_tf/tuning_eval/GT/")