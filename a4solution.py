import numpy as np
from scipy.signal import find_peaks

fs = 44100

def block_audio(x: np.ndarray, block_size: int, hop_size: int, fs: int) -> (np.ndarray, np.ndarray):
    num_blocks = len(x) // hop_size
    x_padded = np.pad(x, (0, block_size - len(x) % hop_size))
    
    xb = np.zeros((num_blocks, block_size), dtype=np.float64)
    time_in_sec = np.zeros(num_blocks, dtype=np.float64)      
    
    for i in range(num_blocks):
        xb[i] = x_padded[i * hop_size: i * hop_size + block_size]
        time_in_sec[i] = (i * hop_size) / fs
    
    return xb, time_in_sec

def get_stft(xb, block_size, hop_size, fs):
    Xb = np.fft.rfft(xb, n=block_size, axis=-1)
    fInHz = np.fft.fftfreq(block_size//2 + 1)
    return Xb, fInHz

def compute_spectrogram(xb, block_size, hop_size, fs):
    X, fInHz = get_stft(xb, block_size, hop_size, fs)
    X = np.abs(X)
    return X, fInHz

def get_spectral_peaks(X):
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
    X, fInHz = compute_spectrogram(xb, blockSize, hopSize, fs)
    top_twenty_freq_bins = get_spectral_peaks(X)
    error_in_cents = np.zeros(top_twenty_freq_bins.flatten().shape)
    
    for i, freq in enumerate(top_twenty_freq_bins.flatten()):
        if freq > 0:
            nearest_pitch = np.argmin(np.abs(freq - equal_temperament_pitches))
            error_in_cents[i] = 1200 * np.log2(freq / equal_temperament_pitches[nearest_pitch])

    hist, bin_edges = np.histogram(error_in_cents, 100, (-50, 50))
    mode_of_delta_c = bin_edges[np.argmax(hist)]
    return 2 ** (mode_of_delta_c / 1200) * 440


# def generate_sine(frequency, amplitude, duration, sample_rate):
#     t = np.linspace(0,duration,int(sample_rate*duration))
#     signal = amplitude*np.sin(2*np.pi*frequency*t)
#     return signal

# test_signal = np.concatenate((generate_sine(441, 1, 1, fs), generate_sine(882, 1, 1, fs)))
# print (estimate_tuning_freq(test_signal, 1024, 512, fs))