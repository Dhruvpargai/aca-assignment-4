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