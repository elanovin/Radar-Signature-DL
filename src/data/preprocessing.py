import numpy as np
from scipy import signal
from typing import Optional

def process_radar_signal(
    radar_signal: np.ndarray,
    fs: float = 1000.0,
    window_size: int = 256,
    overlap: int = 128
) -> np.ndarray:
    """
    Process raw radar signal to generate micro-Doppler spectrogram.
    
    Args:
        radar_signal: Raw radar signal data
        fs: Sampling frequency in Hz
        window_size: STFT window size
        overlap: Overlap between windows
        
    Returns:
        Processed spectrogram as 2D numpy array
    """
    # Apply windowing
    window = signal.windows.hann(window_size)
    
    # Compute STFT
    f, t, Sxx = signal.spectrogram(
        radar_signal,
        fs=fs,
        window=window,
        nperseg=window_size,
        noverlap=overlap,
        mode='magnitude'
    )
    
    # Convert to dB scale
    Sxx_db = 20 * np.log10(Sxx + 1e-10)
    
    # Normalize
    Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db))
    
    return Sxx_norm

def apply_clutter_removal(
    spectrogram: np.ndarray,
    alpha: float = 0.9
) -> np.ndarray:
    """
    Apply moving target indicator (MTI) filter for clutter removal.
    
    Args:
        spectrogram: Input spectrogram
        alpha: MTI filter coefficient
        
    Returns:
        Clutter-removed spectrogram
    """
    # Apply MTI filter along time axis
    filtered = spectrogram - alpha * np.roll(spectrogram, 1, axis=1)
    filtered[:, 0] = filtered[:, 1]  # Fix edge effect
    
    return filtered 