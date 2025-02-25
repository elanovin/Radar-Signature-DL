import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple
from .preprocessing import process_radar_signal

class RadarDataset(Dataset):
    def __init__(self, data_path: str, transform: bool = True):
        """
        Args:
            data_path: Path to the HDF5 data file
            transform: Whether to apply data augmentation
        """
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Load data
        with h5py.File(self.data_path, 'r') as f:
            self.signals = f['signals'][:]
            self.labels = f['labels'][:]
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get raw signal and label
        signal = self.signals[idx]
        label = self.labels[idx]
        
        # Process signal to get spectrogram
        spectrogram = process_radar_signal(signal)
        
        # Apply data augmentation if enabled
        if self.transform:
            spectrogram = self._augment(spectrogram)
        
        # Convert to tensor
        spectrogram_tensor = torch.from_numpy(spectrogram).float()
        
        # Add channel dimension
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
        
        return spectrogram_tensor, label
    
    def _augment(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply random augmentations to the spectrogram."""
        # Add random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.05, spectrogram.shape)
            spectrogram = spectrogram + noise
        
        # Random time shift
        if np.random.random() > 0.5:
            shift = np.random.randint(-10, 10)
            spectrogram = np.roll(spectrogram, shift, axis=1)
        
        return spectrogram 