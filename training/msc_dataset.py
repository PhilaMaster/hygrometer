import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset
from typing import Optional, Callable, List

class MSCDataset(Dataset):
    
    CLASSES = ['stop', 'up']
    
    def __init__(
        self,
        root_dir: str,
        classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_length: int = 16000,
        augment: bool = False  
    ):
        self.root_dir = root_dir
        self.classes = classes if classes is not None else self.CLASSES
        self.transform = transform
        self.target_length = target_length
        self.augment = augment 
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 files in {root_dir}")
            
    def _make_dataset(self) -> List[tuple]:
        samples = []
        if not os.path.isdir(self.root_dir):
            raise RuntimeError(f"Directory not found: {self.root_dir}")
        for filename in os.listdir(self.root_dir):
            if not filename.endswith('.wav'): continue
            for class_name in self.classes:
                if filename.startswith(f"{class_name}_"):
                    file_path = os.path.join(self.root_dir, filename)
                    class_idx = self.class_to_idx[class_name]
                    samples.append((file_path, class_idx))
                    break
        return samples

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        current_length = waveform.shape[1]
        if current_length > self.target_length:
            start = (current_length - self.target_length) // 2
            waveform = waveform[:, start:start + self.target_length]
        elif current_length < self.target_length:
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    # AUGMENTATION
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        # 1. Time Shift (shift audio drom right to left)
        if random.random() < 0.5:
            shift_amt = int(random.uniform(-0.1, 0.1) * self.target_length) # +/- 10%
            waveform = torch.roll(waveform, shifts=shift_amt, dims=1)
        
        # 2. Add Noise (white noise)
        if random.random() < 0.5:
            noise_level = 0.005
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
            
        return waveform

    def __getitem__(self, idx: int) -> dict:
        file_path, label = self.samples[idx]
        
        try:
            waveform, sample_rate = torchaudio.load(file_path, backend="soundfile")
        except TypeError:
            waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        waveform = self._pad_or_trim(waveform)
        
        if self.augment:
            waveform = self._augment_audio(waveform)

        if self.transform is not None:
            features = self.transform(waveform)
            if features.dim() == 4:
                features = features.squeeze(0)
        else:
            features = waveform
        
        return {'x': features, 'y': label}

    def __len__(self): return len(self.samples)
    def get_class_name(self, idx): return self.classes[idx]
    def get_class_distribution(self): return {c:0 for c in self.classes} # Dummy impl