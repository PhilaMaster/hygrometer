import os
import torch
import torchaudio
from torch.utils.data import Dataset


class MSCDataset(Dataset):
    """
    Custom Dataset class for Mini Speech Commands.
    
    Args:
        root (str): Root folder path where the dataset is stored.
        classes (list of str): Ordered list of classes specifying the target keywords.
                               This list will be used to map textual labels to integer indices.
    """
    
    def __init__(self, root, classes):
        self.root = root
        self.classes = classes
        self.label_to_idx = {label: idx for idx, label in enumerate(classes)}
        
        # collect all audio file paths and their corresponding labels
        self.samples = []
        
        # flat structure (root/<label>_*.wav)
        for filename in os.listdir(root):
            if filename.endswith('.wav'):
                label = filename.split('_')[0]
                if label in classes:
                    file_path = os.path.join(root, filename)
                    self.samples.append((file_path, label))
        
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Returns a Dictionary with the following structure:
            {
                "x": torch.Tensor - Audio data as a tensor,
                "sampling_rate": int - Sampling rate of the audio,
                "label": int - Integer label corresponding to the keyword
            }
        """
        file_path, label_str = self.samples[idx]
                
        # Load and Convert label string to integer using the provided label mapping
        waveform, sampling_rate = torchaudio.load(file_path)
        label_int = self.label_to_idx[label_str]
        
        return {
            "x": waveform,
            "sampling_rate": sampling_rate,
            "label": label_int
        }
    
    def label_to_int(self, label_str):
        """
        Convert a label string to its corresponding integer index.
        
        Returns the integer index corresponding to the label
        -1 otherwise
        """
        return self.label_to_idx.get(label_str, -1)