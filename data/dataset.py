from torch.utils.data import Dataset
import torch
import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for LSTM model parameters"""
    hidden_size: int
    num_layers: int
    sequence_length: int
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    num_features: int = 4
    num_epochs: int = 100

class SepsisDataset(Dataset):
    def __init__(self, sequences, labels, feature_names):
        
        self.feature_names = feature_names
        self.sequences = []
        self.comorbidities = []
        self.case_ids = []
        self.labels = torch.tensor(labels, dtype=torch.float)
        
        for seq in sequences:
            case_id = seq[2]
            com = seq[1]
            seq = seq[0]
            for i in range(len(seq)):
                seq[i] = torch.tensor(seq[i], dtype=torch.float32)
            com = torch.tensor(com, dtype=torch.float32)
            self.sequences.append(seq)
            self.comorbidities.append(com)
            self.case_ids.append(case_id)

        self.sequences = [
            torch.cat(seq, dim=0) for seq in self.sequences
        ]

        self.sequences = [
            torch.cat((seq, com), dim=0) for seq, com in zip(self.sequences, self.comorbidities)
        ]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.labels[idx],
            self.case_ids[idx]
        )