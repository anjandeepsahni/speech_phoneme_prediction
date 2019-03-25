import os
import torch
import numpy as np
from torch.utils.data import Dataset as Dataset

SPEECH_DATA_PATH = './../Data'

class SpeechDataset(Dataset):
    def __init__(self, mode='train', device="cpu"):
        # Check for valid mode.
        self.mode = mode
        self.device = device
        valid_modes = {'train', 'dev', 'test'}
        if self.mode not in valid_modes:
            raise ValueError("SpeechDataset Error: Mode must be one of %r." % valid_modes)
        # Load the data and labels (labels = None for 'test' mode)
        self.data, self.labels = self.loadRawData()
        self.data = [torch.from_numpy(data) for data in self.data]
        if self.mode != 'test':
            self.labels = [torch.from_numpy(label) for label in self.labels]
        self.feature_size = self.data[0].size(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode=='test':
            #return self.data[idx].to(self.device)
            return self.data[idx]
        else:
            #return self.data[idx].to(self.device), self.labels[idx].to(self.device)
            return self.data[idx], self.labels[idx]

    def loadRawData(self):
        if self.mode == 'train' or self.mode == 'dev':
            return (
                np.load(os.path.join(SPEECH_DATA_PATH, 'wsj0_{}.npy'.format(self.mode)), encoding='bytes'),
                np.load(os.path.join(SPEECH_DATA_PATH, 'wsj0_{}_merged_labels.npy'.format(self.mode)), encoding='bytes')
                )
        else:   # No labels in test mode.
            return (
                np.load(os.path.join(SPEECH_DATA_PATH, 'transformed_test_data.npy'), encoding='bytes'),
                None
                )

# Modify the batch in collate_fn to sort the
# batch in decreasing order of size.
def SpeechCollateFn(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs, targets
