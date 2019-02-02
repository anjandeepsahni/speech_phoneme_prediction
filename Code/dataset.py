import os
import torch
import numpy as np
from torch.utils.data import Dataset as Dataset
import torch.utils.data.dataloader as dataloader

SPEECH_DATA_PATH = './../Data'

class SpeechDataset(Dataset):
    def __init__(self, mode='train'):
        # Check for valid mode.
        self.mode = mode
        valid_modes = {'train', 'dev', 'test'}
        if self.mode not in valid_modes:
            raise ValueError("SpeechDataset Error: Mode must be one of %r." % valid_modes)
        # Path where data and labels tensor will be dumped/loaded to/from.
        self.dataTensorPath = os.path.join(SPEECH_DATA_PATH, '{}.pt'.format(self.mode))
        self.labelsTensorPath = os.path.join(SPEECH_DATA_PATH, '{}_labels.pt'.format(self.mode))
        self.utteranceIndicesPath = os.path.join(SPEECH_DATA_PATH, '{}_utteranceIndices.pt'.format(self.mode))
        # Check if we already have previously dumped tensor data.
        self.data = None
        self.labels = None
        if os.path.isfile(self.dataTensorPath):
            self.data = torch.load(self.dataTensorPath)
            # If data tensor exists then labels MUST exist for train and dev modes.
            if self.mode != 'test':
                if os.path.isfile(self.labelsTensorPath):
                    self.labels = torch.load(self.labelsTensorPath)
                else:
                    raise ValueError("SpeechDataset Error: Data tensor file found at \
                        %s but labels tensor file missing at %s." % (self.dataTensorPath, self.labelsTensorPath))
            # Load remaining parameters.
            if os.path.isfile(self.utteranceIndicesPath):
                self.utteranceIndices = torch.load(self.utteranceIndicesPath)
            else:
                raise ValueError("SpeechDataset Error: Data tensor file found at \
                    %s but utteranceIndices file missing at %s." % (self.dataTensorPath, self.utteranceIndicesPath))
            self.totalFrameCount = self.data.size()[0]
        else:
            # Load the data and labels (labels = None for 'test' mode)
            train_data, train_labels = self.loadRawData()
            # Convert each utterance into tensor.
            # Stack all frames together to form single indexable tensor.
            # Maintain a list of indices to remember where each utterance starts.
            self.utteranceIndices = []
            self.totalFrameCount = 0
            self.data = torch.from_numpy(np.concatenate(train_data, axis = 0))
            if self.mode != 'test':
                self.labels = torch.from_numpy(np.concatenate(train_labels, axis = 0))
            for i in range(len(train_data)):
                self.utteranceIndices.append(self.totalFrameCount)
                self.totalFrameCount += train_data[i].shape[0]
            # Check data size is equal to total frame count.
            if self.data.size()[0] != self.totalFrameCount:
                raise ValueError("SpeechDataset Error: Data tensor size %d \
                                is not equal to total frame count %d." % (self.data.size()[0], self.totalFrameCount))
            # Check labels size is equal to total frame count.
            if self.mode != 'test':
                if self.labels.size()[0] != self.totalFrameCount:
                    raise ValueError("SpeechDataset Error: Labels tensor size %d \
                                    is not equal to total frame count %d." % (self.labels.size()[0], self.totalFrameCount))
            # Dump data and labels tensor so we don't have to create again.
            torch.save(self.data, self.dataTensorPath)
            if self.mode != 'test':
                torch.save(self.labels, self.labelsTensorPath)
            torch.save(self.utteranceIndices, self.utteranceIndicesPath)

    def __len__(self):
        return self.totalFrameCount

    def __getitem__(self, idx):
        if self.mode != 'test':
            return (self.data[idx], self.labels[idx])
        else:
            return self.data[idx]

    def loadRawData(self):
        if self.mode == 'train' or self.mode == 'dev':
            return (
                np.load(os.path.join(SPEECH_DATA_PATH, '{}.npy'.format(self.mode)), encoding='bytes'),
                np.load(os.path.join(SPEECH_DATA_PATH, '{}_labels.npy'.format(self.mode)), encoding='bytes')
                )
        else:   # No labels in test mode.
            return (
                np.load(os.path.join(SPEECH_DATA_PATH, '{}.npy'.format(self.mode)), encoding='bytes'),
                None
                )
