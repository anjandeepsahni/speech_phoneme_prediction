import os
import torch
import numpy as np
import torch.nn.utils.rnn as rnn
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
        #return 10
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode=='test':
            return self.data[idx]
        else:
            return self.data[idx], (self.labels[idx] + 1)   # Plus 1 because blank_id = 0.

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
    if isinstance(seq_list[0], tuple):
        inputs, targets = zip(*seq_list)
    else:
        inputs = seq_list
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    #print(seq_order)
    #reorder_seq = np.argsort(seq_order)
    #print(reorder_seq)
    #inputs_bak = inputs
    if isinstance(seq_list[0], tuple):
        inputs = [inputs[i] for i in seq_order]
    else:
        inputs = [inputs[i].type(torch.float32) for i in seq_order]     # RNN does not accept Float64.
    #print('inputs_bak=', inputs_bak)
    #print('inputs=', inputs)
    #reordered_inputs = [inputs[i] for i in reorder_seq]
    #print('reordered_inputs=', reordered_inputs)
    inp_lens = [len(seq) for seq in inputs]
    inp_pkd = rnn.pack_sequence(inputs)    # Create packed sequence for rnn.
    if isinstance(seq_list[0], tuple):
        targets = [targets[i] for i in seq_order]
        return inp_pkd, inp_lens, targets
    else:
        return inp_pkd, inp_lens, seq_order     # Return seq_order for test data to rearrange later.
