import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
from phoneme_list import *

#RNN only.
class SpeechRecognizer(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, nlayers, bidirectional, dropout):
        super(SpeechRecognizer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.rnn = nn.LSTM(input_size = input_size, hidden_size=hidden_size,
                            num_layers = nlayers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        self.bidir_factor = 2 if self.bidirectional else 1
        self.scoring = nn.Linear(hidden_size*self.bidir_factor, vocab_size)
        self.hidden = None

    def forward(self, in_pad, lens):
        in_pkd = rnn.pack_padded_sequence(in_pad, lens)
        out_pkd, (h, c) = self.rnn(in_pkd, self.hidden)
        scores = self.scoring(out_pkd)      # Predict the scores.
        return scores

# Bidirectional RNN block.
class BiRnnBlock(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='lstm', nlayers=1):
        super(BiRnnBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.rnn_type = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = self.rnn_type(input_size = input_size, hidden_size=self.hidden_size,
                                    num_layers = nlayers, bidirectional=True)

    def forward(self, seq, seq_len):
        out = rnn.pack_padded_sequence(seq, seq_len)
        out, h = self.rnn(out)
        out, _ = rnn.pad_packed_sequence(out)
        return out, h

# CNN to extract features.
class CnnBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CnnBlock, self).__init__()
        # Input Dim: [batch_size, feat_size, max_seq_len]
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size,
                        kernel_size=3, padding=1),  #stride=1
            nn.BatchNorm1d(hidden_size),
            nn.Hardtanh(inplace=True)   # Helps with exploding gradients.
        )

    def forward(self, x):
        return self.layers(x)

# Locked dropout.
# Borrowed from Gal and Ghahramani / Merity
# https://medium.com/@bingobee01/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, train, dropout=0.5):
        # x:  (L, B, C)
        if dropout == 0 or not train:
            return x
        mask = x.data.new(1, x.size(1), x.size(2))
        mask = mask.bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

# Uses CNN-RNN.
class SpeechRecognizer2(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, vocab_size, nlayers, dropout_all):
        super(SpeechRecognizer2, self).__init__()
        self.nlayers = nlayers
        # Extract features using CNN.
        self.feat_extractor = CnnBlock(input_size, hidden_size)
        # Pass sequence features through RNN layers.
        # Increase RNN layer hidden size by 2x.
        self.rnn = [BiRnnBlock(hidden_size, hidden_size, rnn_type)]
        for i in range(nlayers - 1):
            self.rnn.append(BiRnnBlock(hidden_size * 2, hidden_size, rnn_type))
        self.rnn = torch.nn.ModuleList(self.rnn)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        # Variational Dropout for RNN.
        self.rnn_dropout = LockedDropout()
        # RNN input layer, RNN hidden layer, Linear Layer.
        self.dropout_i, self.dropout_h, self.dropout = dropout_all
        # Initialize weights.
        self.init_weights()

    def forward(self, seq, seq_len, train=True):
        # Input Dim: Padded - [max_seq_len, batch, feat_size]
        seq = seq.permute(1,2,0)
        seq = seq.contiguous()
        # Dim: [batch, feat_size, max_seq_len]
        out = self.feat_extractor(seq)
        out = out .permute(2,0,1)
        # Dim: [max_seq_len, batch, feat_size]
        out = self.rnn_dropout(out, train, self.dropout_i)     # Dropout for input layer.
        for l, rnn in enumerate(self.rnn):
            # out, h = rnn(out, self.hs[l])
            out, h = rnn(out, seq_len) #BnLSTM
            # out, h = rnn(out)
            if l != (self.nlayers - 1):
                out = self.rnn_dropout(out, train, self.dropout_h)
        out = self.rnn_dropout(out, train, self.dropout)
        out = self.classifier(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        # Init classifier bias with prior probability.
        phoneme_dist = torch.from_numpy(np.array(PHONEME_PRIOR))
        self.classifier.bias.data = phoneme_dist.float()
