import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class SpeechRecognizer(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, nlayers, batch_size, bidirectional, dropout):
        super(SpeechRecognizer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.dropout = dropout
        self.rnn = nn.LSTM(input_size = input_size, hidden_size=hidden_size,
                            num_layers = nlayers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        self.bidir_factor = 2 if self.bidirectional else 1
        self.scoring = nn.Linear(hidden_size*self.bidir_factor, vocab_size)
        self.hidden = None

    def forward(self, in_pkd, lens):
        batch_size = len(lens)
        out_pkd, (h, c) = self.rnn(in_pkd, self.hidden)
        out_pad, _ = rnn.pad_packed_sequence(out_pkd, batch_first=True)     # Unpack the output.
        scores = self.scoring(out_pad)      # Predict the scores.
        return scores


'''
#        if self.bidirectional:
#            self.hidden = nn.Parameter(torch.zeros(self.nlayers*2, self.batch_size, self.hidden_size))
#            self.cec = nn.Parameter(torch.zeros(self.nlayers*2, self.batch_size, self.hidden_size))
#        else:
#            self.hidden = nn.Parameter(torch.zeros(self.nlayers, self.batch_size, self.hidden_size))
#            self.cec = nn.Parameter(torch.zeros(self.nlayers, self.batch_size, self.hidden_size))


    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return nn.Parameter(h.detach())
        else:
            return tuple(self.repackage_hidden(v) for v in h)
'''
