import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class SpeechRecognizer(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, nlayers):
        super(SpeechRecognizer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.rnn = nn.LSTM(input_size = input_size, hidden_size=hidden_size, num_layers = nlayers, batch_first=True)
        self.scoring = nn.Linear(hidden_size, vocab_size)

    def forward(self, seq_list):
        batch_size = len(seq_list)
        #for i, seq in enumerate(seq_list):
        #    print('seq %d shape: %s' %(i, seq.shape))
        # Length of each sequence.
        lens = [len(seq) for seq in seq_list]
        # Create packed sequence for rnn.
        in_pkd = rnn.pack_sequence(seq_list)
        #print('in_pkd.data.shape:', in_pkd.data.shape)
        hidden=None     # Clear hidden for each batch.
        out_pkd, hidden = self.rnn(in_pkd, hidden)
        #print('out_pkd.data.shape:', out_pkd.data.shape)
        # Unpack the output.
        out_pad, _ = rnn.pad_packed_sequence(out_pkd, batch_first=True)
        #print('out_pad.shape:', out_pad.shape)
        # Remove the padding and flatten the output for final layer.
        #out_flatten = torch.cat([out_pad[:lens[i],i] for i in range(batch_size)])
        # Predict the scores.
        scores = self.scoring(out_pad)
        #scores_flatten = self.scoring(out_flatten)
        # Recreate the list of scores for each utterance.
        #scores_list = []

        #print('scores.shape:', scores.shape)
        #return scores.view(-1, batch_size, self.vocab_size)
        return scores
