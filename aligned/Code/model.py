import torch
import numpy as np
import torch.nn as nn

class SpeechClassifier(nn.Module):
    def __init__(self, size_list, residual_blocks_idx):     #FIXME: Add params here regarding model size.
        super(SpeechClassifier, self).__init__()
        layers = []
        self.size_list = size_list
        self.layers = []
        self.residual_blocks_idx = residual_blocks_idx
        for i in range(len(size_list) - 2):
            self.layers.append(nn.Linear(size_list[i],size_list[i+1]))
            self.layers.append(nn.BatchNorm1d(size_list[i+1]))
            self.layers.append(nn.LeakyReLU())
            #self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.n_layers = len(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = x.view(-1, self.size_list[0]) # Flatten the input
        out = x
        residual_x = x
        residual_idx = 0
        for i in range(self.n_layers):
            # 3 layers per linear.
            ## linear + b-norm + relu
            out = self.layers[i](out)
            ## apple skip connections at residual_blocks_idx layers.
            if residual_idx < len(self.residual_blocks_idx):
                if ((self.residual_blocks_idx[residual_idx]*3)+1) == i:
                    residual_x = out
                if (((self.residual_blocks_idx[residual_idx]+2)*3)+1) == i:
                    out = out + residual_x
                    residual_idx += 1
        return out
