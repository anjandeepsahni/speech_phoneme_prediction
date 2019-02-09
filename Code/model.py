import torch
import numpy as np
import torch.nn as nn

class SpeechClassifier(nn.Module):
    def __init__(self, size_list):     #FIXME: Add params here regarding model size.
        super(SpeechClassifier, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.size_list[0]) # Flatten the input
        return self.net(x)
