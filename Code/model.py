import torch
import torch.nn as nn
import numpy as np

class SpeechClassifier(nn.Module):
    def __init__(self):     #FIXME: Add params here regarding model size.
        super(SpeechClassifier, self).__init__()
        self.linear1 = nn.Linear(40, 138)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return x
