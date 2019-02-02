import torch
import torch.nn as nn
import numpy as np
import os
from dataset import SpeechDataset
from model import SpeechClassifier
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import tqdm as tqdm

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    model.to(device)
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.long().to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('Train Iteration: %d/%d Loss = %5.4f' % \
                (batch_idx+1, len(train_loader), (running_loss/(batch_idx+1))), \
                end="\r", flush=True)
    end_time = time.time()
    running_loss /= len(train_loader)
    print('\nTraining Loss: %5.4f Time: %d s' % (running_loss, end_time - start_time))
    return running_loss

def val_model(model, val_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.long().to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
            print('Val Iteration: %d/%d Loss = %5.4f' % \
                    (batch_idx+1, len(val_loader), (running_loss/(batch_idx+1))), \
                    end="\r", flush=True)
        end_time = time.time()
        running_loss /= len(val_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('\nValidation Loss: %5.4f Validation Accuracy: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
        return running_loss, acc

if __name__ == "__main__":
    # Instantiate speech dataset.
    speechTrainDataset = SpeechDataset(mode='train')
    speechTestDataset = SpeechDataset(mode='test')
    speechValDataset = SpeechDataset(mode='dev')
    train_loader = DataLoader(speechTrainDataset, batch_size=400,
                                shuffle=False, num_workers=4)
    test_loader = DataLoader(speechTestDataset, batch_size=4,
                                shuffle=False, num_workers=4)
    val_loader = DataLoader(speechValDataset, batch_size=400,
                            shuffle=False, num_workers=4)

    model = SpeechClassifier()
    criterion = nn.CrossEntropyLoss()
    print('='*20)
    print(model)

    optimizer = optim.Adam(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 1
    Train_loss = []
    Val_loss = []
    Val_acc = []
    #Test_loss = []
    #Test_acc = []

    print('='*20)
    for i in range(n_epochs):
        print('Epoch: %d/%d' % (i+1,n_epochs))
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_model(model, val_loader, criterion, device)
        Train_loss.append(train_loss)
        Val_loss.append(val_loss)
        Val_acc.append(val_acc)
        print('='*20)
