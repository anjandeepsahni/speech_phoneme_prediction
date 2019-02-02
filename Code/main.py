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
import csv

MODEL_PATH = './../Models'
TEST_RESULT_PATH = './../Results'

def save_test_results(predictions):
    predictions = list(predictions.numpy())
    predictions_count = list(range(len(predictions)))
    csv_output = [[i,j] for i,j in zip(predictions_count,predictions)]
    result_file_path = os.path.join(TEST_RESULT_PATH, 'result_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S")))
    with open(result_file_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'label'])
        csv_writer.writerows(csv_output)

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
            print('Validation Iteration: %d/%d Loss = %5.4f' % \
                    (batch_idx+1, len(val_loader), (running_loss/(batch_idx+1))), \
                    end="\r", flush=True)
        end_time = time.time()
        running_loss /= len(val_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('\nValidation Loss: %5.4f Validation Accuracy: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
        return running_loss, acc

def test_model(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        start_time = time.time()
        all_predictions = []
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        # Join list of predicted tensors.
        all_predictions = torch.cat(all_predictions, 0)
        # Save predictions in csv file.
        save_test_results(all_predictions)
        end_time = time.time()
        print('\nTotal Test Predictions: %d Time: %d s' % (all_predictions.size()[0], end_time - start_time))

if __name__ == "__main__":
    reload_model = True
    testing_only = True

    # Instantiate speech dataset.
    speechTrainDataset = SpeechDataset(mode='train')
    speechTestDataset = SpeechDataset(mode='test')
    speechValDataset = SpeechDataset(mode='dev')
    train_loader = DataLoader(speechTrainDataset, batch_size=500,
                                shuffle=False, num_workers=4)
    test_loader = DataLoader(speechTestDataset, batch_size=500,
                                shuffle=False, num_workers=4)
    val_loader = DataLoader(speechValDataset, batch_size=500,
                            shuffle=False, num_workers=4)

    model = SpeechClassifier()
    criterion = nn.CrossEntropyLoss()
    print('='*20)
    print(model)

    optimizer = optim.Adam(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if reload_model:
        # Find the list of all saved models.
        model_list = [f for f in os.listdir(MODEL_PATH) if os.path.isfile(os.path.join(MODEL_PATH, f))]
        if model_list:
            # Load the last saved model.
            model_path = os.path.join(MODEL_PATH, model_list[-1])
            model.load_state_dict(torch.load(model_path))

    n_epochs = 1
    Train_loss = []
    Val_loss = []
    Val_acc = []

    print('='*20)

    if not testing_only:
        for i in range(n_epochs):
            print('Epoch: %d/%d' % (i+1,n_epochs))
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_model(model, val_loader, criterion, device)
            Train_loss.append(train_loss)
            Val_loss.append(val_loss)
            Val_acc.append(val_acc)
            print('='*20)

        # Save model parameters.
        finalValAcc = '%.3f'%(Val_acc[-1])
        model_path = os.path.join(MODEL_PATH, 'model_{}_val_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), finalValAcc))
        torch.save(model.state_dict(), model_path)
    else:
        # Only testing the model.
        test_model(model, test_loader, device)
        print('='*20)
