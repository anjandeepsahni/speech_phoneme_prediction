import os
import csv
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import SpeechDataset
from model import SpeechClassifier
from torch.utils.data import DataLoader

# Paths
MODEL_PATH = './../Models'
TEST_RESULT_PATH = './../Results'

# Defaults
DEFAULT_RUN_MODE = 'train'
DEFAULT_TRAIN_BATCH_SIZE = 600
DEFAULT_TEST_BATCH_SIZE = 200

# Hyperparameters.
#LEARNING_RATE = 0.001
LEARNING_RATE = 0.002           #Trying batchnorm.
LEARNING_RATE_STEP = 0.7
FRAME_CONTEXT_RANGE = 12        # On each side.

# Saves test results to csv file for kaggle submission.
def save_test_results(predictions):
    predictions = list(predictions.cpu().numpy())
    predictions_count = list(range(len(predictions)))
    csv_output = [[i,j] for i,j in zip(predictions_count,predictions)]
    result_file_path = os.path.join(TEST_RESULT_PATH,\
            'result_{}.csv'.format((str.split(str.split(args.model_path, '/')[-1], '.pt')[0])))
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

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Speech Classifier.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default=DEFAULT_RUN_MODE, help='\'train\' or \'test\' mode.')
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE, help='Testing batch size.')
    parser.add_argument('--reload_model', type=bool, default=False, help='True/false, if we have to reload a model.')
    parser.add_argument('--model_path', type=str, help='Path to model to be reloaded.')
    return parser.parse_args()

if __name__ == "__main__":
    # Create arg parser.
    args = parse_args()
    print('='*20)
    print('Input arguments:\n%s' % (args))

    # Validate arguments.
    if args.mode == 'test' and (not args.reload_model or args.model_path == None):
        raise ValueError("Input Argument Error: Test mode specified but reload_model is %s and model_path is %s." \
                        % (args.reload_model, args.model_path))

    if (args.reload_model and (args.model_path == None)):
        raise ValueError("Input Argument Error: Reload model specified true but model_path is %s." \
                        % (args.model_path))

    # Instantiate speech dataset.
    speechTrainDataset = SpeechDataset(FRAME_CONTEXT_RANGE, mode='train')
    speechTestDataset = SpeechDataset(FRAME_CONTEXT_RANGE, mode='test', )
    speechValDataset = SpeechDataset(FRAME_CONTEXT_RANGE, mode='dev')
    train_loader = DataLoader(speechTrainDataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=8)
    test_loader = DataLoader(speechTestDataset, batch_size=args.test_batch_size,
                                shuffle=False, num_workers=8)
    val_loader = DataLoader(speechValDataset, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=8)

    # Previously tested models.
    #model_size_list = [440, 512, 1024, 1024, 512, 256, 138]                        #Frame context = 5
    #model_size_list = [680, 800, 1024, 1200, 1200, 800, 512, 138]                  #Frame context = 8
    #model_size_list = [680, 750, 800, 800, 800, 800, 650, 400, 300, 200, 138]      #Frame context = 8
    #model_size_list = [680, 750, 800, 800, 800, 800, 800, 500, 250, 138]           #Frame context = 8
    # Working model.
    model_size_list = [1000, 1200, 1200, 1200, 1200, 1200, 800, 500, 138]           #Frame context = 12
    # Residual block input at 750 -> 800, 800 -> 800, 800 -> 800
    residual_blocks_idx = [0, 2]     # Inputs to residual blocks
    model = SpeechClassifier(model_size_list, residual_blocks_idx)
    criterion = nn.CrossEntropyLoss()
    print('='*20)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = LEARNING_RATE_STEP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device = %s." % (device))

    if args.reload_model:
        model.load_state_dict(torch.load(args.model_path))
        print('Loaded model:', args.model_path)

    n_epochs = 50
    Train_loss = []
    Val_loss = []
    Val_acc = []

    print('='*20)

    if args.mode == 'train':
        for i in range(n_epochs):
            print('Epoch: %d/%d' % (i+1,n_epochs))
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_model(model, val_loader, criterion, device)
            Train_loss.append(train_loss)
            Val_loss.append(val_loss)
            Val_acc.append(val_acc)
            # Checkpoint the model after each epoch.
            finalValAcc = '%.3f'%(Val_acc[-1])
            model_path = os.path.join(MODEL_PATH, 'model_{}_val_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), finalValAcc))
            torch.save(model.state_dict(), model_path)
            print('='*20)
            scheduler.step()
    else:
        # Only testing the model.
        test_model(model, test_loader, device)
        print('='*20)
