import os
import csv
import copy
import time
import nltk
import torch
import argparse
import ctcdecode
import numpy as np
import torch.nn as nn
from phoneme_list import *
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SpeechDataset, SpeechCollateFn
from model import SpeechRecognizer, SpeechRecognizer2

# Paths
MODEL_PATH = './../Models'
TEST_RESULT_PATH = './../Results'

# Defaults
DEFAULT_RUN_MODE = 'train'
DEFAULT_FEATURE_SIZE = 40
DEFAULT_TRAIN_BATCH_SIZE = 40
DEFAULT_TEST_BATCH_SIZE = 40
DEFAULT_LABEL_MAP = [' '] + PHONEME_MAP

# Hyperparameters.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.2e-6
GRADIENT_CLIP = 0.25
DROPOUT = 0.4       # For RNN output.
DROPOUT_H = 0.3     # For RNN hidden.
DROPOUT_I = 0.64    # For RNN input.
BEAM_SIZE = 40
TEST_BEAM_SIZE = 100

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Speech Recognition.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default=DEFAULT_RUN_MODE, help='\'train\' or \'test\' mode.')
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE, help='Testing batch size.')
    parser.add_argument('--model_path', type=str, help='Path to model to be reloaded.')
    parser.add_argument('--model_ensemble', type=bool, default=False, help='True/false, if we have to model ensembling.')
    return parser.parse_args()

# Maps a phoneme index to string.
def map_phoneme_string(ph_int):
    return DEFAULT_LABEL_MAP[ph_int]

# Generates phoneme string for entire batch of predictions.
def generate_phoneme_string(batch_pred):
    # Loop over entire batch list of phonemes and convert them to strings.
    batch_strings = []
    for pred in batch_pred:
        batch_strings.append(''.join(list(map(map_phoneme_string, list(pred.numpy())))))
    return batch_strings

# For calculating Edit/Levenshtein distance.
def calculate_edit_distance(pred, targets):
    assert len(pred) == len(targets)
    dist = []
    for idx, p in enumerate(pred):
        dist.append(nltk.edit_distance(p, targets[idx]))
    return dist

# Saves test results to csv file for kaggle submission.
def save_test_results(predictions, ensemble=False):
    predictions_count = list(range(len(predictions)))
    csv_output = [[i,j] for i,j in zip(predictions_count,predictions)]
    if not ensemble:
        result_file_path = os.path.join(TEST_RESULT_PATH,\
                'result_{}.csv'.format((str.split(str.split(args.model_path, '/')[-1], '.pt')[0])))
    else:
        result_file_path = os.path.join(TEST_RESULT_PATH,\
                'result_ensemble_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S")))
    with open(result_file_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Id', 'Predicted'])
        csv_writer.writerows(csv_output)

def test_model(model, test_loader, decoder, device):
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        all_predictions = []
        for batch_idx, (inputs, inp_lens, _, _, seq_order) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs, inp_lens)
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            # (Batch, Max_Seq_L, Dict) for CTC Loss.
            outputs = torch.transpose(outputs, 0, 1)
            pad_out, _, _, out_lens = decoder.decode(F.softmax(outputs, dim=2).data.cpu(), torch.IntTensor(inp_lens))
            # Iterate over each item in batch.
            y_pred = []
            for i in range(len(out_lens)):
                y_pred.append(pad_out[i, 0, :out_lens[i, 0]])   # Pick the first output, most likely.
            # Calculate the strings for predictions.
            pred_str = generate_phoneme_string(y_pred)
            # Input is sorted as per length for rnn. Resort the output.
            reorder_seq = np.argsort(seq_order)
            pred_str = [pred_str[i] for i in reorder_seq]
            all_predictions.extend(pred_str)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        end_time = time.time()
        # Save predictions in csv file.
        save_test_results(all_predictions)
        print('\nTotal Test Predictions: %d Time: %d s' % (len(all_predictions), end_time - start_time))

def val_model(model, val_loader, criterion, decoder, device):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        dist = []
        start_time = time.time()
        for batch_idx, (inputs, inp_lens, targets, tar_lens, _) in enumerate(val_loader):
            targets_cat = (torch.cat(targets)).to(device)
            inputs = inputs.to(device)
            outputs = model(inputs, inp_lens, False)
            # Max_Seq_L, Batch, Dict) for CTC Loss.
            loss = criterion(F.log_softmax(outputs, dim=2), targets_cat, inp_lens, tar_lens)    # CTCLoss only!
            running_loss += loss.item()
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            # (Batch, Max_Seq_L, Dict) for CTC Loss.
            outputs = torch.transpose(outputs, 0, 1)
            pad_out, _, _, out_lens = decoder.decode(F.softmax(outputs, dim=2).data.cpu(), torch.IntTensor(inp_lens))
            # Iterate over each item in batch.
            y_pred = []
            for i in range(len(out_lens)):
                y_pred.append(pad_out[i, 0, :out_lens[i, 0]])   # Pick the first output, most likely.
            # Calculate the strings for predictions.
            pred_str = generate_phoneme_string(y_pred)
            # Calculate the strings for targets.
            tar_str = generate_phoneme_string(targets)
            # Calculate edit distance between predictions and targets.
            dist.extend(calculate_edit_distance(pred_str, tar_str))
            print('Validation Iteration: %d/%d Loss = %5.4f' % \
                    (batch_idx+1, len(val_loader), (running_loss/(batch_idx+1))), \
                    end="\r", flush=True)
        end_time = time.time()
        acc = sum(dist)/len(dist)    # Average over edit distance.
        running_loss /= len(val_loader)
        print('\nValidation Loss: %5.4f Validation Levenshtein Distance: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
        return running_loss, acc

def train_model(model, train_loader, criterion, optimizer, decoder, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    dist = []
    measure_train_accuracy = False
    for batch_idx, (inputs, inp_lens, targets, tar_lens, _) in enumerate(train_loader):
        targets_cat = (torch.cat(targets)).to(device)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, inp_lens)
        # (Max_Seq_L, Batch, Feat_Size) for CTC Loss.
        loss = criterion(F.log_softmax(outputs, dim=2), targets_cat, inp_lens, tar_lens)    # CTCLoss only!
        loss.backward()
        running_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)    # To avoid exploding gradient issue.
        optimizer.step()
        if measure_train_accuracy:
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            # (Batch, Max_Seq_L, Feat_Size) for CTC Decode.
            outputs = torch.transpose(outputs, 0, 1)
            pad_out, _, _, out_lens = decoder.decode(F.softmax(outputs, dim=2).data.cpu(), torch.IntTensor(inp_lens))
            # Iterate over each item in batch.
            y_pred = []
            for i in range(len(out_lens)):
                # Pick the first output, most likely.
                y_pred.append(pad_out[i, 0, :out_lens[i, 0]])
            # Calculate the strings for predictions.
            pred_str = generate_phoneme_string(y_pred)
            # Calculate the strings for targets.
            tar_str = generate_phoneme_string(targets)
            # Calculate edit distance between predictions and targets.
            dist.extend(calculate_edit_distance(pred_str, tar_str))
        print('Train Iteration: %d/%d Loss = %5.4f' % \
                (batch_idx+1, len(train_loader), (running_loss/(batch_idx+1))), \
                end="\r", flush=True)
    end_time = time.time()
    acc = (sum(dist)/len(dist)) if measure_train_accuracy else -1
    running_loss /= len(train_loader)
    print('\nTraining Loss: %5.4f Training Levenshtein Distance: %5.4f Time: %d s'
            % (running_loss, acc, end_time - start_time))
    return running_loss

if __name__ == "__main__":
    # Parse args.
    args = parse_args()
    print('='*20)
    print('Input arguments:\n%s' % (args))

    # Validate arguments.
    if args.mode == 'test' and args.model_path == None and not args.model_ensemble:
        raise ValueError("Input Argument Error: Test mode specified but model_path is %s." % (args.model_path))

    # Check for CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders.
    speechTrainDataset = SpeechDataset(mode='train')
    speechValDataset = SpeechDataset(mode='dev')
    speechTestDataset = SpeechDataset(mode='test')

    train_loader = DataLoader(speechTrainDataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=4, collate_fn=SpeechCollateFn)
    val_loader = DataLoader(speechValDataset, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=4, collate_fn=SpeechCollateFn)
    test_loader = DataLoader(speechTestDataset, batch_size=args.test_batch_size,
                        shuffle=False, num_workers=4, collate_fn=SpeechCollateFn)

    # Create the model.
    input_size = DEFAULT_FEATURE_SIZE
    vocab_size = len(DEFAULT_LABEL_MAP)

    #---------------------------#
    # Previously tested models.
    #---------------------------#
    # Model_1 -> SpeechRecognizer
    # hidden_size = 100
    # nlayers = 3
    # bidirectional = False
    # dropout = 0

    # Model_2 -> SpeechRecognizer
    # hidden_size = 100
    # nlayers = 3
    # bidirectional = True
    # dropout = 0

    # Model_3 -> SpeechRecognizer -> Gives nan loss.
    # hidden_size = 100
    # nlayers = 3
    # bidirectional = True
    # dropout = 0.3

    # Model_4 -> SpeechRecognizer
    #hidden_size = 256
    #nlayers = 3
    #bidirectional = True
    #dropout = 0

    # Model_5 -> SpeechRecognizer
    # hidden_size = 320
    # nlayers = 4
    # bidirectional = True
    # dropout = 0

    #---------------------------#
    # Working model.
    #---------------------------#
    # Model_6 -> SpeechRecognizer2
    rnn_type = 'lstm'
    hidden_size = 384
    nlayers = 4
    dropout_all = (DROPOUT_I, DROPOUT_H, DROPOUT)
    model = SpeechRecognizer2(rnn_type, input_size, hidden_size, vocab_size, nlayers, dropout_all)
    #model = SpeechRecognizer(input_size, hidden_size, vocab_size,
    #                        nlayers, bidirectional, dropout)
    model.to(device)
    print('='*20)
    print(model)
    model_params = sum(p.size()[0] * p.size()[1] if len(p.size()) > 1 else p.size()[0] for p in model.parameters())
    print('Total model parameters:', model_params)
    print("Running on device = %s." % (device))

    # Setup learning parameters.
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.01, verbose=True)
    if args.mode == 'test':
        BEAM_SIZE = TEST_BEAM_SIZE
    decoder = ctcdecode.CTCBeamDecoder(labels=DEFAULT_LABEL_MAP, blank_id=0, beam_width=BEAM_SIZE)

    if args.model_path != None:
        model.load_state_dict(torch.load(args.model_path))
        print('Loaded model:', args.model_path)

    n_epochs = 50
    print('='*20)

    if args.mode == 'train':
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch+1,n_epochs))
            train_loss = train_model(model, train_loader, criterion, optimizer, decoder, device)
            val_loss, val_acc = val_model(model, val_loader, criterion, decoder, device)
            # Checkpoint the model after each epoch.
            finalValAcc = '%.3f'%(val_acc)
            model_path = os.path.join(MODEL_PATH, 'model_{}_val_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), finalValAcc))
            torch.save(model.state_dict(), model_path)
            print('='*20)
            scheduler.step(val_loss)
    else:
        # Only testing the model.
        test_model(model, test_loader, decoder, device)
        print('='*20)
