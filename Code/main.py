import os
import csv
import time
import nltk
import torch
import argparse
import ctcdecode
import numpy as np
from phoneme_list import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import SpeechRecognizer
from torch.utils.data import DataLoader
from dataset import SpeechDataset, SpeechCollateFn

# Paths
MODEL_PATH = './../Models'
TEST_RESULT_PATH = './../Results'

# Defaults
DEFAULT_RUN_MODE = 'train'
DEFAULT_FEATURE_SIZE = 40
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 64
DEFAULT_LABEL_MAP = [" "] + PHONEME_MAP
DEFAULT_GRADIENT_CLIP = 1

# Hyperparameters.
LEARNING_RATE = 1e-2
LEARNING_RATE_DECAY = 0.1
WEIGHT_DECAY = 5e-5
WARM_UP_EPOCHS = 3

# Model Ensembles Parameters
MODEL_1_PARAMS = (DEFAULT_FEATURE_SIZE, 100, len(DEFAULT_LABEL_MAP), 3, DEFAULT_TEST_BATCH_SIZE, False, 0.0)    #SpeechRecognizer
MODEL_2_PARAMS = (DEFAULT_FEATURE_SIZE, 100, len(DEFAULT_LABEL_MAP), 3, DEFAULT_TEST_BATCH_SIZE, True, 0.0)     #SpeechRecognizer
MODEL1_1 = './../Models/Model_1/model_20190326-173252_val_77.725.pt'
MODEL1_2 = './../Models/Model_1/model_20190326-174258_val_78.036.pt'
MODEL1_3 = './../Models/Model_1/model_20190326-175301_val_78.080.pt'
MODEL1_4 = './../Models/Model_1/model_20190326-180302_val_78.076.pt'
MODEL1_5 = './../Models/Model_1/model_20190326-181304_val_78.073.pt'

MODEL2_1 = './../Models/Model_2/model_20190327-160916_val_83.800.pt'
MODEL2_2 = './../Models/Model_2/model_20190327-162036_val_84.113.pt'
MODEL2_3 = './../Models/Model_2/model_20190327-163152_val_84.071.pt'
MODEL2_4 = './../Models/Model_2/model_20190327-164320_val_84.099.pt'
MODEL2_5 = './../Models/Model_2/model_20190327-165450_val_84.100.pt'

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Speech Recognition.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default=DEFAULT_RUN_MODE, help='\'train\' or \'test\' mode.')
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE, help='Testing batch size.')
    parser.add_argument('--model_path', type=str, help='Path to model to be reloaded.')
    parser.add_argument('--model_ensemble', type=bool, default=False, help='True/false, if we have to model ensembling.')
    return parser.parse_args()

# For transfer learning.
def load_weights(model, model_path, device):
    pretrained_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    model_params = len(model_dict)
    pretrained_params = len(pretrained_dict)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    skip_count = 0
    bad_keys = [key for key in pretrained_dict.keys() if pretrained_dict[key].size() != model_dict[key].size()]
    for key in bad_keys:
        del pretrained_dict[key]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print('Loaded model:', model_path)
    print('Skipped %d/%d params from pretrained for %d params in model.' \
            % (len(bad_keys), pretrained_params, model_params))
    return model

def map_phoneme_string(ph_int):
    return DEFAULT_LABEL_MAP[ph_int]

def generate_phoneme_string(batch_pred):
    # Loop over entire batch list of phonemes and convert them to strings.
    batch_strings = []
    for pred in batch_pred:
        batch_strings.append(''.join(list(map(map_phoneme_string, list(pred.numpy())))))
    return batch_strings

def calculate_edit_distance(pred, targets):
    assert len(pred) == len(targets)
    dist = []
    for idx, p in enumerate(pred):
        dist.append(nltk.edit_distance(p, targets[idx]))
    return dist

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

def test_model_ensemble(model, test_loader, decoder, device):
    with torch.no_grad():
        for m in models:
            m.eval()
            m.to(device)
        start_time = time.time()
        all_predictions = []
        for batch_idx, (inputs, inp_lens, seq_order) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = []
            for idx, m in enumerate(models):
                outputs.append(m(inputs, inp_lens))
            # Take mean of probabilities from each model.
            outputs = torch.mean(torch.stack(outputs), dim=0)
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            pad_out, _, _, out_lens = decoder.decode(outputs, torch.tensor(inp_lens))
            # Iterate over each item in batch.
            y_pred = []
            for i in range(len(inp_lens)):
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

def test_model(model, test_loader, decoder, device):
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        all_predictions = []
        for batch_idx, (inputs, inp_lens, seq_order) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs, inp_lens)
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            pad_out, _, _, out_lens = decoder.decode(outputs, torch.tensor(inp_lens))
            # Iterate over each item in batch.
            y_pred = []
            for i in range(len(inp_lens)):
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

def val_model_ensemble(models, val_loader, criterion, decoder, device):
    with torch.no_grad():
        for m in models:
            m.eval()
            m.to(device)
        running_loss = 0.0
        dist = []
        start_time = time.time()
        for batch_idx, (inputs, inp_lens, targets) in enumerate(val_loader):
            targets_cat = (torch.cat(targets)).to(device)
            inputs = inputs.to(device)
            tar_lens = [len(tar) for tar in targets]
            outputs = []
            loss = 0
            for idx, m in enumerate(models):
                outputs.append(m(inputs, inp_lens))
                print(outputs[idx].shape)
                # Change shape from (Batch, Max_Seq_L, Dict) to (Max_Seq_L, Batch, Dict) for CTC Loss.
                loss += criterion(F.log_softmax(outputs[idx], dim=2).permute(1,0,2), targets_cat, inp_lens, tar_lens)    # CTCLoss only!
            loss = loss/(len(models))
            running_loss += loss.item()
            # Take mean of probabilities from each model.
            outputs = torch.mean(torch.stack(outputs), dim=0)
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            pad_out, _, _, out_lens = decoder.decode(outputs, torch.tensor(inp_lens))
            # Iterate over each item in batch.
            y_pred = []
            for i in range(len(inp_lens)):
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
        acc = 100.0 - (sum(dist)/len(dist))     # Average over edit distance.
        running_loss /= len(val_loader)
        print('\nValidation Loss: %5.4f Validation Accuracy: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
        return running_loss, acc

def val_model(model, val_loader, criterion, decoder, device):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        dist = []
        start_time = time.time()
        for batch_idx, (inputs, inp_lens, targets) in enumerate(val_loader):
            targets_cat = (torch.cat(targets)).to(device)
            inputs = inputs.to(device)
            tar_lens = [len(tar) for tar in targets]
            outputs = model(inputs, inp_lens)
            # Change shape from (Batch, Max_Seq_L, Dict) to (Max_Seq_L, Batch, Dict) for CTC Loss.
            loss = criterion(F.log_softmax(outputs, dim=2).permute(1,0,2), targets_cat, inp_lens, tar_lens)    # CTCLoss only!
            running_loss += loss.item()
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            pad_out, _, _, out_lens = decoder.decode(outputs, torch.tensor(inp_lens))
            # Iterate over each item in batch.
            y_pred = []
            for i in range(len(inp_lens)):
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
        acc = 100.0 - (sum(dist)/len(dist))     # Average over edit distance.
        running_loss /= len(val_loader)
        print('\nValidation Loss: %5.4f Validation Accuracy: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
        return running_loss, acc

def train_model(model, train_loader, criterion, optimizer, decoder, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    dist = []
    for batch_idx, (inputs, inp_lens, targets) in enumerate(train_loader):
        targets_cat = (torch.cat(targets)).to(device)
        inputs = inputs.to(device)
        tar_lens = [len(tar) for tar in targets]
        optimizer.zero_grad()
        outputs = model(inputs, inp_lens)
        print(outputs.shape)
        # Change shape from (Batch, Max_Seq_L, Dict) to (Max_Seq_L, Batch, Dict) for CTC Loss.
        loss = criterion(F.log_softmax(outputs, dim=2).permute(1,0,2), targets_cat, inp_lens, tar_lens)    # CTCLoss only!
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #  torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRADIENT_CLIP)

        optimizer.step()
        running_loss += loss.item()
        # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
        # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
        pad_out, _, _, out_lens = decoder.decode(outputs, torch.tensor(inp_lens))
        # Iterate over each item in batch.
        y_pred = []
        for i in range(len(inp_lens)):
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
    acc = 100.0 - (sum(dist)/len(dist))
    running_loss /= len(train_loader)
    print('\nTraining Loss: %5.4f Training Accuracy: %5.4f Time: %d s' % (running_loss, acc, end_time - start_time))
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
    speechTrainDataset = SpeechDataset(mode='train', device=device)
    speechValDataset = SpeechDataset(mode='dev', device=device)
    speechTestDataset = SpeechDataset(mode='test', device=device)

    train_loader = DataLoader(speechTrainDataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=4, collate_fn=SpeechCollateFn)
    val_loader = DataLoader(speechValDataset, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=4, collate_fn=SpeechCollateFn)
    test_loader = DataLoader(speechTestDataset, batch_size=args.test_batch_size,
                        shuffle=False, num_workers=4, collate_fn=SpeechCollateFn)

    if args.model_ensemble:
        num_model_1 = 3
        num_model_2 = 3
        model_paths = [
            #MODEL1_1,
            MODEL1_2,
            MODEL1_3,
            MODEL1_4,
            #MODEL1_5,
            #MODEL2_1,
            MODEL2_2,
            MODEL2_3,
            MODEL2_4
            #MODEL2_5
        ]
        models = []
        input_size, hidden_size, vocab_size, nlayers, batch_size, bidirectional, dropout = MODEL_1_PARAMS
        for i in range(num_model_1):
            models += [SpeechRecognizer(input_size, hidden_size, vocab_size, nlayers, batch_size, bidirectional, dropout)]
        input_size, hidden_size, vocab_size, nlayers, batch_size, bidirectional, dropout = MODEL_2_PARAMS
        for i in range(num_model_2):
            models += [SpeechRecognizer(input_size, hidden_size, vocab_size, nlayers, batch_size, bidirectional, dropout)]
        # Load all models.
        for idx, m in enumerate(models):
            m.load_state_dict(torch.load(model_paths[idx], map_location=device))
            print('Loaded model:', model_paths[idx])
    else:
        # Create the model.
        input_size = DEFAULT_FEATURE_SIZE
        vocab_size = len(DEFAULT_LABEL_MAP)

        # Model_1 -> SpeechRecognizer
        # hidden_size = 100
        # nlayers = 3
        # bidirectional = False
        # dropout = 0

        # Model_2 -> SpeechRecognizer
        hidden_size = 100
        nlayers = 3
        bidirectional = True
        dropout = 0

        # Model_3 -> SpeechRecognizer -> Gives nan loss.
        # hidden_size = 100
        # nlayers = 3
        # bidirectional = True
        # dropout = 0.3

        model = SpeechRecognizer(input_size, hidden_size, vocab_size,
                                nlayers, args.train_batch_size, bidirectional, dropout)
        model.to(device)
        print('='*20)
        print(model)

    print("Running on device = %s." % (device))

    # Setup learning parameters.
    criterion = nn.CTCLoss()
    if not args.model_ensemble:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = LEARNING_RATE_DECAY)
        if args.model_path != None:
            model = load_weights(model, args.model_path, device)
    decoder = ctcdecode.CTCBeamDecoder(DEFAULT_LABEL_MAP, beam_width=10, log_probs_input=True)

    n_epochs = 50
    print('='*20)

    if args.model_ensemble:
        if args.mode == 'train':
            # Only validate in ensemble mode.
            val_loss, val_acc = val_model_ensemble(models, val_loader, criterion, decoder, device)
        else:
            test_model_ensemble(models, test_loader, decoder, device)
    else:
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
                if (epoch+1) >= WARM_UP_EPOCHS:
                    scheduler.step()
        else:
            # Only testing the model.
            test_model(model, test_loader, decoder, device)
    print('='*20)
