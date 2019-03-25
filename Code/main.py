import os
import csv
import time
import nltk
import torch
import argparse
import ctcdecode
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
DEFAULT_TEST_BATCH_SIZE = 128

# Hyperparameters.
LEARNING_RATE = 1e-2
LEARNING_RATE_DECAY = 0.1
WEIGHT_DECAY = 5e-5
WARM_UP_EPOCHS = 3

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
    return PHONEME_MAP[ph_int]

def generate_phoneme_string(batch_pred):
    # Loop over entire batch list of phonemes and convert them to strings.
    batch_strings = []
    for pred in batch_pred:
        batch_strings.append(''.join(list(map(map_phoneme_string, list(pred.cpu().numpy())))))
    return batch_strings

def calculate_edit_distance(pred, targets):
    #print('len(pred):', len(pred))
    #print('len(targets):', len(targets))
    assert len(pred) == len(targets)
    dist = []
    for idx, p in enumerate(pred):
        dist.append(nltk.edit_distance(p, targets[idx]))
    return dist

def val_model(model, val_loader, criterion, decoder, device):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        y_pred = []
        tar_str = []
        start_time = time.time()
        for batch_idx, (inputs, target) in enumerate(val_loader):
            # Move all inputs and targets to device.
            for d_idx in range(len(inputs)):
                inputs[d_idx] = inputs[d_idx].to(device)
                targets[d_idx] = targets[d_idx].to(device)
            inp_lens = [len(inp) for inp in inputs]
            tar_lens = [len(tar) for tar in targets]    # Do I need to add 1 to each target to account for blank?
            outputs = model(inputs)
            # Change shape from (Batch, Max_Seq_L, Features) to (Max_Seq_L, Batch, Features) for CTC Loss.
            loss = criterion(F.log_softmax(outputs, dim=2).permute(1,0,2), torch.cat(targets), inp_lens, tar_lens)    # CTCLoss only!
            running_loss += loss.item()
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            pad_out, _, _, out_lens = decoder.decode(outputs, torch.tensor(inp_lens))
            # Iterate over each item in batch.
            for i in range(len(inp_lens)):
                # Pick the first output, most likely.
                y_pred.append(pad_out[i, 0, :out_lens[i, 0]] - 1)
            # Calculate the strings for targets.
            tar_str.extend(generate_phoneme_string(targets))
            print('Validation Iteration: %d/%d Loss = %5.4f' % \
                    (batch_idx+1, len(val_loader), (running_loss/(batch_idx+1))), \
                    end="\r", flush=True)

        end_time = time.time()
        # Calculate the strings for predictions.
        pred_str = generate_phoneme_string(y_pred)
        # Calculate edit distance between predictions and targets.
        dist = calculate_edit_distance(pred_str, tar_str)
        acc = (sum(dist)/len(dist))*100.0
        running_loss /= len(val_loader)
        print('\nValidation Loss: %5.4f Inv Validation Accuracy: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
        return running_loss, acc

def train_model(model, train_loader, criterion, optimizer, decoder, device):
    model.train()
    running_loss = 0.0
    y_pred = []
    tar_str = []
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move all inputs and targets to device.
        for d_idx in range(len(inputs)):
            inputs[d_idx] = inputs[d_idx].to(device)
            targets[d_idx] = targets[d_idx].to(device)
        inp_lens = [len(inp) for inp in inputs]
        #print('inp_lens =', inp_lens)
        tar_lens = [len(tar) for tar in targets]    # Do I need to add 1 to each target to account for blank?
        optimizer.zero_grad()
        outputs = model(inputs)
        # Change shape from (Batch, Max_Seq_L, Features) to (Max_Seq_L, Batch, Features) for CTC Loss.
        loss = criterion(F.log_softmax(outputs, dim=2).permute(1,0,2), torch.cat(targets), inp_lens, tar_lens)    # CTCLoss only!
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
        # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
        pad_out, _, _, out_lens = decoder.decode(outputs, torch.tensor(inp_lens))
        # Iterate over each item in batch.
        for i in range(len(inp_lens)):
            # Pick the first output, most likely.
            y_pred.append(pad_out[i, 0, :out_lens[i, 0]] - 1)
        # Calculate the strings for targets.
        #print('len(targets):', len(targets))
        #print('targets[0]:', targets[0])
        #print('targets[1]:', targets[1])
        tar_str.extend(generate_phoneme_string(targets))
        print('Train Iteration: %d/%d Loss = %5.4f' % \
                (batch_idx+1, len(train_loader), (running_loss/(batch_idx+1))), \
                end="\r", flush=True)
    end_time = time.time()
    # Calculate the strings for predictions.
    pred_str = generate_phoneme_string(y_pred)
    #print('tar_str=', tar_str)
    #print('pred_str=', pred_str)
    # Calculate edit distance between predictions and targets.
    dist = calculate_edit_distance(pred_str, tar_str)
    #print('dist:', dist)
    acc = (sum(dist)/len(dist))*100.0
    running_loss /= len(train_loader)
    #print('\nTraining Loss: %5.4f Time: %d s' % (running_loss, end_time - start_time))
    print('\nTraining Loss: %5.4f Inv Training Accuracy: %5.4f Time: %d s' % (running_loss, acc, end_time - start_time))
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

    # Create the model.
    input_size = DEFAULT_FEATURE_SIZE
    vocab_size = len(PHONEME_MAP) + 1   # Plus 1 for blank.
    hidden_size = 100
    nlayers = 3
    model = SpeechRecognizer(input_size, hidden_size, vocab_size, nlayers)
    model.to(device)
    print('='*20)
    print(model)
    print("Running on device = %s." % (device))

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

    # Setup learning parameters.
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = LEARNING_RATE_DECAY)
    decoder = ctcdecode.CTCBeamDecoder([' ']+PHONEME_MAP, beam_width=10, log_probs_input=True)

    n_epochs = 50
    print('='*20)

    if args.mode == 'train':
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch+1,n_epochs))
            train_loss = train_model(model, train_loader, criterion, optimizer, decoder, device)
            val_loss, val_acc = val_model(model, val_loader, criterion, decoder, device)
            # Checkpoint the model after each epoch.
            finalValAcc = '%.3f'%(Val_acc[-1])
            model_path = os.path.join(MODEL_PATH, 'model_{}_val_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), finalValAcc))
            torch.save(model.state_dict(), model_path)
            print('='*20)
            if epoch >= WARM_UP_EPOCHS:
                scheduler.step()
