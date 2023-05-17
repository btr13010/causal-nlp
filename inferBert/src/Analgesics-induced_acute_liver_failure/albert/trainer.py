import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm
import copy
import numpy as np

from utils import CustomDataset, evaluate_loss, TrainConfig, set_seed, data_preparation
from model import SentencePairClassifier

set_seed(42)  # Set seed for reproducibility

if not os.path.exists('./models/'):
    print("Creation of the models' folder...")
    os.makedirs('./models/')

def train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits.squeeze(-1), labels.float())
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()


            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0


        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    path_to_model='models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss, 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()

ANAL_DATA_DIR = "../../../dat/Analgesics-induced_acute_liver_failure/proc"
anal_train_data, anal_test_data, anal_dev_data, anal_features = data_preparation(ANAL_DATA_DIR)
config = TrainConfig()

# Creating instances of training and validation set
print("Reading training data...")
train_set = CustomDataset(anal_train_data, config.maxlen, config.bert_model)
print("Reading validation data...")
val_set = CustomDataset(anal_dev_data, config.maxlen, config.bert_model)

# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=config.bs, num_workers=5)
val_loader = DataLoader(val_set, batch_size=config.bs, num_workers=5)

num_training_steps = config.epochs * len(train_loader)  # The total number of training steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = SentencePairClassifier(config.bert_model, freeze_bert=config.freeze_bert)

if torch.cuda.device_count() > 1:  # if multiple GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net.to(device)

criterion = nn.BCEWithLogitsLoss()

opti = AdamW(net.parameters(), lr=config.lr, weight_decay=1e-2)

t_total = (len(train_loader) // config.iters_to_accumulate) * config.epochs  # Necessary to take into account Gradient accumulation

lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=config.num_warmup_steps, num_training_steps=t_total)

train_bert(net, criterion, opti, config.lr, lr_scheduler, train_loader, val_loader, config.epochs, config.iters_to_accumulate)