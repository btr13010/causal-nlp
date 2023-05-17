import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import random
import os
import tqdm

class TrainConfig:

    bert_model = "albert-base-v2"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
    freeze_bert = False  # if True, freeze the encoder weights and only update the classification layer weights

    maxlen = 128  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
    bs = 16  # batch size
    iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = 2e-5  # learning rate
    epochs = 2  # number of training epochs
    num_warmup_steps = 0 # The number of steps for the warmup phase.

    # # Official hyperparams
    # maxlen = 128
    # bs = 32
    # optimizer = adamw
    # num_warmup_steps = 200
    # lr = 2e-5
    # num_training_steps = 30000
    # save_checkpoints_steps = 3000

def data_preparation(DATA_DIR):
    train_data = pd.read_csv(DATA_DIR+"/train.tsv", sep='\t')
    test_data = pd.read_csv(DATA_DIR+"/test.tsv", sep='\t')
    dev_data = pd.read_csv(DATA_DIR+"/dev.tsv", sep='\t')
    features = pd.read_csv(DATA_DIR+"/feature.tsv", sep='\t')

    return train_data, test_data, dev_data, features

class CustomDataset(Dataset):
    ''' This class tokenizes the sentences '''

    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence at the specified index in the data frame
        sent = str(self.data.loc[index, 'sentence'])

        # Tokenize the sentence to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids
        
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count