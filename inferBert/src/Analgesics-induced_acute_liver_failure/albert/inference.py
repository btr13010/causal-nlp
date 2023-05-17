import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from utils import TrainConfig, CustomDataset, data_preparation
from model import EndPointClassifier

output_dir = '../../../dat/Analgesics-induced_acute_liver_failure/output/'
config = TrainConfig()

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy() # detach tensor from computation graph (gradients are not tracked), save to cpu, and convert to numpy array

def test_prediction(net, device, dataloader, with_labels=True, result_file=f"{output_dir}/test_results.tsv"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(i)+'\t'+str(probs_all[i])+'\n' for i in range(len(probs_all)))
    w.close()

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_dir):
        print("Creation of the output' folder...")
        os.makedirs(output_dir)

    # Load Data
    ANAL_DATA_DIR = "../../../dat/Analgesics-induced_acute_liver_failure/proc"
    _, df_test, _, _ = data_preparation(ANAL_DATA_DIR)

    print("Reading test data...")
    test_set = CustomDataset(df_test, config.maxlen, config.bert_model)
    test_loader = DataLoader(test_set, batch_size=config.bs, num_workers=5)

    # Load the trained model
    path_to_model = '/models/albert-base-v2_lr_2e-05_val_loss_0.35007_ep_3.pt'
    model = EndPointClassifier(config.bert_model)
    if (device == 'cuda:0') and (torch.cuda.device_count() > 1):  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)

    # Predictions on test set
    path_to_output_file = f'{output_dir}/test_results.tsv'
    print("Predicting on test data...")
    test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True, result_file=path_to_output_file)
    print()
    print("Predictions are available in : {}".format(path_to_output_file))