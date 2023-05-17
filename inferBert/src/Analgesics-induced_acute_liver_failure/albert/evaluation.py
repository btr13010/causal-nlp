from datasets import load_metric
import torch
from torch.utils.data import DataLoader
import pandas as pd
import tqdm

from utils import data_preparation, TrainConfig, CustomDataset
from inference import get_probs_from_logits
from model import EndPointClassifier

def evaluate_loss(net, device, dataloader, label_dev):
    """
    Evaluate the loss on a dataset
    """

    net.eval()
    probs_all = []

    with torch.no_grad():
        for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
            seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
            probs_all += probs.tolist()
            
    probs_all = pd.Series(probs_all)
    threshold = 0.5 
    preds = (probs_all >= threshold).astype('uint8') # predicted labels using the above fixed threshold

    metric = load_metric("glue", "mrpc")
    # Compute the accuracy and F1 scores
    score = metric._compute(predictions=preds, references=label_dev)
    print(f"Accuracy: {score['accuracy']:.3f}")
    print(f"F1: {score['f1']:.3f}")

if __name__ == '__main__':

    config = TrainConfig()

    # Load Data
    ANAL_DATA_DIR = "../../../dat/Analgesics-induced_acute_liver_failure/proc"
    _, _, anal_dev_data, _ = data_preparation(ANAL_DATA_DIR)
    label_dev = anal_dev_data['label']  # true labels
    dev_data = CustomDataset(anal_dev_data, config.maxlen, config.bert_model)
    dataloader = DataLoader(dev_data, batch_size=config.bs, num_workers=5)

    # Setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_to_model = '/models/albert-base-v2_lr_2e-05_val_loss_0.35007_ep_3.pt'
    net = EndPointClassifier(config.bert_model, freeze_bert=config.freeze_bert)
    net.load_state_dict(torch.load(path_to_model))
    net.to(device)

    # Evaluate
    evaluate_loss(net, device, dataloader, label_dev)