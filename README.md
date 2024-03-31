# Causality-Informed Text Classification

## Description

Full flow for text classification utilizing counterfactual-informed text embedding. 

## How to run
### Setup Environment
```bash
pip install -r requirements.txt
```

### Generate counterfactual data

Follow instructions in [counterfactual-generation](./counterfactual-generation/README.md)

### Train a classifier using triplet loss in counterfactual data

```bash
cd contrastive-embedding
bash scripts/train.sh
```