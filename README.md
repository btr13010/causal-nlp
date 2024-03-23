# Causality-Informed Text Classification

## Description

Full flow for estimating causal effect utilizing counterfactual-informed text embedding. The estimation is based on the causal transportability property of causal mechanism. 

## TODO

**REMEMBER** to train the initial counterfactual-generation first before moving to adding new loss terms to compare performance between the two methods

- [ ] Train/Generate counterfactual data
- [ ] Data exploration, planning to use a common dataset for all 3 tasks
- [ ] Modify embedding training code to use T5 model (instead of BERT) that is pretrained during the counterfactual data generation 
- [ ] Embedding training without ATT and NCE labels using contrastive representation learning
- [ ] classifier on top of the text embedding model (current code is for image, modify for text)

## How to run
### Setup Environment
```bash
conda env create -f environment.yml
conda activate causal-nlp
```