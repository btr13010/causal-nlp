# InferBERT Reimplementation

Reimplementation of the [InferBERT model](https://www.frontiersin.org/articles/10.3389/frai.2021.659622/full) to preprocess the Analgesics dataset, inference the probabilities using ALBERT, and applying causal inference to the features.

# How to run
Data Preparation
```bash
python data_preprocessing.py
```

Run ALBERT
```bash
python ./albert/inference.py
```

Run Causal Inference
```bash
python causal_inference.py
```