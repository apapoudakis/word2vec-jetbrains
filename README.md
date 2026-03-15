# word2vec-jetbrains
JetBrains Internship Task

This project implements the Word2Vec Skip-gram model with Negative Sampling from scratch using NumPy. The code has three main scripts:.

- **train.py** Entry point for training, loads the yaml configuration and runs the training loop.
- **model.py** Skip-gram model with negative sampling, including forward pass, gradient computation, and weight updates.
- **utils.py** Data preparation utilities, tokenization, vocabulary building, subsampling, context–target pair generation, and negative sampling.


### Environment

Requires Python 3.9+ and the following dependencies:
```
pip install numpy pytest
```


### Data

Download a training corpus (or use the provided text8.txt file) and set its path in `config/train.yaml`.


### Training 

You can specify all the word2vec hyperparameters (e.g., embed_dim, epochs) in train.yaml file. 

```
python train.py --config /../config/train.yaml
```

### References

- Mikolov, T. et al. *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781, 2013.
