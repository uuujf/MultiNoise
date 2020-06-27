## SVHN experiments

The code generates Figure 1 (b), Figure 2 (a) and Figure 3 (b) in the [paper](https://arxiv.org/abs/1906.07405).

### Requirement
1. Python 3.6
2. PyTorch 1.0.0 with GPU support
3. TensorboardX

### Usage

#### Generate data
`python svhn2np.py`

#### Test the performance of the compared methods
- GD: `python gd.py`
- SGD: `python sgd.py`
- MSGD-Fisher: `python ggd.py`
- MSGD-Cov: `python ggdCov.py`
- MSGD-Bernoulli: `python bgd.py`
- MSGD-[Fisher-B]: `python gdStoF.py`
- MSGD-[Cov-B]: `python gdStoCov.py`
- [MSGD-Fisher]-B: `python sgdF.py`
- [MSGD-Cov]-B: `python sgdCov.py`
- GLD Diag (from this [paper](https://arxiv.org/abs/1803.00195)): `python diagF.py`

#### Hyperparameters
See the [paper](https://arxiv.org/abs/1906.07405).