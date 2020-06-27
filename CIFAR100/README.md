## CIFAR-100 experiments

The code generates Table 1 in the [paper](https://arxiv.org/abs/1906.07405).

### Requirement
1. Python 3.6
2. PyTorch 1.0.0 with GPU support
3. TensorboardX

### Usage

#### Generate data
`python cifar100_to_np.py`

#### Test the performance of the compared methods
- SGD: `python sgd.py`
- [MSGD-Fisher]-B: `python sgdF.py`

#### Hyperparameters
See the [paper](https://arxiv.org/abs/1906.07405).