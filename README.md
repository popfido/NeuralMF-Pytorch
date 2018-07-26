# NeuralMF-Pytorch

This is an Implementation of paper [Neural Collaborate Filtering](https://arxiv.org/abs/1708.05031) use Personal Pytorch 
Template.

It is inspired by [Meitu DL-Project-Template](https://github.com/SpikeKing/DL-Project-Template) 
but using ``pytorch`` instead of ``tensorflow`` for pytorch researchers/developers.

It also used a personalized version of ``torchsample`` which contains series of Keras API for ``pytorch``.
You can find it in my personal directory: [Torchsample](https://github.com/popfido/torchsample).

By H. L. Wang 

## Usage

```text
git clone https://github.com/popfido/DL-Project-Template
```

Build and activate of virtualenv

```bash
# Through Virtualenv
virtualenv venv
source venv/bin/activate

# Or Conda
conda create -n venv
```

Install Python Dependecy

```bash
git clone https://github.com/popfido/torchsample

cd torchsample && python setup.py install

pip install -r requirements.txt
```
Noted that the requirement version of ``torchsample`` is 0.2.0, which is a 
special version maintained by myself for the reason that the official torchsample
maintainer has been out of maintain for half an year.

```bash
# use get_movielen.py to get movielen dataset for testing
python data/get_movielen.py

# See help for main script usage
python main.py -help
```

### Directory Structure

```text
├── bases
│   ├── BaseDataLoader.py               - BaseDataLoader Class
│   ├── BaseConfig.py                   - BaseConfig Class
│   └── BaseModule.py                   - BaseModule Class
├── configs                             - Config Directory
│   └── neuralMF_config.json
├── data_loaders                        - DataLoader Class Directory
│   ├── __init__.py
│   ├── simple_mnist_dl.py
├── main.py                             - Main Class
├── models                              - Module Directory
│   ├── __init__.py
│   ├── NeuralMF.py                     - Neural Collaborate Matrix Factorization Module
│   ├── MultilayerPerceptron.py         - Multi-Layer Perceptron Module
│   ├── GeneralizedMF.py                - Generalized Matrix Factorization Module
│   └── ModuleUtils.py                  - Utils Module for other Module
├── requirements.txt                    - Dependencies
└── utils                               - Utils Directory
    ├── __init__.py
    ├── config_utils.py                 - Config Utils
    ├── np_utils.py                     - NumPy Utils
    └── utils.py                        - Other Utils
```

## Main Component

### DataLoader

How to：

1. Create your own DataLoade with BaseDataLoader.
2. Implement ``get_train_data()`` and ``get_test_data()`` Methods；

### Module

How to：

1. Create your own Network Module with BaseModule (also )；
2. Implement ``__init__()`` and ``forward()`` to create the NN strucure you want；

### Config

Define all parameter during training by JSON or from commandline.





