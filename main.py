#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

from __future__ import print_function
from __future__ import absolute_import
import sys, time

from data_loaders.cf_dl import CFDataLoader
from models import NeuralMF, MultiLayerPerceptron, GeneralizedMatrixFactorization
from models.ModuleUtils import RankingModulelTrainer
from configs.NeuralMFConfig import NeuralMFConfig
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from models.ModuleUtils import HitAccuracy, NDCGAccuracy

models = ['NeuralMF', 'MultiLayerPerceptron', 'GeneralMatrixFactorization']


def implicit_load_model(model_name):
    if model_name not in models:
        raise AttributeError('No such model')
    return globals()[model_name]


def train(kwargs):
    """
    Training Process

    :return:
    """
    print('[INFO] Loading Settings...')

    parser = None
    config = None

    try:
        parser = NeuralMFConfig()
        # print(kwargs)
        if '-c' in kwargs:
            args = list(enumerate(kwargs))
            for id, arg in args:
                if '-c' == arg and len(args) >= id+2:
                    config = parser.get_args_from_json(args[id+1][1])
                    break
                else:
                    raise AssertionError("Corresponding config arg not found")
        else:
        config = parser.parse_args(kwargs)
    except Exception as e:
        print('[Exception] Unavailable Settings, %s' % e)
        if parser:
            help(kwargs)
        if '-c' in kwargs:
            print('[Exception] Please refer formatting: python main.py -c configs/neuralMF_config.json')
        exit(0)

    print('[INFO] Loading Data...')
    dl = CFDataLoader(config=config)

    print('[INFO] Build Networks...')
    nb_users, nb_items = dl.get_num_user_and_item()
    parser.args.nb_users = nb_users
    parser.args.nb_items = nb_items
    model = implicit_load_model(config.model)(config, nb_users, nb_items)
    print(model)
    callbacks = [EarlyStopping(patience=10),
                 ReduceLROnPlateau(factor=0.5, patience=5)]
    regularizers = [L2Regularizer(scale=1e-5, module_filter='fc*')]
    constraints = [UnitNorm(frequency=3, unit='batch', module_filter='fc*')]
    initializers = []
    metrics = [HitAccuracy(config.topk),
               NDCGAccuracy(config.topk)]

    print('[INFO] Begin Training...')
    time_str = time.strftime('%m%d_%H:%M:%S.pth')

    trainer = RankingModulelTrainer(
        model=model
        )
    trainer.compile(loss="binary_cross_entropy_with_logits",
                    optimizer='Adam',
                    regularizers=regularizers,
                    constraints=constraints,
                    initializers=initializers,
                    metrics=metrics,
                    callbacks=callbacks)
    trainer.fit_loader(dl.get_train_data(), dl.get_test_data(), num_epoch=config.epochs,
                       verbose=1)
    print('[INFO] Complete Training...')
    model.save(time_str=time_str)
    parser.save(timestamp=time_str)
    print('[INFO] Saved Model into checkpoint directory')


def help():
    """
    print help info： python file.py help
    """

    print(
        '''
        usage : python main.py <function> [--args=value]
        <function> := train | test | help
        example: 
            python main.py train path/to/dataset/root/ --lr=0.01
            python main.py test path/to/dataset/root/
            python main.py help
        avaiable args:
        '''.format(__file__))
    print(NeuralMFConfig().print_args())


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test', 'help']:
        help()
        exit(1)
    func = globals()[sys.argv[1]]
    func(sys.argv[2:])
