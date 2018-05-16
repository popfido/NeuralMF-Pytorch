#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

from __future__ import print_function
from __future__ import absolute_import

from data_loaders.cf_dl import CFDataLoader
from models.NeuralMF import NeuMF
from configs.NeuralMFConfig import NeuralMFConfig
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import CategoricalAccuracy


def train(**kwargs):
    """
    训练模型

    :return:
    """
    print('[INFO] 解析配置...')

    parser = None
    config = None

    try:
        parser = NeuralMFConfig()
        config = parser.parse_args(kwargs)
        # parser.save()
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        if parser:
            help()
        print('[Exception] 参考: python main.py -c configs/simple_mnist_config.json')
        exit(0)

    print('[INFO] 加载数据...')
    dl = CFDataLoader(config=config)

    print('[INFO] 构造网络...')
    model = NeuMF(config=config)

    callbacks = [EarlyStopping(patience=10),
                 ReduceLROnPlateau(factor=0.5, patience=5)]
    regularizers = [L1Regularizer(scale=1e-3, module_filter='conv*'),
                    L2Regularizer(scale=1e-5, module_filter='fc*')]
    constraints = [UnitNorm(frequency=3, unit='batch', module_filter='fc*')]
    initializers = [XavierUniform(bias=False, module_filter='fc*')]
    metrics = [CategoricalAccuracy(top_k=3)]

    print('[INFO] 训练网络...')

    trainer = ModuleTrainer(
        model=model
        )
    trainer.compile(loss='nll_loss',
                    optimizer='adadelta',
                    regularizers=regularizers,
                    constraints=constraints,
                    initializers=initializers,
                    metrics=metrics,
                    callbacks=callbacks)
    trainer.fit_loader(dl.get_train_data(), dl.get_test_data(), num_epoch=config.num_epochs,
                       verbose=1)
    print('[INFO] 训练完成...')


def help():
    """
    print help info： python file.py help
    """

    print(
        '''
        usage : python file.py <function> [--args=value]
        <function> := train | test | help
        example: 
            python {0} train data -e='env0701' --lr=0.01
            python {0} test data='path/to/dataset/root/'
            python {0} help
        avaiable args:
        '''.format(__file__))
    print(NeuralMFConfig().print_args())


if __name__ == '__main__':
    import fire

    fire.Fire()
