#!/usr/bin/env python
# -- coding: utf-8 --

from argparse import ArgumentParser
import datetime

from bases.BaseConfig import BaseConfig
from utils.utils import mkdir_if_not_exist


class NeuralMFConfig(BaseConfig):
    def __init__(self):
        parser = ArgumentParser(description="Train a Nerual Collaborative"
                                            " Filtering model")
        parser.add_argument('data', type=str,
                            help='path to test and training data files')
        parser.add_argument('-e', '--epochs', type=int, default=20,
                            help='number of epochs for training')
        parser.add_argument('-b', '--batch-size', type=int, default=256,
                            help='number of examples for each iteration')
        parser.add_argument('-f', '--factors', type=int, default=8,
                            help='number of predictive factors')
        parser.add_argument('--layers', nargs='+', type=int,
                            default=[64, 32, 16, 8],
                            help='size of hidden layers for MLP')
        parser.add_argument('-n', '--negative-samples', type=int, default=4,
                            help='number of negative examples per interaction')
        parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                            help='learning rate for optimizer')
        parser.add_argument('-k', '--topk', type=int, default=10,
                            help='rank for test examples to be considered a hit')
        parser.add_argument('--no-cuda', action='store_true',
                            help='use available GPUs')
        parser.add_argument('--seed', '-s', type=int,
                            help='manually set random seed for torch')
        parser.add_argument('--threshold', '-t', type=float,
                            help='stop training early at threshold')
        self.parser = parser
        self.args = None

    def parse_args(self, args=None):
        self.args = self.parser.parse_args(args)
        return self.args

    def print_help(self):
        return self.parser.print_help()

    def save(self):
        if self.args is None:
            raise ValueError("Did not parse any arg")
        config = {k: v for k, v in self.args.__dict__.items()}
        config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
        config['local_timestamp'] = str(datetime.now())
        run_dir = "./run/neumf/{}".format(config['timestamp'])
        print("Saving config and results to {}".format(run_dir))
        mkdir_if_not_exist(run_dir)
        utils.save_config(config, run_dir)