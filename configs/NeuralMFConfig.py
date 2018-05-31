#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

from argparse import ArgumentParser, HelpFormatter
import os as _os
import sys as _sys
import datetime

from bases.BaseConfig import BaseConfig
from utils.configUtils import save_config
from utils.utils import mkdir_if_not_exist


class NeuralMFConfig(BaseConfig):
    def __init__(self):
        parser = ArgumentParser(description="Train a Nerual Collaborative"
                                            " Filtering model")
        parser.add_argument('data', type=str,
                            help='path to directory of test and training data files')
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
        self.prog = _os.path.basename(_sys.argv[0])
        self.formatter_class = HelpFormatter

    def parse_args(self, args=None):
        self.args = self.parser.parse_args(args)
        return self.args

    def print_args(self):
        formatter = self._get_formatter()
        # positionals, optionals and user-defined groups
        for action_group in self.parser._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        # epilog
        formatter.add_text(self.parser.epilog)
        args_list = formatter.format_help().split('\n')
        args_list.pop(4)
        return "\n".join(["        " + arg for arg in args_list])

    def _get_formatter(self):
        return self.formatter_class(prog=self.prog)

    def save(self):
        if self.args is None:
            raise ValueError("Did not parse any arg")
        config = {k: v for k, v in self.args.__dict__.items()}
        config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
        config['local_timestamp'] = str(datetime.now())
        run_dir = "./run/neumf/{}".format(config['timestamp'])
        print("Saving config and results to {}".format(run_dir))
        mkdir_if_not_exist(run_dir)
        save_config(config, run_dir)