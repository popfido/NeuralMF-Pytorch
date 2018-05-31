#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""

import torch as th
from torch import functional as F
from torchsample.metrics import Metric, MetricContainer, MetricCallback
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import TQDM, CallbackContainer
from torchsample.regularizers import RegularizerCallback
from torchsample.constraints import ConstraintCallback
from torchsample.modules.Helper import MultiInputNoTargetHelper, BaseHelper
import numpy as np
import functools
import math
import heapq


class Flatten(th.nn.Module):
    """
    flatten input to（batch_size,dim_length）
    """

    def __init__(self):
        super(Flatten, self).__init__()
        # self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)

class TwoInputSingleTargetHelper(BaseHelper):

    def move_to_cuda(self, cuda_device, user, item, targets):
        user = user.cuda(cuda_device)
        item = item.cuda(cuda_device)
        targets = targets.cuda(cuda_device)
        return user, item, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = targets[rand_indices]
        return inputs, targets

    def grab_batch_from_loader(self, loader_iter):
        user, item, label = next(loader_iter)
        return user, item, label

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = tforms[1](target_batch)
        return input_batch, target_batch

    def forward_pass(self, user, item, model):
        return model(user, item)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)

class MultiInputSingleTargetHelper(BaseHelper):

    def move_to_cuda(self, cuda_device, input_batch, targets):
        input_batch_ = [input_.cuda(cuda_device) for input_ in input_batch]
        return input_batch_, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = [th.tensor(input_[batch_idx * batch_size:(batch_idx + 1) * batch_size], requires_grad=True)
                       for input_ in inputs]
        return input_batch, targets

    def grab_batch_from_loader(self, loader_iter):
        input_batch, targets = next(loader_iter)
        return input_batch, targets

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = tforms[1](target_batch)
        return input_batch, target_batch

    def forward_pass(self, user, item, model):
        return model(user, item)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)

class RankingModulelTrainer(ModuleTrainer):

    def __init__(self, model):
        """
        ModelTrainer for high-level training of Pytorch models

        Major Parts
        -----------
        - optimizer(s)
        - loss(es)
        - regularizers
        - initializers
        - constraints
        - metrics
        - callbacks
        """
        super(RankingModulelTrainer, self).__init__(model)
        self.train_metric = False

    def fit_loader(self,
                   loader,
                   val_loader=None,
                   initial_epoch=0,
                   num_epoch=100,
                   cuda_device=-1,
                   verbose=1):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """
        self.model.train()
        # ----------------------------------------------------------------------
        num_inputs, num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        batch_size = loader.batch_size

        if val_loader is not None:
            num_val_inputs, num_val_targets = 1, 1
            if (num_inputs != num_val_inputs) or (num_targets != num_val_targets):
                raise ValueError('num_inputs != num_val_inputs or num_targets != num_val_targets')
        has_val_data = val_loader is not None
        num_batches = int(math.ceil(len_inputs / batch_size))
        # ----------------------------------------------------------------------

        fit_helper = TwoInputSingleTargetHelper()
        fit_loss_fn = fit_helper.get_partial_loss_fn(self._loss_fn)
        fit_forward_fn = fit_helper.get_partial_forward_fn(self.model)

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)
            if self._has_regularizers:
                tmp_callbacks.append(RegularizerCallback(self.regularizer_container))
                fit_loss_fn = _add_regularizer_to_loss_fn(fit_loss_fn,
                                                          self.regularizer_container)
            if self._has_constraints:
                tmp_callbacks.append(ConstraintCallback(self.constraint_container))
            if self._has_metrics:
                self.metric_container.set_helper(fit_helper)
                tmp_callbacks.append(MetricCallback(self.metric_container))

            callback_container = CallbackContainer(self._callbacks + tmp_callbacks)
            callback_container.set_trainer(self)
            callback_container.on_train_begin({'batch_size': loader.batch_size,
                                               'num_batches': num_batches,
                                               'num_epoch': num_epoch,
                                               'has_val_data': has_val_data,
                                               'has_regularizers': self._has_regularizers,
                                               'has_metrics': self._has_metrics})

            for epoch_idx in range(initial_epoch, num_epoch):
                epoch_logs = {}
                callback_container.on_epoch_begin(epoch_idx, epoch_logs)

                batch_logs = {}
                loader_iter = iter(loader)
                for batch_idx in range(num_batches):

                    callback_container.on_batch_begin(batch_idx, batch_logs)

                    user, item, label = fit_helper.grab_batch_from_loader(loader_iter)
                    if cuda_device >= 0:
                        user, item, label = fit_helper.move_to_cuda(cuda_device, user, item, label)

                    # ---------------------------------------------
                    self._optimizer.zero_grad()
                    output_batch = fit_forward_fn(user, item)
                    loss = fit_loss_fn(output_batch, label)
                    loss.backward()
                    self._optimizer.step()
                    # ---------------------------------------------

                    if self._has_regularizers:
                        batch_logs['reg_loss'] = self.regularizer_container.current_value
                    if self.train_metric and self._has_metrics:
                        metrics_logs = self.metric_container(output_batch, label)
                        batch_logs.update(metrics_logs)

                    batch_logs['loss'] = loss.item()
                    callback_container.on_batch_end(batch_idx, batch_logs)

                epoch_logs.update(self.history.batch_metrics)
                if has_val_data:
                    val_epoch_logs = self.evaluate_loader(val_loader,
                                                          cuda_device=cuda_device,
                                                          verbose=verbose)
                    self._in_train_loop = False
                    # self.history.batch_metrics.update(val_epoch_logs)
                    # epoch_logs.update(val_epoch_logs)
                    epoch_logs.update(val_epoch_logs)
                    epoch_logs.update(batch_logs)
                    # TODO how to fix this?
                    # self.history.batch_metrics.update(val_epoch_logs)

                callback_container.on_epoch_end(epoch_idx, epoch_logs)

                if self._stop_training:
                    break
        self.model.eval()

    def evaluate_loader(self,
                        data,
                        cuda_device=-1,
                        verbose=1):
        self.model.eval()
        # num_inputs, num_targets = _parse_num_in(data)
        batch_size = 1
        len_inputs = len(data)
        num_batches = int(math.ceil(len_inputs / batch_size))

        evaluate_helper = MultiInputSingleTargetHelper()
        eval_loss_fn = evaluate_helper.get_partial_loss_fn(self._loss_fn)
        eval_forward_fn = evaluate_helper.get_partial_forward_fn(self.model)
        eval_logs = {}

        if self._has_metrics:
            metric_container = MetricContainer(self._metrics, prefix='val_')
            metric_container.set_helper(evaluate_helper)
            metric_container.reset()

        batches_seen = 0
        with th.no_grad():
            for i in range(len_inputs):
                input_batch, target_batch = data[i]
                preds = []
                # loss_avg = []
                batches = [(th.tensor(input_batch[0][i:i + batch_size]),
                            th.tensor(input_batch[1][i:i + batch_size]))
                           for i in range(0, len(input_batch), batch_size)]
                for batch in batches:
                    if cuda_device >= 0:
                        batch = evaluate_helper.move_to_cuda(cuda_device, input_batch, target_batch)

                    output_batch = eval_forward_fn(batch[0], batch[1]).data.cpu().numpy()

                    # loss_avg.append(eval_loss_fn(output_batch, target_batch).item())
                    preds += list(output_batch.flatten())
                map_item_score = {item: pred for item, pred in zip(input_batch[1], preds)}

                batches_seen += 1
                # eval_logs['val_loss'] += np.mean(loss_avg)

                if self._has_metrics:
                    metrics_logs = metric_container(map_item_score, target_batch)
                    eval_logs.update(metrics_logs)
            #for k, v in eval_logs.items():
            #    eval_logs[k] = v / (batches_seen * 1.)
        self.model.train()
        return eval_logs


def BCE_with_logits_loss(input, target, weight=None, size_average=True, reduce=True):
    if weight is not None:
        return F.binary_cross_entropy_with_logits(input, target,
                                                  weight,
                                                  size_average,
                                                  reduce=reduce)
    else:
        return F.binary_cross_entropy_with_logits(input, target,
                                                  size_average=size_average,
                                                  reduce=reduce)


def _calculate_hit(ranked, gt_item):
    return int(gt_item in ranked)


def _calculate_ndcg(ranked, gt_item):
    for i, item in enumerate(ranked):
        if item == gt_item:
            return np.log(2) / np.log(i + 2)
    return 0.


def _add_regularizer_to_loss_fn(loss_fn,
                                regularizer_container):
    def new_loss_fn(output_batch, target_batch):
        return loss_fn(output_batch, target_batch) + regularizer_container.get_value()

    return new_loss_fn


def _parse_num_inputs_and_targets_from_loader(loader):
    """ NOT IMPLEMENTED """
    # batch = next(iter(loader))
    if isinstance(loader.dataset, th.utils.data.dataset.Dataset):
        num_inputs = num_targets = 1
    else:
        num_inputs = loader.dataset.num_inputs
        num_targets = loader.dataset.num_targets
    return num_inputs, num_targets


class HitAccuracy(Metric):

    def __init__(self, top_k=10):
        self.total_count = 0.
        self.top_k = top_k
        self.hits = 0.

        self._name = 'hit_metric'

    def reset(self):
        self.total_count = 0.
        self.hits = 0.

    def __call__(self, y_pred, y_true):
        ranked = heapq.nlargest(self.top_k, y_pred, key=y_pred.get)
        self.hits += _calculate_hit(ranked, y_true)
        self.total_count += 1
        return self.hits / self.total_count


class NDCGAccuracy(Metric):

    def __init__(self, top_k=10):
        self.total_count = 0.
        self.top_k = top_k
        self.ndcg = 0.

        self._name = 'ndcg_metric'

    def reset(self):
        self.total_count = 0.
        self.ndcg = 0.

    def __call__(self, y_pred, y_true):
        ranked = heapq.nlargest(self.top_k, y_pred, key=y_pred.get)
        self.ndcg += _calculate_ndcg(ranked, y_true)
        self.total_count += 1
        return self.ndcg / self.total_count


class HitAndNDCGAccuracy(Metric):
    def __init__(self, top_k=5):
        self.hit_count = 0.
        self.ndcg_sum = 0.
        self.total_count = 0.
        self.top_k = top_k

        self._name = 'hit_metric'

    def reset(self):
        self.hit_count = 0.
        self.ndcg_sum = 0.
        self.total_count = 0.

    def __call__(self, y_pred, y_true):
        ranked = heapq.nlargest(self.top_k, y_pred, key=y_pred.get)
        self.hit_count += _calculate_hit(ranked, y_true)
        self.ndcg_sum += _calculate_ndcg(ranked, y_true)
        self.total_count += 1
        return self.hit_count / self.total_count, self.ndcg_sum / self.total_count