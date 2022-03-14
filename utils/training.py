# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import os
import numpy as np


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, eval_ema=False, ema_model=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :param eval_ema: flag to indicate if an exponential weighted average model
    should be evaluated (For CLS-ER)
    :param ema_model: if eval ema is set to True, which ema model (plastic or stable)
    should be evaluated (For CLS-ER)
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """

    curr_model = model.net
    if eval_ema:
        if ema_model == 'stable_model':
            print('setting evaluation model to stable model')
            curr_model = model.stable_model
        elif ema_model == 'plastic_model':
            print('setting evaluation model to plastic model')
            curr_model = model.plastic_model

    status = curr_model.training
    curr_model.eval()

    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = curr_model(inputs, k)
            else:
                outputs = curr_model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        print(f'Task {k} Accuracy: {correct / total * 100}')
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    curr_model.train(status)

    return accs, accs_mask_classes


def save_task_perf(savepath, results, n_tasks):

    results_array = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i >= j:
                results_array[i, j] = results[i][j]

    np.savetxt(savepath, results_array, fmt='%.2f')


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []

    model_stash = create_stash(model, args, dataset)

    lst_ema_models = ['plastic_model', 'stable_model']
    ema_loggers = {}
    ema_results = {}
    ema_results_mask_classes = {}
    ema_task_perf_paths = {}

    checkpoint_path = os.path.join(args.output_dir, "checkpoints", dataset.NAME, args.experiment_id)
    os.makedirs(checkpoint_path, exist_ok=True)

    for ema_model in lst_ema_models:
        if hasattr(model, ema_model):
            ema_results[ema_model], ema_results_mask_classes[ema_model] = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args.output_dir, args.experiment_id)
        task_perf_path = os.path.join(args.output_dir, "results", dataset.SETTING, dataset.NAME, model.NAME, args.experiment_id, 'task_performance.txt')

        for ema_model in lst_ema_models:
            if hasattr(model, ema_model):
                print('=' * 50)
                print(f'Creating Logger for {ema_model}')
                print('=' * 50)
                ema_loggers[ema_model] = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args.output_dir, args.experiment_id + f'_{ema_model}')
                ema_task_perf_paths[ema_model] = os.path.join(args.output_dir, "results", dataset.SETTING, dataset.NAME, model.NAME, args.experiment_id + f'_{ema_model}', 'task_performance.txt')

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
        model.writer = tb_logger.loggers[dataset.SETTING]

    if dataset.NAME != 'gcil-cifar100':
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()

    if model.NAME != 'icarl' and model.NAME != 'pnn' and dataset.NAME != 'gcil-cifar100':
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    model.num_tasks = dataset.N_TASKS
    model.num_classes = dataset.N_TASKS * dataset.N_CLASSES_PER_TASK

    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]

            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

            for ema_model in lst_ema_models:
                if hasattr(model, ema_model):
                    ema_accs = evaluate(model, dataset, eval_ema=True, ema_model=ema_model, last=True)
                    ema_results[ema_model][t - 1] = ema_results[ema_model][t - 1] + ema_accs[0]

                    if dataset.SETTING == 'class-il':
                        ema_results_mask_classes[ema_model][t - 1] = ema_results_mask_classes[ema_model][t - 1] + ema_accs[1]

        n_batches = len(train_loader)
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                model.iteration = (epoch * n_batches) + i
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
            if 'sail' in args.model:
                model.adjust_learning_rate(epoch)
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        print()
        print('*' * 30)
        print('Evaluating Model')
        print('*' * 30)
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

        # Evaluate on EMA model
        for ema_model in lst_ema_models:
            if hasattr(model, ema_model):
                print('=' * 30)
                print(f'Evaluating {ema_model}')
                print('=' * 30)
                ema_accs = evaluate(model, dataset, eval_ema=True, ema_model=ema_model)

                ema_results[ema_model].append(ema_accs[0])
                ema_results_mask_classes[ema_model].append(ema_accs[1])
                ema_mean_acc = np.mean(ema_accs, axis=1)
                print_mean_accuracy(ema_mean_acc, t + 1, dataset.SETTING)

                if args.csv_log:
                    ema_loggers[ema_model].log(ema_mean_acc)

                if args.tensorboard:
                    tb_logger.log_accuracy(np.array(ema_accs), ema_mean_acc, args, t, identifier=f'{ema_model}_')

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)

        for ema_model in lst_ema_models:
            if ema_model in ema_loggers:
                ema_loggers[ema_model].add_bwt(ema_results[ema_model], ema_results_mask_classes[ema_model])
                ema_loggers[ema_model].add_forgetting(ema_results[ema_model], ema_results_mask_classes[ema_model])

        if model.NAME != 'icarl' and model.NAME != 'pnn' and dataset.NAME != 'gcil-cifar100':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
        save_task_perf(task_perf_path, results, dataset.N_TASKS)
        for ema_model in lst_ema_models:
            if ema_model in ema_loggers:
                ema_loggers[ema_model].write(vars(args))
                save_task_perf(ema_task_perf_paths[ema_model], ema_results[ema_model], dataset.N_TASKS)

    # Save Models
    lst_models = ['plastic_model', 'stable_model', 'net']
    for ema_model in lst_models:
        if hasattr(model, ema_model):
            torch.save(getattr(model, ema_model), os.path.join(checkpoint_path, ema_model + '.pt'))