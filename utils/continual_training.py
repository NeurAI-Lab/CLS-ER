# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_gcl_dataset
from models import get_model
from utils.status import progress_bar
from utils.tb_logger import *
from utils.status import create_fake_stash
from models.utils.continual_model import ContinualModel
from argparse import Namespace


def evaluate(model: ContinualModel, dataset, eval_ema=False, ema_model=None) -> float:
    """
    Evaluates the final accuracy of the model.
    :param model: the model to be evaluated
    :param dataset: the GCL dataset at hand
    :param eval_ema: flag to indicate if an exponential weighted average model
    should be evaluated (For CLS-ER)
    :param ema_model: if eval ema is set to True, which ema model (plastic or stable)
    should be evaluated (For CLS-ER)
    :return: a float value that indicates the accuracy
    """
    curr_model = model.net
    if eval_ema:
        if ema_model == 'stable_ema_model':
            print('setting evaluation model to stable ema')
            curr_model = model.stable_ema_model
        elif ema_model == 'plastic_ema_model':
            print('setting evaluation model to plastic ema')
            curr_model = model.plastic_ema_model

    curr_model.eval()
    correct, total = 0, 0
    while not dataset.test_over:
        inputs, labels = dataset.get_test_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = curr_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.sum(predicted == labels).item()
        total += labels.shape[0]

    acc = correct / total * 100
    return acc


def train(args: Namespace):
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    if args.csv_log:
        from utils.loggers import CsvLogger

    dataset = get_gcl_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    model.net.to(model.device)

    model_stash = create_fake_stash(model, args)

    lst_ema_models = ['plastic_model', 'stable_model']
    ema_loggers = {}
    ema_results = {}
    ema_results_mask_classes = {}
    ema_task_perf_paths = {}

    for ema_model in lst_ema_models:
        if hasattr(model, ema_model):
            ema_results[ema_model], ema_results_mask_classes[ema_model] = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args.output_dir, args.experiment_id)
        for ema_model in lst_ema_models:
            if hasattr(model, ema_model):
                print('=' * 50)
                print(f'Creating Logger for {ema_model}')
                print('=' * 50)
                ema_loggers[ema_model] = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args.output_dir, args.experiment_id + f'_{ema_model}')
                ema_task_perf_paths[ema_model] = os.path.join(args.output_dir, "results", dataset.SETTING, dataset.NAME, model.NAME, args.experiment_id + f'_{ema_model}', 'task_performance.txt')

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model.writer = tb_logger.loggers[dataset.SETTING]

    model.net.train()
    epoch, i = 0, 0
    model.iteration = 0

    while not dataset.train_over:
        model.iteration += 1
        inputs, labels, not_aug_inputs = dataset.get_train_data()
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        not_aug_inputs = not_aug_inputs.to(model.device)
        loss = model.observe(inputs, labels, not_aug_inputs)
        progress_bar(i, dataset.LENGTH // args.batch_size, epoch, 'C', loss)
        if args.tensorboard:
            tb_logger.log_loss_gcl(loss, i)
        i += 1

    acc = evaluate(model, dataset)
    print('Accuracy:', acc)

    for ema_model in lst_ema_models:
        if hasattr(model, ema_model):
            print('=' * 30)
            print(f'Evaluating {ema_model}')
            print('=' * 30)
            dataset = get_gcl_dataset(args)
            ema_accs = evaluate(model, dataset, eval_ema=True, ema_model=ema_model)
            print('Accuracy:', ema_accs)
            if args.csv_log:
                ema_loggers[ema_model].log(ema_accs)

    if args.csv_log:
        csv_logger.log(acc)
        csv_logger.write(vars(args))
    for ema_model in ema_loggers:
        ema_loggers[ema_model].write(vars(args))
