# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay with Knowledge Distillation.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # Dataset Division
    parser.add_argument('--rand_division', action='store_true', default=False)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--reg_weight', type=float, default=0.1)
    return parser


class KdEr(ContinualModel):
    NAME = 'kder'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(KdEr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.prev_model = deepcopy(self.net)
        self.prev_model = self.prev_model.to(self.device)
        self.consistency_loss = nn.MSELoss(reduction='mean')
        self.current_task = 0

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        loss = 0

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            # Add a consistency term
            l_reg = self.consistency_loss(self.prev_model(buf_inputs).detach(), self.net(buf_inputs))
            loss = self.args.reg_weight * l_reg

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

            # Log values
            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        self.prev_model.load_state_dict(self.net.state_dict())

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset) -> None:
        self.current_task += 1
