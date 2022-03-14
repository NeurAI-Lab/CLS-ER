import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
from torch.nn import functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)

    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--plastic_model_update_freq', type=float, default=0.90)
    parser.add_argument('--plastic_model_alpha', type=float, default=0.999)

    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class CLSER(ContinualModel):
    NAME = 'clser'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CLSER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # Initialize plastic and stable model
        self.plastic_model = deepcopy(self.net).to(self.device)
        self.stable_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = args.plastic_model_update_freq
        self.plastic_model_alpha = args.plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        loss = 0

        if not self.buffer.is_empty():

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            stable_model_logits = self.stable_model(buf_inputs)
            plastic_model_logits = self.plastic_model(buf_inputs)

            stable_model_prob = F.softmax(stable_model_logits, 1)
            plastic_model_prob = F.softmax(plastic_model_logits, 1)

            label_mask = F.one_hot(buf_labels, num_classes=stable_model_logits.shape[-1]) > 0
            sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
            sel_idx = sel_idx.unsqueeze(1)

            ema_logits = torch.where(
                sel_idx,
                stable_model_logits,
                plastic_model_logits,
            )

            l_cons = torch.mean(self.consistency_loss(self.net(buf_inputs), ema_logits.detach()))
            l_reg = self.args.reg_weight * l_cons
            loss += l_reg

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_cons', l_cons.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

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

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
        )

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.plastic_model_update_freq:
            self.update_plastic_model_variables()

        if torch.rand(1) < self.stable_model_update_freq:
            self.update_stable_model_variables()

        return loss.item()

    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
