import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

criterion_MSE = nn.MSELoss(reduction='mean')


def cross_entropy(y, labels):
    l_ce = F.cross_entropy(y, labels)
    return l_ce


def distillation(student_scores, teacher_scores, T):

    p = F.log_softmax(student_scores / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / student_scores.shape[0]

    return l_kl


def fitnet_loss(A_t, A_s, rand=False, noise=0.1):
    """Given the activations for a batch of input from the teacher and student
    network, calculate the fitnet loss from the paper
    FitNets: Hints for Thin Deep Nets https://arxiv.org/abs/1412.6550

    Note: This function assumes that the number of channels and the spatial dimensions of
    the teacher and student activation maps are the same.

    Parameters:
        A_t (4D tensor): activation maps from the teacher network of shape b x c x h x w
        A_s (4D tensor): activation maps from the student network of shape b x c x h x w

    Returns:
        l_fitnet (1D tensor): fitnet loss value
"""
    if rand:
        rand_noise =  torch.FloatTensor(A_t.shape).uniform_(1 - noise, 1 + noise)
        A_t = A_t * rand_noise

    return criterion_MSE(A_t, A_s)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y, rand=False, noise=0.1):
    if rand:
        rand_noise = torch.FloatTensor(y.shape).uniform_(1 - noise, 1 + noise).cuda()
        y = y * rand_noise

    return (at(x) - at(y)).pow(2).mean()


def FSP_loss(fea_t, short_t, fea_s, short_s, rand=False, noise=0.1):

    a, b, c, d = fea_t.size()
    feat = fea_t.view(a, b, c * d)
    a, b, c, d = short_t.size()
    shortt = short_t.view(a, b, c * d)
    G_t = torch.bmm(feat, shortt.permute(0, 2, 1)).div(c * d).detach()

    a, b, c, d = fea_s.size()
    feas = fea_s.view(a, b, c * d)
    a, b, c, d = short_s.size()
    shorts = short_s.view(a, b, c * d)
    G_s = torch.bmm(feas, shorts.permute(0, 2, 1)).div(c * d)

    return criterion_MSE(G_s, G_t)


def similarity_preserving_loss(A_t, A_s):
    """Given the activations for a batch of input from the teacher and student
    network, calculate the similarity preserving knowledge distillation loss from the
    paper Similarity-Preserving Knowledge Distillation (https://arxiv.org/abs/1907.09682)
    equation 4

    Note: A_t and A_s must have the same batch size

    Parameters:
        A_t (4D tensor): activation maps from the teacher network of shape b x c1 x h1 x w1
        A_s (4D tensor): activation maps from the student network of shape b x c2 x h2 x w2

    Returns:
        l_sp (1D tensor): similarity preserving loss value
"""

    # reshape the activations
    b1, c1, h1, w1 = A_t.shape
    b2, c2, h2, w2 = A_s.shape
    assert b1 == b2, 'Dim0 (batch size) of the activation maps must be compatible'

    Q_t = A_t.reshape([b1, c1 * h1 * w1])
    Q_s = A_s.reshape([b2, c2 * h2 * w2])

    # evaluate normalized similarity matrices (eq 3)
    G_t = torch.mm(Q_t, Q_t.t())
    # G_t = G_t / G_t.norm(p=2)
    G_t = torch.nn.functional.normalize(G_t)

    G_s = torch.mm(Q_s, Q_s.t())
    # G_s = G_s / G_s.norm(p=2)
    G_s = torch.nn.functional.normalize(G_s)

    # calculate the similarity preserving loss (eq 4)
    l_sp = (G_t - G_s).pow(2).mean()

    return l_sp


class SlicedWassersteinDiscrepancy(nn.Module):
    """PyTorch adoption of https://github.com/apple/ml-cvpr2019-swd"""
    def __init__(self, mean=0, sd=1, device='cpu'):
        super(SlicedWassersteinDiscrepancy, self).__init__()
        self.dist = torch.distributions.Normal(mean, sd)
        self.device = device

    def forward(self, p1, p2):
        if p1.shape[1] > 1:
            # For data more than one-dimensional input, perform multiple random
            # projection to 1-D
            proj = self.dist.sample([p1.shape[1], 128]).to(self.device)
            proj *= torch.rsqrt(torch.sum(proj.pow(2), dim=0, keepdim=True))

            p1 = torch.mm(p1, proj)
            p2 = torch.mm(p2, proj)

        p1, _ = torch.sort(p1, 0, descending=True)
        p2, _ = torch.sort(p2, 0, descending=True)

        wdist = (p1 - p2).pow(2).mean()

        return wdist


class RKD(object):
    """
    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.
    relational knowledge distillation.
    arXiv preprint arXiv:1904.05068, 2019.
    """
    def __init__(self, device, eval_dist_loss=True, eval_angle_loss=False):
        super(RKD, self).__init__()
        self.device = device
        self.eval_dist_loss = eval_dist_loss
        self.eval_angle_loss = eval_angle_loss
        self.huber_loss = torch.nn.SmoothL1Loss()

    @staticmethod
    def distance_wise_potential(x):
        x_square = x.pow(2).sum(dim=-1)
        prod = torch.matmul(x, x.t())
        distance = torch.sqrt(
            torch.clamp( torch.unsqueeze(x_square, 1) + torch.unsqueeze(x_square, 0) - 2 * prod,
            min=1e-12))
        mu = torch.sum(distance) / torch.sum(
            torch.where(distance > 0., torch.ones_like(distance),
                        torch.zeros_like(distance)))

        return distance / (mu + 1e-8)

    @staticmethod
    def angle_wise_potential(x):
        e = torch.unsqueeze(x, 0) - torch.unsqueeze(x, 1)
        e_norm = torch.nn.functional.normalize(e, dim=2)
        return torch.matmul(e_norm, torch.transpose(e_norm, -1, -2))

    def eval_loss(self, source, target):

        # Flatten tensors
        source = source.reshape(source.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        # normalize
        source = torch.nn.functional.normalize(source, dim=1)
        target = torch.nn.functional.normalize(target, dim=1)

        distance_loss = torch.tensor([0.]).to(self.device)
        angle_loss = torch.tensor([0.]).to(self.device)

        if self.eval_dist_loss:
            distance_loss = self.huber_loss(
                self.distance_wise_potential(source), self.distance_wise_potential(target)
            )

        if self.eval_angle_loss:
            angle_loss = self.huber_loss(
                self.angle_wise_potential(source), self.angle_wise_potential(target)
            )

        return distance_loss, angle_loss
