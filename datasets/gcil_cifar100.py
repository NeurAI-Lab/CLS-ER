from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from collections import Counter, OrderedDict
from copy import deepcopy
import math
from numpy.random import choice
from argparse import Namespace
import numpy as np


# =============================================================================
# Sampler
# =============================================================================
class sampler(object):
    def __init__(self, train_y, test_y, epoch_size, weight_dist, pretrain_class_nb, pretrain, args=None):
        self.class_numb = 0
        self.epoch_size = epoch_size  # total count of samples in this epoch
        self.pretrain = pretrain
        self.pretrain_class_nb = pretrain_class_nb  # the number of classes for pre-training
        self.new_class_num = max(train_y) - pretrain_class_nb + 1
        self.counter = Counter(np.array(train_y))
        self.chosen_class_sizes = None
        self.end_training = False  # end the incremental training if data of any class run out
        self.index_class_map_train = {class_: np.where(train_y == class_)[0] for class_ in self.counter.keys()}  # key: class, value: all indices in the dataset
        self.index_class_map_train_fixed = deepcopy(self.index_class_map_train)  # for later reference
        self.index_class_map_test = {class_: np.where(test_y == class_)[0] for class_ in self.counter.keys()}
        self.experienced_classes = OrderedDict()
        self.current_batch_class_indices = None
        self.experienced_counts = []
        self.class_not_in_this_batch = []  # used for distillation loss
        self.args = args
        if 'unif' in weight_dist or 'noise' in weight_dist:
            self.class_weight_dist = {class_: 1 for class_ in self.counter.keys()}  # to be modified
        elif weight_dist == 'longtail':  # long tail weight dist
            # this is similar to the long-tailed cifar. with eponential decay, with 0.97, the imbalance factor around 20 (class 0 appears 20 times more often than class 99)
            self.class_weight_dist = {class_: math.pow(0.984, class_) for class_ in self.counter.keys()}

    def sample_class_sizes(self, chosen_class_sizes):
        if chosen_class_sizes:
            self.chosen_class_sizes = chosen_class_sizes
        # pretrain
        elif self.pretrain and len(self.experienced_classes.keys()) < self.pretrain_class_nb:
            sampled_classes = list(choice(list(self.counter.keys()), size=self.pretrain_class_nb, replace=False))
            self.chosen_class_sizes = {sampled_class: self.counter[sampled_class] for sampled_class in sampled_classes}

        else:
            non_empty_classes = np.array([class_ for class_ in self.counter.keys() if self.counter[class_] != 0])

            self.class_numb = choice(min(len(non_empty_classes), self.args.phase_class_upper), size=1)[0] + 1  # this number should be greater than 0

            # we sample remaining class uniformly
            sampled_classes = list(choice(non_empty_classes, size=self.class_numb, replace=False))

            if 'noise' in self.args.weight_dist:
                weight_for_sampled_classes = [self.class_weight_dist[sampled_class] + max(np.random.normal(0, 0.2), -0.99) for sampled_class in sampled_classes]
            else:
                weight_for_sampled_classes = [self.class_weight_dist[sampled_class] for sampled_class in sampled_classes]
            normalized_weight_for_sampled_classes = [weight / sum(weight_for_sampled_classes) for weight in weight_for_sampled_classes]

            samples = []
            while not len(set(samples)) == self.class_numb:
                samples = list(choice(sampled_classes, size=self.epoch_size, replace=True, p=normalized_weight_for_sampled_classes))

            # count of data in chosen classes, however, we can not have it more than # of samples left for this class in counter
            # total_sampled_classes_weight = sum(list(map(self.class_weight_dist.get, sampled_classes)))
            self.chosen_class_sizes = {sampled_class: min(samples.count(sampled_class), self.counter[sampled_class]) for sampled_class in sampled_classes}

        # update records
        self.counter.subtract(Counter(self.chosen_class_sizes))

        # non_empty_class_numb = len(np.array([class_ for class_ in self.counter.keys() if self.counter[class_] != 0]))
        # if non_empty_class_numb < self.new_class_num:
        #     self.end_training = True

        return self.chosen_class_sizes

    # output a list of sample indices
    def sample_train_data_indices(self, current_batch_class_indices=None):
        if current_batch_class_indices:
            self.current_batch_class_indices = current_batch_class_indices
            chosen_class_sizes = {_class: len(_class_indices) for _class, _class_indices in current_batch_class_indices.items()}
            self.class_numb = len(chosen_class_sizes)
        else:
            # sample and remove indices
            self.current_batch_class_indices = {}
            chosen_class_sizes = None

        # get the class sizes
        _ = self.sample_class_sizes(chosen_class_sizes=chosen_class_sizes)

        for class_, size_ in self.chosen_class_sizes.items():
            if current_batch_class_indices:
                class_indices = self.current_batch_class_indices[class_]
            else:
                class_indices = list(choice(self.index_class_map_train[class_], size_, replace=False))
                # store data indices for this class
                self.current_batch_class_indices[class_] = class_indices
            # remove sampled indices
            self.index_class_map_train[class_] = list(set(self.index_class_map_train[class_]) - set(class_indices))
            # update record
            if class_ in self.experienced_classes:
                self.experienced_classes[class_] += class_indices
                self.experienced_counts[list(self.experienced_classes.keys()).index(class_)] += size_
            else:
                self.experienced_classes[class_] = class_indices
                self.experienced_counts.append(size_)

        self.class_not_in_this_batch = list(set(range(len(self.experienced_classes))) - set([self.map_labels(i) for i in self.chosen_class_sizes.keys()]))

        return np.concatenate([indices for indices in self.current_batch_class_indices.values()]).astype(int), len(self.chosen_class_sizes)

    # output a list of current epoch class test indices and a list of cumulative class test indices
    def sample_test_data_indices(self):
        # the indices of the current epoch classes
        current_test_ind = np.concatenate([self.index_class_map_test[class_] for class_ in self.chosen_class_sizes.keys()])
        # the indices of all past classes
        cumul_test_ind = np.concatenate([self.index_class_map_test[class_] for class_ in self.experienced_classes])

        return current_test_ind, cumul_test_ind

    # map original labels to the order labels
    def map_labels(self, original_label):
        return list(self.experienced_classes.keys()).index(original_label)

    # convert the sample index in the whole dataset to its index in the class
    def map_index_in_class(self, class_, indices_in_dataset):
        return [np.where(self.index_class_map_train_fixed[class_] == index_in_dataset)[0][0] for index_in_dataset in indices_in_dataset]  # index in its class


class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class GCILCIFAR100(ContinualDataset):

    NAME = 'gcil-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    N_CLASSES = 100

    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
             ]
    )

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """

        super(GCILCIFAR100, self).__init__(args)


        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        # Get training and test datatset
        trainset = MyCIFAR100(base_path() + 'CIFAR100', train=True, download=True, transform=self.TRANSFORM)

        testset = MyCIFAR100(base_path() + 'CIFAR100', train=False, download=True, transform=test_transform)

        self.X_train_total = np.array(trainset.data)
        self.Y_train_total = np.array(trainset.targets)
        self.X_valid_total = np.array(testset.data)
        self.Y_valid_total = np.array(testset.targets)

        self.current_batch_class_indices = None
        self.current_training_indices = None

        self.ind_sampler = sampler(
            self.Y_train_total,
            self.Y_valid_total,
            epoch_size=self.args.epoch_size,
            pretrain=self.args.pretrain,
            pretrain_class_nb=self.args.pretrain_class_nb,
            weight_dist=self.args.weight_dist,
            args=self.args,
        )

        np.random.seed(self.args.gil_seed)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True, download=True, transform=transform)
        test_dataset = CIFAR100(base_path() + 'CIFAR100', train=False, download=True, transform=test_transform)


        indice_train, num_classes = self.ind_sampler.sample_train_data_indices(current_batch_class_indices=self.current_batch_class_indices)
        indice_test, indice_test_cumul = self.ind_sampler.sample_test_data_indices()

        self.current_training_indices = indice_train

        # access data for this phase
        X_train = self.X_train_total[indice_train]
        Y_train = self.Y_train_total[indice_train]

        X_test_cumul = self.X_valid_total[indice_test_cumul]
        Y_test_cumul = self.Y_valid_total[indice_test_cumul]

        print('=' * 30)
        print('samples for current Task')
        (unique, counts) = np.unique(Y_train, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies)

        # label mapping
        map_Y_train = np.array([self.ind_sampler.map_labels(i) for i in Y_train])  # train labels: the order of classes
        map_Y_test_cumul = np.array([self.ind_sampler.map_labels(i) for i in Y_test_cumul])

        print('X_train size: ', len(Y_train))
        print('number of classes: ', self.ind_sampler.class_numb)

        train_dataset.data = X_train.astype('uint8')
        train_dataset.targets = map_Y_train
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, drop_last=True)

        test_dataset.data = X_test_cumul.astype('uint8')
        test_dataset.targets = map_Y_test_cumul
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        self.test_loaders.append(test_loader)
        self.train_loader = train_loader

        return train_loader, test_loader

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True, download=True, transform=transform)

        X_train = self.X_train_total[self.current_training_indices]
        Y_train = self.Y_train_total[self.current_training_indices]
        map_Y_train = np.array([self.ind_sampler.map_labels(i) for i in Y_train])  # train labels: the order of classes

        train_dataset.data = X_train.astype('uint8')

        train_dataset.targets = map_Y_train
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, dropout=True)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), GCILCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(GCILCIFAR100.N_CLASSES)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform


