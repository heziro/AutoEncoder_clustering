import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.nn as nn

import losses
import models
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.optim import lr_scheduler
import training

import copy
import utils


def load_data(batch_size=32, num_workers=0, small_trainset=False, n_samples=-1, dataset_name='mnist',
              supervised_samples_percent=0., val_split=0.2, transform=None):
    if transform is None:
        transform=transforms.ToTensor()
    if dataset_name == 'mnist':
        train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())


    elif dataset_name == 'cifar10':
        train_set = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())

    if small_trainset:
        indices = np.where(train_set.targets == 0)[0][:int(n_samples / 2)]
        indices = np.concatenate((indices, torch.where(train_set.targets == 1)[0][:int(n_samples / 2)]))
        # Warp into Subsets and DataLoaders
        train_set = torch.utils.data.Subset(train_set, indices)
        supervised_set = list(train_set)[:int(len(train_set) * supervised_samples_percent)]
    else:
        if n_samples != -1:
            train_set = list(train_set)[:n_samples]
        supervised_set = list(train_set)[:int(len(train_set) * supervised_samples_percent)]

    train_set, val_set = torch.utils.data.random_split(train_set, [int(len(train_set) * (1 - val_split)),
                                                                   int(len(train_set) * val_split)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    # val_loader = None
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    supervised_loader = torch.utils.data.DataLoader(supervised_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, supervised_loader
