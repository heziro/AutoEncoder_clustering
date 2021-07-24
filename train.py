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
import dataset

import copy
import utils






if __name__ == "__main__":

    # params
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_name = 'mnist'
    if dataset_name == 'cifar10':
        input_shape = (32, 32, 3)
        dim = 1000
    elif dataset_name == 'mnist':
        input_shape = (28, 28, 1)
        dim = 10
    n_clusters = 10

    lr = 0.001
    weight_decay = 0.0
    sched_gamma = 0.1
    sched_step = 200

    model = models.AutoEncoder(input_shape=input_shape, dim=dim)

    rec_criterion = nn.MSELoss(size_average=True)
    cluster_criterion = nn.KLDivLoss(size_average=False)
    contrastive_criterion = losses.ConLoss(device=device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)

    optimizer_pre = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler_pre = lr_scheduler.StepLR(optimizer_pre, step_size=sched_step, gamma=sched_gamma)

    optimizer_sup = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler_sup = lr_scheduler.StepLR(optimizer_sup, step_size=sched_step, gamma=sched_gamma)

    batch_size = 256
    epochs = 50
    pretrained_epochs = 300
    sup_epochs = 30
    gamma = 0.1
    con_gamma = 1
    small_trainset = False
    n_samples = -1
    pretrain = False
    path = "C:/Users/heziro/projects/AutoEncoder_clustering/artifact/model_299.pth"
    vis = False
    sup_train = True

    print(f'device is: {device}')

    train_loader, val_loader, test_loader, supervised_loader = dataset.load_data(batch_size=batch_size,
                                                                         small_trainset=small_trainset,
                                                                         n_samples=n_samples,
                                                                         dataset_name=dataset_name,
                                                                         supervised_samples_percent=0.01)

    model.to(device)
    if pretrain:
        model = training.pretrained(model, train_loader, test_loader, rec_criterion, optimizer_pre, scheduler_pre, batch_size,
                           pretrained_epochs, vis=vis, device=device, path=path)
    else:
        model.load_state_dict(torch.load(path, map_location=device))

    # init mu with k-means
    print("init centroids")
    model = training.init_mu(model, train_loader, 'cpu')

    model.to(device)
    if sup_train:
        # train with labels (contrastive loss)
        model = training.train_supervise(model, supervised_loader, contrastive_criterion, con_gamma, optimizer_sup, scheduler_sup, device, sup_epochs)

    training.train(model=model, train_loader=train_loader, val_loader=val_loader, rec_criterion=rec_criterion,
                   cluster_criterion=cluster_criterion, optimizer=optimizer, scheduler=scheduler,
                   batch_size=batch_size, epochs=epochs, vis=vis, device=device, pretrain=pretrain, path=path,
                   gamma=gamma)


