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
import copy
import utils


def load_data(batch_size=32, num_workers=0, small_trainset=False, n_samples=-1, dataset_name='mnist'):
    if dataset_name == 'mnist':
        train_set = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    elif dataset_name == 'cifar10':
        train_set = datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())

    if small_trainset:
        indices = np.where(train_set.targets == 0)[0][:int(n_samples / 2)]
        indices = np.concatenate((indices, torch.where(train_set.targets == 1)[0][:int(n_samples / 2)]))
        # Warp into Subsets and DataLoaders
        train_set = torch.utils.data.Subset(train_set, indices)
    else:
        if n_samples != -1:
            train_set = list(train_set)[:n_samples]

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def pretrained(model, train_loader, test_loader, rec_criterion, optimizer_pre, scheduler_pre, batch_size, epochs,
               vis=False, device='cpu', save=False, path=None):
    history = []

    save = True

    model.train()
    model.to(device)
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_pre.zero_grad()
            x, q, latent = model(inputs)
            loss = rec_criterion(x, inputs)
            loss.backward()
            optimizer_pre.step()
        scheduler_pre.step()
        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        history.append((epoch, inputs, x), )

        if save:
            torch.save(model.state_dict(), path + "/model_cifar_" + str(epoch) + '.pth')
    if vis:
        model.eval()
        test_iter = iter(test_loader)
        images, labels = test_iter.next()
        outputs = model(images)

        plt.figure(figsize=(15, 15))
        n_examples = outputs[0].shape[0]
        for i in range(n_examples):
            img = images[i].reshape(28, 28, -1).detach().numpy()
            plt.subplot(n_examples, 2, 2 * i + 1)
            plt.title("Original img (pretrained)")
            plt.imshow(img)

            rec_img = outputs[0][i].reshape(28, 28, -1).detach().numpy()
            plt.subplot(n_examples, 2, 2 * i + 2)
            plt.title("Reconstruct img (pretrained)")
            plt.imshow(rec_img)
        plt.show()

    return model


def init_mu(model, train_loader, device):
    model.to(device)
    features = None
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        model.eval()
        # model.to(device)
        x, q, latent = model(inputs)
        if features is None:
            features = latent.cpu().detach().numpy()
        else:
            features = np.concatenate((features, latent.cpu().detach().numpy()), 0)
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    km.fit_predict(features)
    mu = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_mu(mu.to(device))

    model.train()
    return model


def train(model, train_loader, test_loader, rec_criterion, cluster_criterion, contrastive_criterion, optimizer, optimizer_pre, scheduler,
          scheduler_pre, batch_size, epochs, gamma, pretrained_epochs, vis=False, device='cpu', pretrain=True,
          path=None):
    model.to(device)

    if pretrain:
        model = pretrained(model, train_loader, test_loader, rec_criterion, optimizer_pre, scheduler_pre, batch_size,
                           pretrained_epochs, vis=vis, device=device, path=path)
    else:
        model.load_state_dict(torch.load("C:/Users/heziro/projects/AutoEncoder_clustering/artifact/model_299.pth",
                                         map_location=device))

    update_interval = 80

    con_gamma = 0.1
    tol = 1e-2
    finished = False
    # init mu with kmeans
    print("init centroids")
    model = init_mu(model, train_loader, 'cpu')
    total_loss = 0
    all_points = []
    clusters = np.array([])
    model.train()
    history = []
    model.to(device)

    print("calculate prediction")
    output_distribution, labels, preds_prev = utils.calculate_predictions(model, train_loader, device)
    target_distribution = target(output_distribution)

    for epoch in range(epochs):
        scheduler.step()
        model.train(True)
        batch_num = 1
        for i, data in enumerate(train_loader, 0):
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                output_distribution, labels, preds = utils.calculate_predictions(model, train_loader, device)
                target_distribution = target(output_distribution)
                nmi = utils.metrics.nmi(labels, preds)
                ari = utils.metrics.ari(labels, preds)
                acc = utils.metrics.acc(labels, preds)
                print('NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

                print(
                    'Epoch:{}, Total Loss:{:.4f}, Reconstruction Loss:{:.4f}, Clustering Loss:{:.4f}'.format(epoch + 1,
                                                                                                             float(
                                                                                                                 total_loss),
                                                                                                             float(
                                                                                                                 rec_loss),
                                                                                                             float(
                                                                                                                 clustering_loss * gamma)))

                # check stop criterion
                delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                preds_prev = np.copy(preds)
                if delta_label < tol:
                    print('Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    print('Reached tolerance threshold. Stopping training.')
                    finished = True
                    break

            tar_dist = target_distribution[((batch_num - 1) * batch_size):(batch_num * batch_size), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                x, q, latent = model(inputs)
                # all_points.append(latent.cpu().detach().numpy())
                # clusters = np.append(clusters, labels[0].cpu().detach().numpy())

                rec_loss = rec_criterion(x, inputs)
                con_loss = con_gamma * contrastive_criterion(latent, model.clustering.weight[labels])
                # clustering_loss = kl_loss(p, q)
                # clustering_loss = -1.0 * gamma * (cluster_criterion(q, tar_dist) / batch_size)
                clustering_loss = gamma * cluster_criterion(torch.log(q), tar_dist) / batch_size
                total_loss = rec_loss + clustering_loss + con_loss
                total_loss.backward()
                optimizer.step()
            batch_num = batch_num + 1

        if finished:
            break

        print("con loss: ", con_loss)
        print('Epoch:{}, Total Loss:{:.4f}, Reconstruction Loss:{:.4f}, Clustering Loss:{:.4f}'.format(epoch + 1,
                                                                                                       float(
                                                                                                           total_loss),
                                                                                                       float(rec_loss),
                                                                                                       float(
                                                                                                           clustering_loss * gamma)))
        history.append((epoch, inputs, x), )

        if vis and epoch % 50 == 0:
            plt.figure()
            all_points = np.array(all_points).reshape(-1, 2)
            scatter = plt.scatter(all_points[:, 0], all_points[:, 1], c=list(clusters.astype(np.int32)))
            plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1])
            plt.show()

        all_points = []
        clusters = np.array([])

    model.eval()
    test_iter = iter(test_loader)
    images, labels = test_iter.next()
    model.to(device)
    outputs = model(images)

    plt.figure(figsize=(15, 15))
    n_examples = outputs[0].shape[0]
    for i in range(n_examples):
        img = images[i].reshape(28, 28, -1).detach().numpy()
        plt.subplot(n_examples, 2, 2 * i + 1)
        plt.title("Original img")
        plt.imshow(img)

        rec_img = outputs[0][i].reshape(28, 28, -1).detach().numpy()
        plt.subplot(n_examples, 2, 2 * i + 2)
        plt.title("Reconstruct img")
        plt.imshow(rec_img)
    plt.show()

    pass


def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist


def kl_loss(p, q):
    # return -1.0*nn.KLDivLoss(reduction='sum')(p, q)
    return -1.0 * torch.sum(torch.sum(p * torch.log(p / q), dim=1), dim=0)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_name = 'mnist'
    if dataset_name == 'cifar10':
        input_shape = (32, 32, 3)
    elif dataset_name == 'mnist':
        input_shape = (28, 28, 1)
    n_clusters = 10
    dim = 10
    # model = models.ClusterAutoEncoder(input_shape=input_shape, n_clusters=n_clusters, dim=10)
    model = models.CAE_3(input_shape=(28, 28, 1), dim=dim)
    # model = CAE_3_for_cifar

    rec_criterion = nn.MSELoss(size_average=True)
    cluster_criterion = nn.KLDivLoss(size_average=False)
    contrastive_criterion = losses.ConLoss(device=device)
    lr = 0.001
    weight_decay = 0.0
    sched_gamma = 0.1
    sched_step = 200
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)

    optimizer_pre = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler_pre = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)

    batch_size = 256
    epochs = 50
    pretrained_epochs = 300
    gamma = 0.1
    small_trainset = False
    n_samples = -1
    pretrain = False
    path = "C:/Users/heziro/projects/AutoEncoder_clustering/artifact"
    vis = False

    print(f'device is: {device}')

    train_loader, test_loader = load_data(batch_size=batch_size, small_trainset=small_trainset, n_samples=n_samples,
                                          dataset_name=dataset_name)

    # get some random training images
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()

    # show images
    # u.imshow(torchvision.utils.make_grid(images))

    train(model, train_loader, test_loader, rec_criterion, cluster_criterion, contrastive_criterion, optimizer, optimizer_pre, scheduler,
          scheduler_pre, batch_size, epochs, gamma=gamma, pretrained_epochs=pretrained_epochs, vis=vis, device=device,
          pretrain=pretrain, path=path)
