import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import models
import utils as u
import torch.optim as optim
from sklearn.cluster import KMeans


def load_data(batch_size=16, num_workers=2):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(
        './data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def pretrained(model, train_loader, test_loader, rec_criterion, optimizer, batch_size, epochs):

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            x, q, latent, p = model(inputs)
            # show reconsruction of x
            # plt.imshow(x[0, 0].data.numpy());
            # plt.show()
            #
            loss = rec_criterion(x, inputs)
            # total_loss = (1.0 - gamma) * rec_loss + gamma * clustering_loss * 0.01
            loss.backward()
            optimizer.step()

    return model


def init_mu(model, train_loader, device):
    features = None
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        model.eval()
        x, q, latent, p = model(inputs)
        if features is None:
            features = latent.cpu().detach().numpy()
        else:
            features = np.concatenate((features, latent.cpu().detach().numpy()), 0)
    km = KMeans(n_clusters=model.n_clusters, n_init=20)
    km.fit_predict(features)
    mu = torch.from_numpy(km.cluster_centers_)
    model.cluster_layer.set_mu(mu.to(device))

    model.train()
    return model




def train(model, train_loader, test_loader, rec_criterion, cluster_criterion, optimizer, batch_size, epochs, gamma, pretrained_epochs):

    model = pretrained(model, train_loader, test_loader, rec_criterion, optimizer, batch_size, pretrained_epochs)
    # init mu with kmeans
    model = init_mu(model, train_loader, 'cpu')
    total_loss = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            x, q, latent, p = model(inputs)
            # show reconsruction of x
            # plt.imshow(x[0, 0].data.numpy());
            # plt.show()
            #
            rec_loss = rec_criterion(x, inputs)
            clustering_loss = kl_loss(p, q)
            total_loss = (1.0-gamma)*rec_loss + gamma*clustering_loss *0.01
            total_loss.backward()
            optimizer.step()


        print('epoch: [%d] rec_loss: %.3f, clustering_loss: %.3f, total loss: %.3f' %
              (epoch + 1, rec_loss, clustering_loss, total_loss))

    test_iter = iter(test_loader)
    images, labels = test_iter.next()
    outputs = model(images)
    pass

def kl_loss(p, q):
    return -1.0 * torch.sum(torch.sum(p*torch.log(p/q), dim=1), dim=0)

if __name__ == "__main__":
    model = models.ClusterAutoEncoder(input_shape=(32, 32, 1), n_clusters=10, dim=11)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    rec_criterion = nn.MSELoss()
    cluster_criterion = nn.MSELoss()
    batch_size = 32
    epochs = 10
    pretrained_epochs = 1
    gamma = 0.5
    train_loader, test_loader = load_data(batch_size=batch_size)

    # get some random training images
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    # show images
    # u.imshow(torchvision.utils.make_grid(images))

    train(model, train_loader, test_loader, rec_criterion, cluster_criterion, optimizer, batch_size, epochs, gamma,
          pretrained_epochs=pretrained_epochs)
