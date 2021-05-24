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


def train():
    batch_size = 32
    trainloader, testloader = load_data(batch_size=batch_size)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    u.imshow(torchvision.utils.make_grid(images))

    model = models.ClusterAutoEncoder(input_shape=(32, 32, 1), n_clusters=10)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_loss = 0
    for epoch in range(10):

        # running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs[0], inputs)
            # loss = criterion(outputs[0], inputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.data

            # running_loss += loss.item()

        print('epoch: [%d] loss: %.3f total loss: %.3f' %
              (epoch + 1, loss, total_loss))

    testiter = iter(testloader)
    images, labels = testiter.next()
    outputs = model(images)
    return model


if __name__ == "__main__":
    model = train()
