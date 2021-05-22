import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import models


def load_data():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)


def train():

    model = models.ClusterAutoEncoder(input_shape=(32, 32, 1), n_clusters=10)
    model(torch.rand(3, 1, 32, 32))


if __name__ == "__main__":
    train()
