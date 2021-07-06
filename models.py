import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf - paper
# https://github.com/XifengGuo/DCEC/blob/master/DCEC.py - TF code

class clusteringLayer(nn.Module):
    def __init__(self, n_clusters, dim, alpha=1.0) -> None:
        super(clusteringLayer, self).__init__()

        self.dim = dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu = nn.Parameter(torch.Tensor(self.n_clusters, self.dim))
        self.mu = nn.init.xavier_uniform_(self.mu)
        pass

    def forward(self, x):
        q1 = 1.0 / (1.0 + torch.norm(x.unsqueeze(1) - self.mu, p=2, dim=-1)**2)
        q2 = 1.0 / torch.sum((1.0 + torch.norm(x.unsqueeze(1) - self.mu, p=2, dim=-1)**2), dim=-1)
        q = q1 / q2.unsqueeze(1)

        # q1 = ((1.0 + torch.abs(x.unsqueeze(1) - torch.t(self.mu)) ** 2) /
        #       self.alpha) ** ((-1 * self.alpha + 1.0) / 2.0)
        # q2 = torch.sum(((1.0 + torch.abs(x.unsqueeze(1) - torch.t(self.mu)) ** 2) / self.alpha)
        #                ** ((-1 * self.alpha + 1.0) / 2.0), dim=1)
        # q = torch.sum(q1, dim=2) / torch.sum(q2, dim=1).view(-1, 1)
        return q

    def set_mu(self, tensor):
        self.mu = nn.Parameter(tensor)


class ClusterAutoEncoder(nn.Module):
    def __init__(self, input_shape, n_clusters, dim) -> None:
        super(ClusterAutoEncoder, self).__init__()

        self.input_shape = input_shape
        self.dim = dim
        self.n_clusters = n_clusters
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        self.embedding = nn.Linear(8*4, self.dim)
        self.deembedding = nn.Linear(self.dim, 8*4)

        self.cluster_layer = clusteringLayer(
            self.n_clusters, dim=self.dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def calc_p(self, q):
        p1 = q*q / (torch.sum(q, dim=0))
        p2 = torch.sum((q*q / (torch.sum(q, dim=0))), dim=1)
        return p1 / p2.unsqueeze(1)


    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        latent = self.embedding(x)
        q = self.cluster_layer(latent)
        x = self.deembedding(latent)
        x = x.view(-1, 8, 2, 2)
        x = self.decoder(x)
        # clac p
        p = self.calc_p(q)
        return x, q, latent, p
