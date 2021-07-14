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
        # q1 = 1.0 / (1.0 + torch.norm(x.unsqueeze(1) - self.mu, p=2, dim=-1)**2)
        # q2 = 1.0 / torch.sum((1.0 + torch.norm(x.unsqueeze(1) - self.mu, p=2, dim=-1)**2), dim=-1)
        # q = q1 / q2.unsqueeze(1)
        # return q

        x = x.unsqueeze(1) - self.mu
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x



    def set_mu(self, tensor):
        self.mu = nn.Parameter(tensor)


class ClusterAutoEncoder(nn.Module):
    def __init__(self, input_shape, n_clusters, dim) -> None:
        super(ClusterAutoEncoder, self).__init__()

        self.input_shape = input_shape
        self.dim = dim
        self.n_clusters = n_clusters
        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        self.embedding2 = nn.Linear(8*4, self.dim)
        self.deembedding2 = nn.Linear(self.dim, 64*64)

        self.embedding = nn.Linear(64, self.dim)
        self.deembedding = nn.Linear(self.dim, 64)

        self.cluster_layer = clusteringLayer(
            self.n_clusters, dim=self.dim)

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def calc_p(self, q):
        p1 = q**2 / (torch.sum(q, dim=0))
        p2 = torch.sum((q**2 / (torch.sum(q, dim=0))), dim=1)
        return p1 / p2.unsqueeze(1)


    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = torch.flatten(x, start_dim=1)
    #     latent = self.embedding(x)
    #     q = self.cluster_layer(latent)
    #     x = self.deembedding(latent)
    #     x = x.view(-1, 64, 1, 1)
    #     x = self.decoder(x)
    #     # clac p
    #     p = self.calc_p(q)
    #     return x, q, latent, p

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        latent = self.embedding(x)
        q = self.cluster_layer(latent)
        x = self.deembedding(latent)
        x = x.view(-1, 64, 1, 1)
        x = self.decoder(x)
        # clac p
        # p = self.calc_p(q)
        return x, q, latent


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x