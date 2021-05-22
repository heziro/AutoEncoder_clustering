import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf - paper
# https://github.com/XifengGuo/DCEC/blob/master/DCEC.py - TF code

class clusteringLayer(nn.Module):
    def __init__(self, n_clusters, in_features, alpha=1.0) -> None:
        super(clusteringLayer, self).__init__()

        self.in_features = in_features
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu = nn.Parameter(torch.Tensor(
            self.n_clusters, self.in_features))
        self.mu = nn.init.xavier_uniform_(self.mu)
        pass

    def forward(self, x):
        q1 = ((1.0 + torch.abs(x.unsqueeze(1) - torch.t(self.mu))**2) /
              self.alpha)**((-1*self.alpha + 1.0)/2.0)
        q2 = torch.sum(((1.0 + torch.abs(x.unsqueeze(1) - torch.t(self.mu))**2)/self.alpha)
                       ** ((-1*self.alpha + 1.0)/2.0), dim=1)
        q = torch.sum(q1, dim=2) / torch.sum(q2, dim=1).view(-1, 1)
        return q


class ClusterAutoEncoder(nn.Module):
    def __init__(self, input_shape, n_clusters) -> None:
        super(ClusterAutoEncoder, self).__init__()

        self.input_shape = input_shape
        self.n_clusters = n_clusters
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        # self.in_features = self.input_shape[0] // 3 // 2 * \
        #     self.input_shape[1] // 3 // 2
        self.in_features = 2*8*2
        self.embedding = nn.Linear(self.in_features, self.n_clusters)
        self.deembedding = nn.Linear(self.n_clusters, self.in_features)

        self.cluster_layer = clusteringLayer(
            self.n_clusters, in_features=self.n_clusters+1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.embedding(x)
        q = self.cluster_layer(x)
        x = self.deembedding(x)
        x = x.view(-1, 8, 2, 2)
        x = self.decoder(x)
        return x, q
