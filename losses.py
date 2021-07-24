import torch
import torch.nn as nn

#  reference: https://github.com/HobbitLong/SupContrast

class ConLoss(nn.Module):

    def __init__(self, device, temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, centroids, labels=None):
        batch_size = features.shape[0]
        d = torch.cdist(features, centroids)
        I = torch.eye(batch_size, device=self.device)
        e = torch.exp(-1.0*d)
        loss = (-1.0 * torch.log(torch.sum(((e*I) / ((torch.sum(e, 1)))), 0))).mean()
        return loss