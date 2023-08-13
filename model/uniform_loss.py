import torch
from torch import nn


class UniformLoss(nn.Module):

    def __init__(self, temperature=2):
        super(UniformLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-self.temperature).exp().mean().log()
