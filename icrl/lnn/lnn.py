from torch import nn
from neuron import Or
import torch


class LNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(Or(3))

    def forward(self, x):
        x = self.layers[0](x)
        return x.view(-1, 2, 1)

    @torch.no_grad()
    def project_params(self):
        for layer in self.layers:
            layer.project_params()
