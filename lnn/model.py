from torch import nn
from lnn.neuron import Or
import torch


class LNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(Or(num_inputs=2))

    def forward(self, x):
        x = self.layers[0](x)
        return x

    @torch.no_grad()
    def project_params(self):
        for layer in self.layers:
            layer.project_params()
