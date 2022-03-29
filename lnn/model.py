from torch import nn
from lnn.neuron import Or, DynamicOr
import torch


class LNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(DynamicOr(num_inputs=2, alpha=0.75))

    def forward(self, x):
        x = self.layers[0](x)
        return x

    @torch.no_grad()
    def project_params(self):
        for layer in self.layers:
            pass
            # layer.project_params()
