from torch import nn
from lnn.utils import val_clamp
import torch


class LogicNeuron(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.num_inputs = num_inputs
        self.weights = nn.Parameter(torch.Tensor(self.num_inputs))
        self.bias = nn.Parameter(torch.Tensor(1))

        torch.nn.init.constant_(self.weights, 1.0)
        torch.nn.init.constant_(self.bias, 1.0)

    @torch.no_grad()
    def project_params(self):
        self.weights.data = self.weights.data.clamp(0, 1)
        self.bias.data = self.bias.data.clamp(0, self.num_inputs)


class And(LogicNeuron):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)

    def forward(self, x):
        """
        x: torch.Tensor([batch_size, 2, num_inputs])

        return: torch.Tensor([batch_size, 2])
        """

        x = val_clamp(self.bias - ((1 - x) @ self.weights))
        return x.view(-1, 2, 1)


class Or(LogicNeuron):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)

    def forward(self, x):
        """
        x: torch.Tensor([batch_size, 2, num_inputs])

        return: torch.Tensor([batch_size, 2])
        """

        x = val_clamp(1 - self.bias + (x @ self.weights))
        return x.view(-1, 2, 1)
