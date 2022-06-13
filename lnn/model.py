from torch import nn
from lnn.neuron import Or, DynamicOr, DynamicAnd
import numpy as np
import torch


class LNN(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        #self.and1 = DynamicAnd(num_inputs=4, alpha=0.9)
        #self.and2 = DynamicAnd(num_inputs=4, alpha=0.9)
        self.and3 = DynamicAnd(num_inputs=4, alpha=0.85)
        #self.and4 = DynamicAnd(num_inputs=4, alpha=0.9)
        self.or1 = DynamicOr(num_inputs=2, alpha=0.75, final=False)
        #self.or2 = DynamicOr(num_inputs=2, alpha=0.75)
        #self.or3 = DynamicOr(num_inputs=2, alpha=0.75)
        #self.or4 = DynamicOr(num_inputs=2, alpha=0.75)

    def forward(self, x):
        # y1 = self.and1(
        #    torch.cat((x[:, 0][:, None], x[:, 1][:, None], x[:, 5][:, None], x[:, 7][:, None]), dim=-1))[:, None]
        # y2 = self.and2(
        #    torch.cat((x[:, 1][:, None], x[:, 2][:, None], x[:, 6][:, None], x[:, 8][:, None]),dim=-1))[:, None]
        y3 = self.and3(
            torch.cat((x[:, 2][:, None], x[:, 3][:, None], x[:, 7][:, None], x[:, 5][:, None]), dim=-1))[:, None]
        # y4 = self.and4(
        #    torch.cat((x[:, 3][:, None], x[:, 0][:, None], x[:, 8][:, None], x[:, 6][:, None]),dim=-1))[:, None]

        #temp = torch.cat((y1, y2), dim=-1)
        #y5 = self.or1(torch.cat((y1, y2), dim=-1))[:, None]
        #y6 = self.or2(torch.cat((y3, y4), dim=-1))[:, None]
        #y7 = self.or3(torch.cat((y5, y6), dim=-1))[:, None]
        # return self.or4(torch.cat((x[:, 4][:, None], y7), dim=-1))[:, None]
        return self.or1(torch.cat((x[:, 4][:, None], y3), dim=-1))[:, None]

    @torch.no_grad()
    def project_params(self):
        #elf.and1.project_params()
        #elf.and2.project_params()
        self.and3.project_params()
        #self.and4.project_params()
        self.or1.project_params()
        #self.or2.project_params()
        #self.or3.project_params()
        #self.or4.project_params()


"""
class LNN(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()

        #self.layers = nn.ModuleList()
        self.and1 = DynamicAnd(num_inputs=2, alpha=alpha)
        self.and2 = DynamicAnd(num_inputs=2, alpha=alpha)
        self.and3 = DynamicAnd(num_inputs=2, alpha=alpha)
        self.and4 = DynamicAnd(num_inputs=2, alpha=alpha)
        self.or1 = DynamicOr(num_inputs=2, alpha=alpha)
        self.or2 = DynamicOr(num_inputs=2, alpha=alpha)
        self.or3 = DynamicOr(num_inputs=2, alpha=alpha)
        self.or4 = DynamicOr(num_inputs=2, alpha=alpha)

    def forward(self, x):
        y1 = self.and1(
            torch.cat((x[:, 0][:, None], x[:, 5][:, None]), dim=-1))[:, None]
        y2 = self.and2(
            torch.cat((x[:, 1][:, None], x[:, 6][:, None]), dim=-1))[:, None]
        y3 = self.and3(
            torch.cat((x[:, 2][:, None], x[:, 7][:, None]), dim=-1))[:, None]
        y4 = self.and4(
            torch.cat((x[:, 3][:, None], x[:, 8][:, None]), dim=-1))[:, None]

        temp = torch.cat((y1, y2), dim=-1)
        y5 = self.or1(torch.cat((y1, y2), dim=-1))[:, None]
        y6 = self.or2(torch.cat((y3, y4), dim=-1))[:, None]
        y7 = self.or3(torch.cat((y5, y6), dim=-1))[:, None]
        return self.or4(torch.cat((x[:, 4][:, None], y7), dim=-1))[:, None]

    @torch.no_grad()
    def project_params(self):
        self.and1.project_params()
        self.and2.project_params()
        self.and3.project_params()
        self.and4.project_params()
        self.or1.project_params()
        self.or2.project_params()
        self.or3.project_params()
        self.or4.project_params()
"""
