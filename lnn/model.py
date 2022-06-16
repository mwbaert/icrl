from torch import nn
from lnn.neuron import Or, DynamicOr, DynamicAnd
import numpy as np
import torch


class LNN(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.and1 = DynamicAnd(num_inputs=8, alpha=0.9,
                                name="and1", final=False)
        #self.and2 = DynamicAnd(num_inputs=8, alpha=0.9,
        #                        name="and2", final=False)
        self.and3 = DynamicAnd(num_inputs=8, alpha=0.9,
                                name="and3", final=False)
        #self.and4 = DynamicAnd(num_inputs=8, alpha=0.9,
        #                        name="and4", final=False)

        self.or1 = DynamicOr(num_inputs=2, alpha=0.9, name="or1")

        self.layers = []
        self.layers.append(self.and1)
        #self.layers.append(self.and2)
        self.layers.append(self.and3)
        #self.layers.append(self.and4)
        self.layers.append(self.or1)

    def forward(self, x):
        y1 = self.and1(x)[:, None]
        #y2 = self.and2(x)[:, None]
        y3 = self.and3(1-x)[:, None]
        #y4 = self.and4(1-x)[:, None]

        return self.or1(torch.cat((y1, y3), dim=-1))[:, None]

    @torch.no_grad()
    def project_params(self):
        self.and1.project_params()
        #self.and2.project_params()
        self.and3.project_params()
        #self.and4.project_params()
        self.or1.project_params()

    def regLoss(self):
        loss = 0
        for layer in self.layers:
            loss += layer.regLoss()

        return loss/len(self.layers)

    def print(self):
        for layer in self.layers:
            print(layer.name, end=': ')
            for param in layer.parameters():
                print(param.data)


"""
class LNN(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.and10 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and10", final=False)
        self.and20 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and20", final=False)
        self.and30 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and30", final=False)
        self.and40 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and40", final=False)
        self.and50 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and50", final=False)
        self.and60 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and60", final=False)
        self.and70 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and70", final=False)

        self.and11 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and11", final=False)
        self.and21 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and21", final=False)
        self.and31 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and31", final=False)
        self.and41 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and41", final=False)
        self.and51 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and51", final=False)
        self.and61 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and61", final=False)
        self.and71 = DynamicAnd(num_inputs=2, alpha=0.75,
                                name="and71", final=False)

        self.or1 = DynamicOr(num_inputs=2, alpha=0.75, name="or1")

        self.layers = []
        self.layers.append(self.and10)
        self.layers.append(self.and20)
        self.layers.append(self.and30)
        self.layers.append(self.and40)
        self.layers.append(self.and50)
        self.layers.append(self.and60)
        self.layers.append(self.and70)

        self.layers.append(self.and11)
        self.layers.append(self.and21)
        self.layers.append(self.and31)
        self.layers.append(self.and41)
        self.layers.append(self.and51)
        self.layers.append(self.and61)
        self.layers.append(self.and71)

        self.layers.append(self.or1)

        #self.or1 = DynamicOr(num_inputs=2, alpha=0.75, final=False)
        #self.or2 = DynamicOr(num_inputs=2, alpha=0.75)
        #self.or3 = DynamicOr(num_inputs=2, alpha=0.75)
        #self.or4 = DynamicOr(num_inputs=2, alpha=0.75)

    def forward(self, x):
        y10 = self.and10(
            torch.cat((x[:, 0][:, None], x[:, 1][:, None]), dim=-1))[:, None]
        y20 = self.and20(
            torch.cat((x[:, 2][:, None], x[:, 3][:, None]), dim=-1))[:, None]
        y30 = self.and30(
            torch.cat((x[:, 4][:, None], x[:, 5][:, None]), dim=-1))[:, None]
        y40 = self.and40(
            torch.cat((x[:, 6][:, None], x[:, 7][:, None]), dim=-1))[:, None]
        y50 = self.and50(torch.cat((y10, y20), dim=-1))[:, None]
        y60 = self.and60(torch.cat((y30, y40), dim=-1))[:, None]
        out1 = self.and70(torch.cat((y50, y60), dim=-1))[:, None]

        y11 = self.and11(
            torch.cat((1.0-x[:, 0][:, None], 1.0-x[:, 1][:, None]), dim=-1))[:, None]
        y21 = self.and21(
            torch.cat((1.0-x[:, 2][:, None], 1.0-x[:, 3][:, None]), dim=-1))[:, None]
        y31 = self.and31(
            torch.cat((1.0-x[:, 4][:, None], 1.0-x[:, 5][:, None]), dim=-1))[:, None]
        y41 = self.and41(
            torch.cat((1.0-x[:, 6][:, None], 1.0-x[:, 7][:, None]), dim=-1))[:, None]
        y51 = self.and51(torch.cat((y11, y21), dim=-1))[:, None]
        y61 = self.and61(torch.cat((y31, y41), dim=-1))[:, None]
        out2 = self.and71(torch.cat((y51, y61), dim=-1))[:, None]

        return self.or1(torch.cat((out1, out2), dim=-1))[:, None]

        #temp = torch.cat((y1, y2), dim=-1)
        #y5 = self.or1(torch.cat((y1, y2), dim=-1))[:, None]
        #y6 = self.or2(torch.cat((y3, y4), dim=-1))[:, None]
        #y7 = self.or3(torch.cat((y5, y6), dim=-1))[:, None]
        # return self.or4(torch.cat((x[:, 4][:, None], y7), dim=-1))[:, None]
        # return self.or1(torch.cat((y2, y3), dim=-1))[:, None]

    @torch.no_grad()
    def project_params(self):
        self.and10.project_params()
        self.and20.project_params()
        self.and30.project_params()
        self.and40.project_params()
        self.and50.project_params()
        self.and60.project_params()
        self.and70.project_params()
        self.and11.project_params()
        self.and21.project_params()
        self.and31.project_params()
        self.and41.project_params()
        self.and51.project_params()
        self.and61.project_params()
        self.and71.project_params()
        self.or1.project_params()
        # self.or1.project_params()
        # self.or2.project_params()
        # self.or3.project_params()
        # self.or4.project_params()

    def regLoss(self):
        loss = 0
        for layer in self.layers:
            loss += layer.regLoss()

        return loss/len(self.layers)

    def print(self):
        for layer in self.layers:
            print(layer.name, end=': ')
            for param in layer.parameters():
                print(param.data)

"""
