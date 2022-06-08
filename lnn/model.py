from torch import nn
from lnn.neuron import Or, DynamicOr, DynamicAnd
import numpy as np
import torch


class LNN(nn.Module):
    def __init__(self):
        super().__init__()

        #self.layers = nn.ModuleList()
        self.and1 = DynamicAnd(num_inputs=2, alpha=0.7)
        self.and2 = DynamicAnd(num_inputs=2, alpha=0.7)
        self.and3 = DynamicAnd(num_inputs=2, alpha=0.7)
        self.and4 = DynamicAnd(num_inputs=2, alpha=0.7)
        self.or1 = DynamicOr(num_inputs=2, alpha=0.7)
        self.or2 = DynamicOr(num_inputs=2, alpha=0.7)
        self.or3 = DynamicOr(num_inputs=2, alpha=0.7)

    def forward(self, x):
        #return self.or1(x)
        y1 = self.and1(torch.cat((x[:,0][:,None], x[:,4][:,None]), dim=-1))[:,None]
        y2 = self.and2(torch.cat((x[:,1][:,None], x[:,5][:,None]), dim=-1))[:,None]
        y3 = self.and3(torch.cat((x[:,2][:,None], x[:,6][:,None]), dim=-1))[:,None]
        y4 = self.and4(torch.cat((x[:,3][:,None], x[:,7][:,None]), dim=-1))[:,None]

        temp=torch.cat((y1, y2), dim=-1)
        y5 = self.or1(torch.cat((y1, y2), dim=-1))[:,None]
        y6 = self.or2(torch.cat((y3, y4), dim=-1))[:,None]
        return self.or3(torch.cat((y5,y6), dim=-1))[:,None]

    #def forward(self, x):
    #    #return self.or1(x)
    #    y1 = self.and1(torch.cat((x[:,:,0][:,:,None], x[:,:,4][:,:,None]), dim=2))[:,:,None]
    #    y2 = self.and2(torch.cat((x[:,:,1][:,:,None], x[:,:,5][:,:,None]), dim=2))[:,:,None]
    #    y3 = self.and3(torch.cat((x[:,:,2][:,:,None], x[:,:,6][:,:,None]), dim=2))[:,:,None]
    #    y4 = self.and4(torch.cat((x[:,:,3][:,:,None], x[:,:,7][:,:,None]), dim=2))[:,:,None]
#
    #    y5 = self.or1(torch.cat((y1, y2), dim=-1))[:,:,None]
    #    y6 = self.or2(torch.cat((y3, y4), dim=-1))[:,:,None]
    #    return self.or3(torch.cat((y5,y6), dim=-1))

    @torch.no_grad()
    def project_params(self):
        for layer in self.layers:
            pass
            # layer.project_params()
