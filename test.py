from torch.utils.data import TensorDataset, DataLoader
from lnn.model import LNN
from torch import nn
import numpy as np
import torch


class LogicalNet(nn.Module):
    def __init__(self):
        super(LogicalNet, self).__init__()

        self.model = LNN()
        self.print_weights()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.tensor) -> torch.tensor:
        y = self.model(x)
        #y = torch.mean(y, -1)
        #y = y[:, None]

        return y

    def print_weights(self):
        # print(self.model.and1.weights)
        # print(self.model.and2.weights)
        print(self.model.and1.weights)
        print(self.model.and2.weights)
        print(self.model.and3.weights)
        print(self.model.and4.weights)
        print(self.model.or1.weights)
        print(self.model.or2.weights)
        print(self.model.or3.weights)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for i in range(40):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # for name, param in model.named_parameters():
            #    print(name, torch.isfinite(param.grad).all())
            #    print(param)
            #    print(param.grad)

            # model.print_weights()
            optimizer.step()

            if batch % 1 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # model.print_weights()
    model.eval()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        print(X)
        print(pred)
        print()
    model.print_weights()


x = [np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
     np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
     np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
     np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
     np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
     np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
     np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
     np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
     np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
     np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]

y = [np.array([1.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([1.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([1.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([0.0]),
     np.array([1.0])]

dataloader = DataLoader(TensorDataset(
    torch.Tensor(x), torch.Tensor(y)), batch_size=1, shuffle=False)

model = LogicalNet()

train(dataloader, model, model.criterion, model.optimizer)
