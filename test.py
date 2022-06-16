from torch.utils.data import TensorDataset, DataLoader
from lnn.model import LNN
from torch import nn
import numpy as np
import torch


class LogicalNet(nn.Module):
    def __init__(self, num_inputs):
        super(LogicalNet, self).__init__()

        self.model = LNN(num_inputs)
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
    for i in range(200):
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
            model.model.project_params()

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


and10 = [0.0489, 0.0041]
and20 = [0.0000, 5.6331]
and30 = [0.0000, 2.0223]
and40 = [2.7826, 3.2358]
and50 = [5.3796, 5.2826]
and60 = [3.5189, 2.8880]
and70 = [5.3398, 3.5037]
and11 = [5.5248, 5.5544]
and21 = [7.0914, 0.0000]
and31 = [4.9871, 3.2532]
and41 = [3.1845, 3.3110]
and51 = [5.8729, 5.2679]
and61 = [3.6962, 3.7909]
and71 = [5.2022, 3.9120]
or1 = [5.0019, 4.8384]

neurons = [and10,
           and20,
           and30,
           and40,
           and50,
           and60,
           and70,
           and11,
           and21,
           and31,
           and41,
           and51,
           and61,
           and71,
           or1]

max_val = np.max(neurons)
print(max_val)
neurons /= max_val

for neuron in neurons:
    print(neuron)

#x = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
#     np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
#     np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
#     np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
#     np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
#     np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
#     np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
#     np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
#     np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
#     np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
#     np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
#     np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
#     np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
#     np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
#     np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
#     np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])]
#
#y = [np.array([1.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([1.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([1.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([0.0]),
#     np.array([1.0])]
#
#dataloader = DataLoader(TensorDataset(
#    torch.Tensor(x), torch.Tensor(y)), batch_size=8, shuffle=True)
#
#model = LogicalNet(9)
#
#train(dataloader, model, model.criterion, model.optimizer)
#