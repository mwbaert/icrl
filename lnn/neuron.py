from re import X
from torch import nn, no_grad
from lnn.utils import val_clamp
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np


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

        # x = val_clamp(1 - self.bias + (x @ self.weights))
        x = torch.sigmoid(1 - self.bias + (x @ self.weights))
        return x.view(-1, 2, 1)


class DynamicNeuron(nn.Module):
    def __init__(self, num_inputs, alpha=0.5):
        super().__init__()

        self.num_inputs = num_inputs
        #self.weights = nn.Parameter(torch.rand(self.num_inputs))
        self.weights = nn.Parameter(torch.Tensor(self.num_inputs))
        torch.nn.init.constant_(self.weights, 0.1)
        #torch.nn.init.xavier_uniform_(self.weights)
        self.f = DynamicActivation(self.num_inputs, alpha)
        self.kappa = None

    def forward(self, x):
        x = x @ self.weights
        # update activation using weights (should go to DynamicNeuron.forward)
        self.f.update_activation(self.weights, self.kappa)

        # call activation function
        return self.f(x)

    def plotActivation(self):
        self.f.plot()


class DynamicOr(DynamicNeuron):
    def __init__(self, num_inputs, alpha=0.6):
        super().__init__(num_inputs, alpha)
        self.kappa = torch.tensor(0.0)


class DynamicAnd(DynamicNeuron):
    def __init__(self, num_inputs, alpha=0.6):
        super().__init__(num_inputs, alpha)
        self.kappa = torch.tensor(1.0)

class Attention(nn.Module):
    def __init__(self, num_inputs):
        super().__init__(num_inputs)
        
        self.num_inputs = num_inputs
        self.weights = nn.Parameter(torch.Tensor(self.num_inputs))
        self.temp = 1

    def forward(self, x):
        one_hot = F.gumbel_softmax(self.weights, tau=self.temp, hard=True)
        index = (one_hot == 1).nonzero(as_tuple=True)[0]

        # this is probaby not correct
        return x[index]

    def update_temp(self):
        # TODO
        print("temp update not implemented")

class DynamicActivation(nn.Module):
    def __init__(self, num_inputs, alpha=0.6):
        super().__init__()

        # bound given in paper
        assert alpha > (num_inputs/(num_inputs+1)
                        ), "alpha should be bigger than n/(n+1)"

        self.num_inputs = num_inputs
        self.alpha = torch.tensor(alpha)
        self.eps = 1e-2

        self.x_min = 0
        self.x_f = 1 - self.alpha
        self.x_t = self.alpha
        self.x_mid = 0.5
        self.x_max = 1
        self.y_t = self.alpha
        self.y_f = 1 - self.alpha

    def forward(self, x):
        y = torch.zeros_like(x) - 1

        a = (2*torch.log((1-self.alpha)/self.alpha))/(self.x_f - self.x_t)
        b = torch.log(self.alpha/(1-self.alpha)) + (a*self.x_f)
        #b = torch.clamp(b, max=70)

        y = 1/(1+torch.exp((-a*x)+b))

        if torch.any(y < 0) or torch.any(y > 1) or torch.any(y == -1):
            raise ValueError(
                "output of activation expected in [0, 1], " f"received {y}"
            )
        return y

    def plot(self):
        x_max_ = self.x_max.item()
        x = np.arange(0, x_max_, 0.1)
        x = [[x[i], x[i]] for i in range(len(x))]
        x = torch.tensor(x)
        x_ = x[:, 0].detach().numpy()
        with torch.no_grad():
            y = self.forward(x)
        y_ = y[:, 0].detach().numpy()

        plt.plot(x_, y_)
        plt.grid()
        plt.show()

    def input_regions(self, x) -> torch.Tensor:
        result = torch.zeros_like(x)
        result = result.masked_fill(
            (self.x_min <= x) * (x <= self.x_f), 1)
        result = result.masked_fill((self.x_f < x) * (x < self.x_t), 2)
        result = result.masked_fill(
            (self.x_t <= x) * (x <= self.x_max), 3)
        if torch.any(result == 0):
            raise ValueError(
                "Unknown input regions. Expected all values from "
                f"[1, 2, 3], received  {result}"
            )
        return result

    def update_activation(self, weights, kappa):
        # bias = weights[0] = weights[1:].sum()
        bias = weights.sum()
        self.x_max = bias

        # do not know what this means or what I should choose.
        # TODO look at paper
        # w_m = self.TransparentMax.apply(
        #    weights.max()
        #    if self.w_m_slacks == "max"
        #    else weights.mean()
        #    if self.w_m_slacks == "mean"
        #    else weights.min()
        # )
        # max is the default option
        w_m = weights.max()

        # extra operation which are not mentioned in the paper
        # I think this is to stabilize training and prevent Nan values
        n = weights.shape[-1]
        k = 1 + kappa * (n - 1)
        self.x_f = bias - self.alpha * (
            w_m + ((n - k) / (n - 1 + self.eps)) * (bias - w_m)
        )
        self.x_t = self.alpha * \
            (w_m + ((k - 1) / (n - 1 + self.eps)) * (bias - w_m))
        self.g_f = self.divide(self.y_f, self.x_f, fill=0)
        self.g_z = self.divide(self.y_t - self.y_f, self.x_t -
                               self.x_f, fill=float("inf"))
        self.g_t = self.divide(1 - self.y_t, self.x_max - self.x_t, fill=0)
        self.g_f_inv = self.divide(torch.ones_like(
            self.g_f), self.g_f, fill=float("inf"))
        self.g_z_inv = self.divide(torch.ones_like(self.g_z), self.g_z, fill=0)
        self.g_t_inv = self.divide(torch.ones_like(
            self.g_t), self.g_t, fill=float("inf"))
        uniques = [self.x_min, self.x_f, self.x_t, self.x_max]
        if len(uniques) < len(set(uniques)):
            raise ValueError(
                "expected unique values for input control "
                f"points, received {uniques}"
            )

    @ staticmethod
    def divide(divident: torch.Tensor, divisor: torch.Tensor, fill=1.0) -> torch.Tensor:
        """
        Divide the bounds tensor (divident) by weights (divisor) while
            respecting gradient connectivity
        shortcurcuits a div 0 error with the fill value
        """
        shape = divident.shape
        if divident.dim() < 2:
            divident = divident.reshape(1, -1)
        div = divident.masked_select(
            divisor != 0) / divisor.masked_select(divisor != 0)
        result = divident.masked_scatter(divisor != 0, div)
        result = result.masked_fill(divisor == 0, fill)
        return result.reshape(shape)
