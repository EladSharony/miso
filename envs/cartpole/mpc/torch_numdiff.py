import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import itertools


def grad(net, inputs, eps=1e-4):
    """
    Computes the gradient of a network with respect to its inputs using finite differences.
    :param net: network
    :param inputs: inputs
    :param eps: epsilon
    :return: gradient of the network with respect to the inputs
    """
    assert (inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xp = torch.zeros(nBatch * nDim, nDim, dtype=inputs.dtype, device=inputs.device)
    xn = xp.clone()
    e = 0.5 * eps * torch.eye(nDim, dtype=inputs.dtype, device=inputs.device)
    for b in range(nBatch):
        for i in range(nDim):
            xp[b * nDim + i] = (inputs.data[b].clone() + e[i])
            xn[b * nDim + i] = (inputs.data[b].clone() - e[i])
    xs = torch.cat((xp, xn))
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fs_p, fs_n = torch.split(fs, nBatch * nDim)
    g = ((fs_p - fs_n) / eps).view(nBatch, nDim, fDim).squeeze(2)
    return g


def hess(net, inputs, eps=1e-4):
    """
    Computes the Hessian of a network with respect to its inputs using finite differences.
    :param net: network
    :param inputs: inputs
    :param eps: epsilon
    :return: Hessian of the network with respect to the inputs
    """
    assert (inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xpp = torch.zeros(nBatch * nDim * nDim, nDim, dtype=inputs.dtype, device=inputs.device)
    xpn, xnp, xnn = xpp.clone(), xpp.clone(), xpp.clone()
    e = eps * torch.eye(nDim, dtype=inputs.dtype, device=inputs.device)
    for b, i, j in itertools.product(range(nBatch), range(nDim), range(nDim)):
        xpp[b * nDim * nDim + i * nDim + j] = (inputs.data[b].clone() + e[i] + e[j])
        xpn[b * nDim * nDim + i * nDim + j] = (inputs.data[b].clone() + e[i] - e[j])
        xnp[b * nDim * nDim + i * nDim + j] = (inputs.data[b].clone() - e[i] + e[j])
        xnn[b * nDim * nDim + i * nDim + j] = (inputs.data[b].clone() - e[i] - e[j])
    xs = torch.cat((xpp, xpn, xnp, xnn))
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fpp, fpn, fnp, fnn = torch.split(fs, nBatch * nDim * nDim)
    h = ((fpp - fpn - fnp + fnn) / (4 * eps * eps)).view(nBatch, nDim, nDim, fDim).squeeze(3)
    return h


def test():
    torch.manual_seed(0)

    class Net(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = F.softplus(self.fc1(x))
            x = self.fc2(x).squeeze()
            return x

    # class Net(Function):
    #     def forward(self, inputs):
    #         Q = torch.eye(3,3)
    #         return 0.5*inputs.mm(Q).unsqueeze(1).bmm(inputs.unsqueeze(2)).squeeze()

    net = Net().double()
    nBatch = 4
    x = Variable(torch.randn(nBatch, 2).double())
    x.requires_grad = True
    y = net(x)
    y.backward(torch.ones(nBatch).double())
    print(x.grad)
    x_grad = grad(net, x, eps=1e-4)
    print(x_grad)
    x_hess = hess(net, x, eps=1e-4)
    print(x_hess)


if __name__ == '__main__':
    test()
