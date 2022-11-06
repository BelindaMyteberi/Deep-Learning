#As a last exercise, we would like to make use of existing neural network objects of the PyTorch library. Here, most of the code is
#already implemented for you. You are only asked to find where the error gradient of the first weight parameter has been stored, and
#to print it.
import torch
import torch.nn as nn
# 1. Get the data and parameters
X,T = utils.getdata()
W,B = utils.getparams()
# 2. Convert to PyTorch objects
X = torch.Tensor(X)
T = torch.Tensor(T)
W = [torch.nn.Parameter(torch.Tensor(w.T)) for w in W]
B = [torch.nn.Parameter(torch.Tensor(b)) for b in B]
# 3. Build the neural network
net = torch.nn.Sequential(
nn.Linear(4,6),nn.ReLU(),
nn.Linear(6,6),nn.ReLU(),
nn.Linear(6,6),nn.ReLU(),
nn.Linear(6,1))
for l,w,b in zip(list(net)[::2],W,B):
l.weight = w
l.bias = b
# 4. Compute the forward pass and the error gradient
Y = net.forward(X)
err = ((Y-T)**2).mean()
err.backward()
# 5. Show error gradient w.r.t. the 1st weight parameter (TODO: replace by your code)
solution.exercise3(net)