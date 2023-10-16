import torch
from torch import nn

test_A=torch.arange(0,9,dtype=torch.float32).reshape((1,1,3,3))
print(test_A)
print('_'*100)
net=nn.AdaptiveAvgPool2d((1,1))
print(net(test_A))

