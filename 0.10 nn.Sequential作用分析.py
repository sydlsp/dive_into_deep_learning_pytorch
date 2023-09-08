import torch
from torch import nn

net=nn.Sequential(nn.Linear(10,5))  #Linear 是构造线性层的 参数是输入维度，输出维度  在这里要分清神经网咯的参数个数并不是神经元的数目相加而是相乘
net_iter=net.parameters()
print (next(net_iter))