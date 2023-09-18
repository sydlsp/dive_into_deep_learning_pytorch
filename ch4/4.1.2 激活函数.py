import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

"""
    Relu激活函数
"""
x=torch.arange(-8.0,8.0,0.1,requires_grad=True)
y=torch.relu(x)
d2l.plot(x.detach(),y.detach(),'x','relu(x)',figsize=(5,2.5))
plt.show()
y.backward(x.detach(),retain_graph=True)