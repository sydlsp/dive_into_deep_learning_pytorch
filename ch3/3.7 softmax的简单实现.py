import torch
from torch import nn
from d2l import torch as d2l

"""
加载数据
"""

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

"""
利用flatten来平整输入的形状
"""
#flatten的简单演示
# test=torch.arange(0,10).reshape(2,5)
# print (test)
# print(test.flatten())
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01) #对每一层初始化参数 均值为0，方差为0.01
net.apply(init_weight)

loss=nn.CrossEntropyLoss(reduction='none')
updater=torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs=10
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)#实际上这行代码就是把之前从零实现的东西都集合到一起了