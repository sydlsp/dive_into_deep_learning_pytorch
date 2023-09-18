import torch
from torch import nn
from d2l import torch as d2l

"""
初始化一个含有一个隐藏层的神经网络，神经元数分别为784，256，10 256那层结果得出来后用Relu函数激活
"""
net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

batch_size,lr,num_epoches=256,0.1,10
"""
定义损失函数以及优化函数
"""
loss=nn.CrossEntropyLoss()
updater=torch.optim.SGD(net.parameters(),lr=lr)
"""
数据导入与模型训练
"""
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epoches,updater)

