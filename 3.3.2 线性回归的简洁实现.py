import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

"""生成数据集"""
ture_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(ture_w,true_b,1000)

"""读取数据集"""
def load_array(data_arrays,batch_size,is_train=True):
    """"""
    """构造一个pytorch数据迭代器"""
    dataset=data.TensorDataset(*data_arrays)  #相当于把data_arrays打包
    return data.DataLoader(dataset,batch_size,shuffle=is_train)#在这里返回的是一个迭代器

batch_size=10
data_iter=load_array((features,labels),batch_size)

"""查看迭代器中输出的方式"""
#print (next(iter(data_iter)))

"""定义模型"""

net=nn.Sequential(nn.Linear(2,1))
#上面的代码 sequential 可以理解为 lists of layers  实际上存放的是每一层的相关参数 nn.liner 表示输入是二维的 输出是一维的

"""初始化模型参数"""
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

"""定义损失函数"""
loss=nn.MSELoss()

"""定义优化算法,直接用小批量随机梯度下降法"""
trainer=torch.optim.SGD(net.parameters(),lr=0.01)  #给定的是网络中的所有参数 以及学习率

"""训练过程"""
num_epochs=3


for num_epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()#反向传播
        trainer.step() #参数更新
    #简要评估训练效果
    l=loss(net(features),labels)
    print (f'epoch {num_epoch+1}, loss {l:f}')

"""输出第1层的权重和偏差"""
print(net[0].weight)
print(net[0].bias)

