import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

"""
加载数据
"""
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

"""
定义模型的初始化参数
"""
num_inputs,num_outputs,num_hiddens=784,10,256#这表明输入是784个单元，隐藏层是256个单元，最后分成10类
w1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))#也就是说b的初始化大小和下一层的大小是一样的
w2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True))
b2=nn.Parameter(torch.randn(num_outputs,requires_grad=True))

params=[w1,b1,w2,b2]  #将所有参数组成一个大矩阵
"""
定义模型
"""

def net(X):
    X=X.reshape(-1,num_inputs)
    H=torch.relu(X@w1+b1) #这里是第一个隐藏层的参数获得  就是 wx+b再激活
    return (H@w2+b2)  #实际上就是最后输出

loss=nn.CrossEntropyLoss()  #定义损失函数 实际上是把类里面的函数给取出来
"""
再搞一个累加器出来
"""
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1: #在这里y_hat.shape 返回的是 x个x行x列的形式
        y_hat=y_hat.argmax(axis=1)  #在这里是axis=1表示按行来进行求最大值 返回的是每行所在最大值的下标
    cmp=y_hat.type(y.dtype)==y #这里是把y_hat的数据类型转换成y的数据类型再进行比较，返回的是cmp是bool型的数据
    return float(cmp.type(y.dtype).sum())   #这里就得到了预测正确的数量

class Accumulater:
    """在n个变量上累加"""
    def __init__(self,n):
       self.data=[0.0]*n

    def add (self, *args):   #在这里*args表示任意多个无名参数，将其变成一个tuple
        self.data=[a+float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data=[0.0]*len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

"""
训练过程,这是书上给出的，下面要逐步拆解这个东西
"""
num_epochs,lr=10,0.1
updater=torch.optim.SGD(params,lr=lr)# 实际上是构造了一个优化器对象

def train(net,train_iter,num_epochs):#自己想一下要什么参数
    for epoch in range(num_epochs):#定义训练轮数
        if isinstance(net,torch.nn.Module):
            net.train()
        matric = Accumulater(3)
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y)

            if isinstance(updater,torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(X.shape[0])
            matric.add(float(l.sum()) * len(y), accuracy(y_hat, y), y.numel())
        print (matric[0]/matric[1],matric[1]/matric[2])

train(net,train_iter,3)





# d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
#
# d2l.plt.show()


