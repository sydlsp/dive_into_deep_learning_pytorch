import torch
from torch import nn
from d2l import torch as d2l

"""
dropout我觉得其实是避免某些神经元”话语权”过大，因为每个神经元都存在一定概率被丢弃，那么神经元的“话语权”自然会趋于一致
同时。也正是因为每个神经元有一定概率会被丢弃。所以神经元之间的相互关系会被削弱，从一定程度上减少过拟合情况。
"""
"""
手动定义dropout函数
"""

def drop_layer(X,dropout):  #dropout是保留概率
    assert 0<=dropout<=1 #也就是说丢弃的概率是在0-1之间的
    if (dropout==0):
        return torch.zeros_like(X)
    if (dropout==1):
        return X

    mask=(torch.randn(X.shape)>dropout).float() #mask用来随机决定哪一个神经元被丢弃掉
    # 这里要着重注意乘号并不是矩阵乘法，而是对应位置相乘结果[1,2,6]*[1,3,4]=[ 1,  6, 24]，下面除以(1-dropout)
    #对应的是不改变X的整体期望，实际上就是dropout的定义公式
    return mask*X/(1.0-dropout)

"""
定义模型参数
"""
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

"""
定义模型
"""
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):  #我们写的Net类继承了nn.Module类，需要我们在自己的类中重写init函数以及forward函数
    def __init__(self,num_inputs, num_outputs, num_hiddens1, num_hiddens2,is_training=True):
        #这里指按照父类nn.Module初始化的方法来初始化()里的东西，但括号里没有参数，我觉得这句话在本案例中可以不要，可以实验一下
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.training=is_training
        self.lin1=nn.Linear(num_inputs,num_hiddens1)
        self.lin2=nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3=nn.Linear(num_hiddens2,num_outputs)
        self.relu=nn.ReLU()
        #后四行都是实例化一个类

        #pytorch文档中使用线性层的方法
        # >> > m = nn.Linear(20, 30)
        # >> > input = torch.randn(128, 20)
        # >> > output = m(input)
        # >> > print(output.size())
        # torch.Size([128, 30])

    def forward(self,X):
        #这里直接用 类名(参数)的原因是因为框架作者使用了魔法函数 __call__(应该是这样的)
        #在框架作者用了魔法函数后 self.lin1(X.reshape(-1,self.num_inputs)) 等价于 self.lin1.forward((X.reshape(-1,self.num_inputs)))  这个可以点Linear类源码找到
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))#实际上是传递一开始的输入向量X并进行计算
        if (self.training==True):
            H1=drop_layer(H1,dropout1)
        H2=self.relu(self.lin2(H1))
        if (self.training == True):
            H2 = drop_layer(H2, dropout2)
        out=self.lin3(H2)

        return out

net=Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)


"""
训练和测试
"""

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
#根据上面魔法函数的理解 训练过程中有一个y_hat=net(X) (这句话可以查train_ch3函数里的train_epoch_ch3函数找到) 应该也是这个原理
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()




