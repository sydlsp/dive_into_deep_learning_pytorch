import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt

"""
卷积神经网络的卷积层其实是为了特征提取，池化层是为了增加平移不变性
对于卷积层来讲，最主要的是理解好输入通道、输入通道和卷积核的关系

可以按照这样的方式理解：
有几个输入通道，那么一个卷积核就深多少
有几个输入通道，就有几个卷积核
"""
"""
修改输入图片的大小
"""
class Reshape(nn.Module):
    def forward(self,x):
        return x.reshape(-1,1,28,28)

"""
定义网络
"""
net=nn.Sequential(Reshape(),nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2,stride=2),
                  nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2,stride=2),
                  nn.Flatten(),
                  nn.Linear(16*5*5,120),nn.Sigmoid(),
                  nn.Linear(120,84),nn.Sigmoid(),
                  nn.Linear(84,10)
                  )#网络最后不加softmax是因为CrossEntropyLoss自带Softmax

"""
模型检查
"""
X=torch.rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape\t',X.shape)

"""
数据加载
"""
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)


"""
定义训练和评价过程
"""
def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device=net(iter(net.parameters())).device
    metric=d2l.Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                X=[x.to(device) for x in X]  #这里暂时用不到是为了后续微调BERT使用的
            else:
                X=X.to(device)
            y=y.to(device)
            metric.add(d2l.accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]
def train(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) #实际上就是初始化神经网络的值
    net.apply(init_weights)
    print("training on device",device) #看一下是不是在gpu上工作的
    net.to(device) #把神经网络放到gpu上
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()  #自带了Softmax函数，自动进行了Softmax过程
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer,num_batches=d2l.Timer(),len(train_iter)
    for epoch in range(num_epochs):
        metric=d2l.Accumulator(3)  #用于统计指标
        net.train()
        for i,(X,y) in enumerate(train_iter):  #enumerate 相当于加了一个序号
            timer.start()
            optimizer.zero_grad()
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter,device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


"""
开始训练
"""
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train(net,train_iter,test_iter,num_epochs=10,lr=0.09,device=device)

