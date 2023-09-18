import torch
from torch import nn
from IPython import display
from d2l import torch as d2l
import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt




def get_dataloader_workers():
    return 1

"""设定小批量大小 得到测试数据迭代器和训练数据迭代器"""
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


"""初始化模型参数  之所以这样是因为图片是28*28=784 拉成一条 一共有10个类别"""
num_inputs=784
num_outputs=10
"""初始化权重矩阵 权重矩阵的m行n列 m行对应输入特征数 n列对应输出特征数目 这里一定要牢记"""
w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)

params=[w,b]

"""定义Softmax函数"""
def Softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)  #按行求和
    return X_exp/partition  #这里应用到了广播机制

"""定义模型:
   模型实际上还是简单的单层神经网络模型 在使用神经网络矩阵乘法的时候注意是x*w 还有要更改一下矩阵的维度让其能够正确相乘"""
def net(X):
    return (Softmax(torch.matmul(X.reshape(-1,w.shape[0]),w)+b))

"""定义损失函数"""
def cross_entropy (y_hat,y):
    return -torch.log(y_hat[range[len(y_hat)],y])

"""
定义分类精度
"""
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1: #在这里y_hat.shape 返回的是 x个x行x列的形式
        y_hat=y_hat.argmax(axis=1)  #在这里是axis=1表示按行来进行求最大值 返回的是每行所在最大值的下标
    cmp=y_hat.type(y.dtype)==y #这里是把y_hat的数据类型转换成y的数据类型再进行比较，返回的是cmp是bool型的数据
    return float(cmp.type(y.dtype).sum())   #这里就得到了预测正确的数量
"""
计算在指定数据集上模型的精度
"""
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

def evaluate_accuracy(net,data_iter):

    if isinstance(net,torch.nn.Module):  #isinstance函数的作用就是判断net和torch.nn.Mudoule 的类型是否相同
        net.eval()#进入评估模式，网络的参数并不会被更新

    matric=Accumulater(2)   #实际上是度量，累加得出正确的预测数以及预测总数，在这里实际上是实例化了一个类

    with torch.no_grad():#不需要反向传播
        for X,y in data_iter:
            matric.add(accuracy(net(X),y),y.numel())   #这里需要理解一下zip的是一个迭代器中（准确预测的数目，总个数）
    return  matric[0]/matric[1]  #先这样简单理解一下吧

"""
训练函数
"""
loss=nn.CrossEntropyLoss()
updater=torch.optim.SGD(params,lr=0.01)


def train_epoch_ch3(net,train_iter,loss,updater):
    #模型训练一轮
    if isinstance(net,torch.nn.Module):
        net.train()

    matric=Accumulater(3)  #分别是训练损失总和，训练准确数总和以及样本数
    for X,y in train_iter:

        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):#这一行就是说如果采用的是框架给出的优化器
            updater.zero_grad()#梯度清零
            l.mean().backward()#在这里对l右边的东西进行求导 右边的东西是什么？  实际上经过分析得到是参数矩阵
            updater.step()  #参数矩阵进行更新
        else:   #这里的情况是没有采用框架
            l.sum.backward()
            updater(X.shape[0])
        #print (l)
        matric.add(float(l.sum())*len(y), accuracy(y_hat, y), y.numel())
    return matric[0]/matric[2],matric[1]/matric[2]


"""
定义实用程序类绘图
"""
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        plt.draw()
        plt.pause(0.001)
        display.clear_output(wait=True)


"""
定义多轮训练模型
"""
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):#思考一下要什么参数
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 3],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):# 多轮训练
        train_matrics=train_epoch_ch3(net,train_iter,loss,updater)#测试
        test_acc=evaluate_accuracy(net,test_iter)#评估
        print('{}'.format(epoch)+","+'{}'.format(train_matrics))
        animator.add(epoch + 1, train_matrics + (test_acc,))
    train_loss, train_acc = train_matrics


num_epochs=10
train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
d2l.plt.show()