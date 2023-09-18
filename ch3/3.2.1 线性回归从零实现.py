import random
import torch
from d2l import torch as d2l


"""利用代码合成数据集"""
def synthetic_data(w,b,num_examples):
    """"""
    """在这里我们生成一个y=xw+b+噪声的数据集"""
    x=torch.normal(0,1,(num_examples,len(w)))#生成一个均值为0，方差为1的正态分布 后面括号里的形状
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape(-1,1)

ture_w=torch.tensor([2,-3.4])
ture_b=4.2
features,labels=synthetic_data(ture_w,ture_b,1000)
"""至此，实现数据集已经生成了，下面进行读取数据集的操作"""

def data_iter(batch_size,features,labels):
    num_examples=len(features)  #读取数据集的总的个数
    indices=list(range(num_examples))#实际上就是生成了一个列表，列表里面是0，1到numexamples-1
    #下面进行的是打乱操作
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])  #在这里用min是防止越界超出的
        yield features[batch_indices],labels[batch_indices]
        """在这里yield比较抽象
        一个带有 yield 的函数就是一个 generator，
        它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，
        直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行
        可以理解为每次遍历时往features[batch_indices], labels[batch_indices]加入特征，第一次循环为            
        features[indices[0]
              ...
              indices[batch_size]"""

batch_size=10
"""对小批量阅读结果进行展示 没有实质性作用"""
# for X,y in data_iter(batch_size,features,labels):
#     print(X,'\n',y)
#     break

"""初始化模型参数"""
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

"""定义模型"""
def linreg(X,w,b):
    return torch.matmul(X,w)+b  #矩阵乘法

"""定义损失函数"""
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2


"""定义优化算法"""
def sgd(params,lr,batch_size):
    with torch.no_grad(): #在这里并不需要求梯度，只是用到梯度的计算结果
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_() #每次梯度清零

"""训练过程"""
lr=0.03
num_epoch=3
net=linreg
loss=squared_loss

for epoch in range(num_epoch):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():#这一行表示在评测的时候不进行优化
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss   {float(train_l.mean()):f}')
"""由于是人工数据集 知道真正的w和b计算一下三轮训练的差值 简要再看一下结果 """
print (ture_w-w.reshape(ture_w.shape))
print (ture_b-b)