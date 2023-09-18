import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

"""
生成数据集  使用三阶多项式生成训练数据和测试数据的标签
y=5+1.2x-3.4*x^2/2+5.6*x^3/6+噪声 噪声服从(0,0.01)的正态分布
"""
max_degree=20 #多项式的最大阶数
n_train,n_test=100,100 #训练数据和测试数据集的大小

#x各次方前面的系数
true_w=np.zeros(max_degree)
true_w[:4]=np.array([5,1.2,-3.4,5.6])

features=np.random.normal(size=(n_train+n_test,1))  #实际上是生成x
np.random.shuffle(features)
poly_features=np.power(features,np.arange(max_degree).reshape(1,-1))  #这里得到(1,x,x^2,x^3……,x^19)  这里用到了python的广播机制
for i in range(max_degree):
    poly_features[:,i]/=math.gamma(i+1)  #gramm(n)=(n-1)!   这一行代码表示对poly_features的每一列除以一个数

labels=np.dot(poly_features,true_w)  #得到y
labels+=np.random.normal(scale=0.1,size=labels.shape)

"""
将numpy数组转化为tensor
"""
# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

"""
对模型进行训练和测试
"""
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

"""
定义训练函数
"""
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))  #单层神经网络，实际上这个函数是调整输入神经元的数目来验证过拟合和欠拟合的
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size=batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size=batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01) #小批量随机梯度优化
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
#train(poly_features[:n_train, :4], poly_features[n_train:, :4],labels[:n_train], labels[n_train:])
# 从多项式特征中选择前2个维度，即1和x  欠拟合
train(poly_features[:n_train, :2], poly_features[n_train:, :2],labels[:n_train], labels[n_train:])
# 从多项式特征中选取所有维度   过拟合
#train(poly_features[:n_train, :], poly_features[n_train:, :],labels[:n_train], labels[n_train:])