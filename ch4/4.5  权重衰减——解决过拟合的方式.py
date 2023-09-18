import torch
from torch import nn
from d2l import torch as d2l

"""
在训练网络的过程中，经常使用限制W的范围来减少过拟合的情况，例如可以强制性的把
||W||^2<θ来限制W的大小。
在实际过程中，一般采用软性的方式 即在损失函数后增加L2范数作为惩罚项来限制W的范围
"""

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

"""
定义函数初始化模型参数
"""
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

"""
定义L2范数惩罚
"""
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

"""
训练代码实现
"""
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())

train(lambd=0)

train(lambd=3)

d2l.plt.show()

"""
通过实验可以看出增加L2范数作为惩罚后，明显的W的L2范数减小，同时过拟合现象得到缓解
也就是权重衰减的含义
好
"""
































