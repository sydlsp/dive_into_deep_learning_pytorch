import torch
import torchvision
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from d2l import torch as d2l



"""
CIFAR10数据集的加载
"""
# all_images=torchvision.datasets.CIFAR10(train=True,root='./data',download=True)
#print(all_images.data.shape)  #打印一下训练集的形状是50000张图，每张图3通道，32*32大小
#图像展示
# d2l.show_images([all_images[i][0] for i in range(32)],4,8,scale=0.8)
# plt.show()
#用自己的方法查看一张图像
# plt.imshow(all_images[0][0])
# plt.show()

"""
定义训练和测试时图片增广的方式
"""
train_augs=torchvision.transforms.Compose(
    [torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.ToTensor()])  #左右翻转，变成向量


test_augs=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  #变成向量


"""
定义辅助函数，用于读取图像和应用图像增广
"""
def load_cifar10(is_train,augs,batch_size):
    #这里用数据增广并没有说增加了图片的张数，可以理解为增加了图片的特征数量
    dataset=torchvision.datasets.CIFAR10(root='./data',train=is_train,transform=augs,download=True)
    dataloader=torch.utils.data.DataLoader(
        dataset,batch_size=batch_size,shuffle=is_train,num_workers=4  #num_workers是设置进程数
    )
    return  dataloader

"""
定义训练函数，其实是对于每个batch_size来言的
"""
def train_batch_13(net,X,y,loss,trainer,devices):
    """代码是利用多GPU训练的代码"""
    if isinstance(X,list):
        X=[x.to(devices[0]) for x in X]
    else:
        X=X.to(devices[0])

    y=y.to(devices[0])
    #设置训练模型
    net.train()
    trainer.zero_grad()
    pred=net(X)
    l=loss(pred,y)
    l.sum().backward()
    trainer.step()
    train_loss_sum=l.sum()
    train_acc_sum=d2l.accuracy(pred,y)
    return train_loss_sum,train_acc_sum


def train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs,devices=d2l.try_all_gpus()):
    """代码是利用多GPU运算的代码"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])  #多GPU训练的代码，暂时可以不用管

    for epoch in range(num_epochs):
        #4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i,(features,labels) in enumerate(train_iter):
            timer.start()
            l,acc=train_batch_13(net,features,labels,loss,trainer,devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


batch_size,devices,net=256,d2l.try_all_gpus(),d2l.resnet18(10,3)


"""
显式的初始化网络权重，其实不写也会自动初始化的
"""
def init_weights(m):
    if type(m) in [nn.Linear,nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)


"""
总函数
"""
def train_with_data_aug(train_augs,test_augs,net,batch_size,devices,lr=0.001):
    train_iter=load_cifar10(True,train_augs,batch_size)
    test_iter=load_cifar10(False,test_augs,batch_size)
    loss=nn.CrossEntropyLoss(reduction='none')   #reduction相当于返回一个列表，列表中存放的是每个样本的损失值 reduction='sum'相当于对这些损失值求和，reduction='mean'相当于对这些损失值求平均
    trainer=torch.optim.Adam(net.parameters(),lr=lr)
    train_ch13(net,train_iter,test_iter,loss,trainer,10,devices)


train_with_data_aug(train_augs,test_augs,net,batch_size,devices)
plt.show()







