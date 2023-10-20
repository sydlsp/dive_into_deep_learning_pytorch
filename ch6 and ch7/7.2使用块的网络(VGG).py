import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

"""
VGG实际上是更深更规整的Alexnet
"""

"""
，其实就是卷积层和池化层的组合
"""
def vgg_block(num_convs,in_channels,out_channels):  #卷积层的个数，输入通道，输出通道
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))  #vgg网络的每个卷积层并不改变图像的大小
        layers.append(nn.ReLU())
        in_channels=out_channels#过了第一层，其他每个卷积层的输入和输出通道数是一样的
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))  #是在这里改变的 new_size=(old_size-2)/2+1=old_size/2
    return nn.Sequential(*layers)  #*表示解包

"""
定义vgg_11的网络架构
"""
conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512))  #定义每个vgg块的卷积层数以及输出通道数，vgg网络一般来说有五个vgg块构成
def vgg(conv_arch):
    conv_blks=[]
    in_channels=1
    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels=out_channels #通道数要对齐
    #加上全连接层返回最终的vgg_11网络
    return nn.Sequential(*conv_blks,nn.Flatten(),
                         nn.Linear(out_channels*7*7,4096),nn.ReLU(),nn.Dropout(p=0.5),
                         nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),
                         nn.Linear(4096,10))

net=vgg(conv_arch)

"""
检查网络参数
"""
test_x=torch.randn(size=(1,1,224,224))
for blk in net:
    test_x=blk(test_x)
    print(blk.__class__.__name__,'output shape\t',test_x.shape)

"""
为了使网络能够训练，减少一下通道数
"""
ratio=4
small_conv_arch=[(pair[0],pair[1]//ratio) for pair in conv_arch]

lr,num_epoches,batch_size=0.05,10,64
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=224)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
回顾一下train里有什么：
初始化网络参数，把网络放到gpu上，定义优化函数，损失函数，num_epoches轮大循环，在每个小循环中，把训练数据放到gpu上，
然后就是常见过程 
"""
d2l.train_ch6(net,train_iter,test_iter,num_epoches,lr,device=device)
plt.show()

"""
数据量比较大跑一跑看看效果就好，准确率从跑的结果上来看比较高
"""

