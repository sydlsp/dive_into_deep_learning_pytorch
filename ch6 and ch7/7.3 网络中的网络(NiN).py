import torch
from torch import nn
from d2l import torch as d2l
import time

"""
Alexnet和VGG都在扩大和加深传统的卷积神经网络，但在最后用了全连接层，这样网络的参数会非常大
NiN：针对这个问题提出了一种新的想法：在每个像素通道上分别使用多层感知机，其实是减少了参数量
NiN虽然目前使用的不是很多，但是NiN的思想要有了解
"""

def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
        nn.ReLU(),nn.Conv2d(out_channels,out_channels,kernel_size=1),#实际上图像的大小没有改变
        nn.ReLU(),nn.Conv2d(out_channels,out_channels,kernel_size=1),#同理，图像的大小没有改变
        nn.ReLU()
    )

#这儿的通道数以及具体的参数都是和VGG相对应的
net=nn.Sequential(
    nin_block(1,96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(3,stride=2),
    nin_block(96,256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(3,stride=2),
    nin_block(256,384,kernel_size=3,stride=1,padding=1),
    nn.MaxPool2d(3,stride=2),nn.Dropout(p=0.5),
    nin_block(384,10,kernel_size=3,stride=1,padding=1), #10是因为fashion_minst有10个类别
    nn.AdaptiveAvgPool2d((1,1)),  #这里应该相当于把每一个通道求平均，平均成一个1*1的张量，实际上就是一个数字了，但通道数没有变
    nn.Flatten()#拉平了实际上对于一个样本也就是一个数
)

"""
查看每层的输出形状
"""
test_x=torch.rand(size=(1,1,224,224))
for layer in net:
    test_x=layer(test_x)
    print(layer.__class__.__name__,'output shape:\t',test_x.shape)


device=device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr,num_epoches,batch_size=0.1,10,128
start_time=time.time()
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=224)
d2l.train_ch6(net,train_iter,test_iter,num_epoches,lr,device=device)
end_time=time.time()
print(end_time-start_time)

