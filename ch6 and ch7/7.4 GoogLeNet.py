import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import time

"""
不得不说的是：GoogLeNet蕴含了太多的超参数，超参数的选择很可能是大规模实验超参数的结果
虽然GoogLeNet的网络结构很复杂，但相较于VGG的最后全连接，体现了类似NiN网络中利用1*1卷积代替全连接层可以使得参数量下降的好处
"""

"""
GoogLeNet的核心设计在于Inception块，电影《Inception》中译叫做《盗梦空间》
Inception的设计思想在于：如果我不知道该选择什么大小的卷积核，那我干脆就把卷积核1*1，3*3，5*5，池化等等都试一下，燃油组合起来。
有点像”千层饼“，这样就不仅仅局限于一种卷积核了，每种卷积核的作用通过通道数在所有通道数的比例体现。
有点集大成的感觉：”所有的我都来试一试“1。
"""

"""
定义Inception块，Inception块最值得把握的一个点是他只改变通道数的多少，并不改变图像的长和宽
"""
class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs): #**kwargs则是将一个可变的关键字参数的字典传给函数实参
        super(Inception,self).__init__(**kwargs)
        #线路1，单1*1的卷积层
        self.p1_1=nn.Conv2d(in_channels,c1,kernel_size=1)
        #线路2，1*1卷积接3*3卷积
        self.p2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        #线路3，1*1卷积接5*5卷积
        self.p3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        #线路4，3*3最大汇聚接1*1卷积
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1=F.relu(self.p1_1(x))
        p2=F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3=F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4=F.relu(self.p4_2(self.p4_1(x)))

        #在通道上连接输出
        return torch.cat((p1,p2,p3,p4),dim=1)  #dim=1 是怎么来的 对于卷积神经网络来说一般是四维的张量(批量大小,通道数,行,列)所以这里dim=1是按照通道数堆叠
"""
Google的五个大块
"""
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

"""
查看网络输出的大小
"""
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

"""
训练网络
"""
lr,num_epoches,batch_size=0.1,10,128
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_time=time.time()
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=96)
d2l.train_ch6(net,train_iter,test_iter,num_epoches,lr,device)
end_time=time.time()
print(end_time-start_time)









