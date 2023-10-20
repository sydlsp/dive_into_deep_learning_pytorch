import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

"""
有这样一个问题：就是神经网络越深越好吗？
关于这个问题其实答案是否定的，神经网络越深可能最佳结果离最优目标距离越来越远
ResNet想了这样一件事f(x)=x+g(x) x可以理解为已经得到的结果范围，我训练新的网络g(x)其实是相当于把网络加深了
但f(x)表达式中存在我已经得到的结果范围，这样相加可以使得我的结果范围变大，也就是说应该不会使得网络越深结果越差的情况发生
同时由于f(x)=x+g(x)算梯度的时候其实是x的梯度+g(x)的梯度，有点乘法变加法的感觉(可以理解串联是乘法，并联是加法)，有效的减缓了梯度消失的问题，
也就是意味着ResNet可以使网络变得很深。
"""


"""
定义一个ResNet块，还是看图说话，我们称之为残差块
"""
class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)  #卷积图像大小并没有改变
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)  #这个卷积层图像大小是一定不会变的
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)#这样的设计保证了左右两边输出的图像大小是一致的
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)  #注意BatchNorm2d是自带参数的所以不能用一个BatchNorm2d
        self.relu=nn.ReLU(inplace=True)  #inplace表示对原变量进行覆盖


    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))  #这儿很明显是先bn再relu
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X  #其实就是左右两边连接
        return F.relu(Y)

"""
Residual的两种形态
"""
#输入和输出形状一致
# blk=Residual(3,3)
# X=torch.rand(4,3,6,6)
# y=blk(X)
# print(y.shape)

#增加输出通道数，高宽减半
# blk=Residual(3,6,use_1x1conv=True,strides=2)
# X=torch.rand(4,3,6,6)
# y=blk(X)
# print(y.shape)


"""
ResNet整体定义
"""
#ResNet前两层和GoogLeNet一样
b1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                 nn.BatchNorm2d(64),nn.ReLU(),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
#ResNet在后面接了4个残差网络块，每个模块使用若干个同样输出通道数的残差块，每一个残差块将上一个模块的通道数翻倍并将高宽减半
def resnet_block(input_channels,num_channels,num_residual,first_block=False):
    blk=[]
    #这里就是说一个大的blk会包含多个残差块，第一个小的残差块会增加通道数，图像尺寸减半
    #下面的残差块就保持输入输出一样了
    #同时要注意是不是第一个大的blk块，第一个大的blk块不需要增加通道数也不需要将图像尺寸减半，因为b1已经减半了
    for i in range(num_residual):
        if i==0 and not first_block:
            blk.append(
                Residual(input_channels,num_channels,use_1x1conv=True,strides=2)
            )
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b2=nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3=nn.Sequential(*resnet_block(64,128,2))
b4=nn.Sequential(*resnet_block(128,256,2))
b5=nn.Sequential(*resnet_block(256,512,2))


"""
将上述定义块组合
"""
net=nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),
                  nn.Flatten(),nn.Linear(512,10))

"""
查看网络输出形状的变化
"""
X=torch.rand(size=(1,1,224,224))

for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)


"""
训练模型
"""
lr,num_epoches,batch_size=0.05,10,256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=96)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
d2l.train_ch6(net,train_iter,test_iter,num_epoches,lr,device=device)


