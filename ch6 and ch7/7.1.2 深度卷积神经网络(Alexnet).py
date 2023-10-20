import torch
from torch import nn
from d2l import torch as d2l
"""
定义Alexnet,实际上就是看图说话了
"""
net=nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),  #size=(224-11+2*1)/4+1=54 54*54*96
    nn.MaxPool2d(kernel_size=3,stride=2),  #size=(54-3)/2+1=26  26*26*96
    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(), #size=(26-5+2*2)/1+1=26 26*26*256
    nn.MaxPool2d(kernel_size=3,stride=2), #size=(26-3)/2+1=12     12*12*256
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), #size=(12-3+2*1)/1+1=12  12*12*384
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(), #size=(12-3+2*1)/1+1=12  12*12*384
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), #size=(12-3+2*1)/1+1=12  12*12*384
    nn.MaxPool2d(kernel_size=3, stride=2),  #size=(12-3)/2+1=5 5*5*256
    nn.Flatten(), #展开
    nn.Linear(6400,4096),nn.ReLU(), # 5*5*256=6400 是这样来的
    nn.Dropout(p=0.5),
    nn.Linear(4096,4096),nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,10)
)

"""
看网络各层输出的形状
"""
test_tensor=torch.randn(1,1,224,224)
for layer in net:
    test_tensor=layer(test_tensor)
    print(layer.__class__.__name__,'output shape\t',test_tensor.shape)

batch_size=128
# 这里是把原来的28*28的图片转化为224*224的，
# 因为Alexnet的输入就是3*224*224,上面的转法是为了更准确的复现Alexnet
# 由于这里用的是fashion_minst是灰度图，所以上面定义的网络输入通道是1
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=224)

lr,num_epoches=0.01,10
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

d2l.train_ch6(net,train_iter,test_iter,num_epoches,lr,device=device)








