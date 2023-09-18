import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()

"""下载数据集并读到内存中"""

trans=transforms.ToTensor() #将图像数据编程浮点数的形式
minst_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
min_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
print (len(min_test))
print (len(minst_train))