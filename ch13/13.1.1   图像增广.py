import torch
import torchvision
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from d2l import torch as d2l

"""
图片打印
"""
d2l.set_figsize()
img=Image.open('./cat1.jpg')
plt.imshow(img)

# plt.figure()  #创建一个独立的绘图区域
# img_1=Image.open('./cat2.jpg')
# plt.imshow(img_1)
#
# plt.pause(0)  #一次性展示所有绘图区域的图


# trans=torchvision.transforms.RandomHorizontalFlip()
# img1=trans(img)
# plt.imshow(img1)
# plt.pause(0)


"""
定义辅助函数用来图像增广
"""
#aug是图像增广的办法，这个函数就是对一张图进行图像增广num_rows*num_cols次，并输出出来
def apply(img,aug,num_rows=2,num_cols=4,scale=1.5):
    Y=[aug(img) for _ in range(num_rows*num_cols)]
    d2l.show_images(Y,num_rows,num_cols,scale=scale)

"""
左右翻转
"""
apply(img,torchvision.transforms.RandomHorizontalFlip())

"""
上下翻转
"""
apply(img,torchvision.transforms.RandomVerticalFlip())
#plt.pause(0)  #一次性显示上面所有的图


"""
随机剪裁
"""
shape_aug=torchvision.transforms.RandomResizedCrop(size=(200,200),scale=(0.1,1),ratio=(0.5,2))
apply(img,shape_aug)


"""
随机改变图像亮度  
"""
#亮度，对比度，饱和度，色温
color_aug=torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)
apply(img,color_aug)


"""
结合多种图像增广方法
"""
augs=torchvision.transforms.Compose(
    [torchvision.transforms.RandomHorizontalFlip(),color_aug,shape_aug]
)
apply(img,augs)

plt.pause(0)




