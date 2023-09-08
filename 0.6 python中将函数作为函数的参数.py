import torch


add=[1,2,3,4,5]

def add(list):
    return list.sum()

A=torch.arange(0,10)

def mul(add_,num):  #在这里传入了一个参数add_ 实参对应的是add 也就是一个函数
    return add_(A)/num

print(add(A))
print (mul(add,10))