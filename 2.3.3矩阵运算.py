import torch
A=torch.arange(25).reshape(-1,5)#在这里我们用到了之前所学习到的 python会自动根据25 5 来计算-1代表什么
print(A)
"""矩阵的转置运算"""
B=A.T
print (B)

"""矩阵的加法"""
C=A+B
print(C)

"""矩阵的按元素相乘 哈达玛积  在代码写法上写的是乘号但是要与真正的矩阵乘法相区分开"""
D=A*B
print (D)

"""矩阵与一个标量相加或者相乘 不改变矩阵的形状 每个元素都会与标量相加相乘"""

N=torch.arange(10)
n=2
print(N)
print(N+n)
print(N*n)
"""需要注意的是在python中有广播机制 在使用的时候要注意"""