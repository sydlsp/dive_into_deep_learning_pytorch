import torch
"""数据操作和切片实际上和python矩阵使用是一样的"""
x=torch.arange(12,dtype=torch.int).reshape(3,4)
print (x)
print(x[0])
print(x[-1])
print (x[2,3])
print(x[0:2])#这个表示访问0，1两行
print(x[0:2,:])#在这里第二个表示所有元素
x[0:2]=100
print(x)
print ("ok")