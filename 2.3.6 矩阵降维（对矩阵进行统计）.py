import torch
A=torch.arange(20,dtype=torch.float32).reshape(4,5)
print (A)
"""统计矩阵的形状 元素个数 总和"""
print(A.shape)
print(A.numel())
print(A.sum())

"""统计矩阵每一行的值或者每一列的值"""

#统计每一列的值
A_sum_axis0=A.sum(axis=0)
print(A_sum_axis0)
#统计每一行的值
A_sum_axis1=A.sum(axis=1)
print (A_sum_axis1)

"""统计矩阵的平均值  两种方式均可"""
print (A.mean())
print (A.sum()/A.numel())

"""与上面sum函数类似 mean函数也可以按照行或者列来进行计算"""

print(A.mean(axis=0))
print(A.mean(axis=1))



