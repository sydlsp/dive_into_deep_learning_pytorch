import numpy as np
A=np.arange(0,25).reshape(5,-1)
print(A)

"""
在numpy中行和列用逗号隔开
"""
print(A[0:2,0:2])  #相当于0，1行和0，1列重叠的部分
print(A[:,:1])     #相当于所有行和第0列的重叠部分
print(A[:,1])      #相当于所有行的一号元素
print(A[1::3,:1])  #这个相当于第1行 1+3行与第一列的重叠部分