#
# a=[0.0]*3  #表示生成了三个0.0 的数字
# print (a)
#
# """
# 理解python函数参数名中*args的含义:是把传入的参数转化为元组的形式
# """
# def test(*args):
#
#     return args
#
# list_0=[1,1,3,4,5]
#
# print (test(list_0))
#
# """
# 理解python中的zip:zip实际上就是一个压缩
# """
# list_1=[1,2,3,4,5]
# list_2=[7,8,9,9,9,10,11]
# print(list(zip(list_1,list_2)))
#
# """
# 理解一行很抽象的代码 a+float(b) for a,b in zip(self.data,args)
# """
#
# test_a=[1,2,3,4,5]
# tuple_0=[7,8,9,9,9,10,11]
# for x,y in zip(test_a,tuple_0):  #这行代码实际上是把zip里面的每一对元素给拿出来了
#     print (x,y,x+y)
#
# print([x+y for x,y in zip(test_a,tuple_0)] )  #这一行就是把上面的代码整合成一个列表
# print("ok")

"""
以上是解读内容，下面实际写一下实操一下 这里又学到了新的python知识  args参数
"""
class Accumulate:
    def __init__(self,n):
        self.data=[0.0]*n

    def add(self,*args):   #args表示任何多个参数，是一个元组

        self.data=[a+float(b) for a,b in zip(self.data,args)]

A=Accumulate(3)
print(A.data)

A.add(1,2,3)

print(A.data)

A.add(7,8,9)

print(A.data)