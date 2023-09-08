from collections import Iterable
"""
理解迭代：在python中可以用for循环遍历的就可以称之为可迭代对象  例如列表 元组 字符串 字典 集合
"""

#下面利用刚 知晓的isinstance 函数来进行判断  instance 的中文意思是实例
a=100
print (isinstance(a,Iterable))
b=[1,2,3]
print (isinstance(b,Iterable))
c=(1,2,3,4,5)
print (isinstance(c,Iterable))

"""
但是值得注意的是，可迭代对象并不是迭代器，但是利用iter()可以将可迭代对象变为迭代器
"""
example=['a','b','c','d','e']
new_example=iter(example)
print (next(new_example))
print (next(new_example))
print ("ok")
#也可以利用for 循环对迭代器进行遍历
for i in new_example:
    print(i)

print ("yes")

#迭代器随用随生成吧，和指针有点像
for i in iter(example):
    print(i)
