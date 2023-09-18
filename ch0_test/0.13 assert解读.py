""""""
"""
在一个程序没有完善之前，与其让他在运行时崩溃，不如让他在出现错误条件的时候就崩溃
"""
def zero(s):
    a=int(s)
    assert a>0  #assert的含义是说如果a是大于0的，那么程序继续向下运行 否则抛出异常
    return a

print(zero(100))
print(zero(-1))