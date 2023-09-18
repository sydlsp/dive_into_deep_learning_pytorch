class Test:
    def __init__(self,n):
        self.num=n
    def add_list(self,list_a):  #在这个例子中我们证明了self参数并不会接收什么东西，在python传递参数中还是要注意参数个数对其的问题
        return list_a[0]


test=Test(3)
print (test.num)
print(test.add_list([1,2,3]))