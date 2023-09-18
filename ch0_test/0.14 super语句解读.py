""""""
"""
super(子类名，self).__init__()  实际是按照父类初始化的方法来初始化子类的属性
"""


class Person:
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

    def printinfo(self):
        print(self.name, self.gender)


class Stu(Person):
    def __init__(self, name, gender, school):
        super(Stu, self).__init__(name, gender)  # 使用父类的初始化方法来初始化子类,在这里注意一下参数名的对应关系就好
        self.school = school

    def printinfo(self):  # 对父类的printinfo方法进行重写
        print(self.name, self.gender, self.school)


if __name__ == '__main__':
    stu = Stu('djk', 'man', 'nwnu')
    stu.printinfo()