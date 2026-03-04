"""
self介绍:
概述:
它是Python内置的关键字，用于表示本类当前对象的引用.
作用:
1个类是可以有多个对象的，这多个对象都可以通过工对象名，的方式访间类中的行为(函数)
函数默认有self属性，医数通过self来区分到底是哪个对系调用的该函数.
大自话:
谁调用函数，self就代表哪个对象
"""

class Car:

    def run(self):
        print(f'{self} 汽车在跑')


c1 = Car()
c1.run()
print(c1)

print('-' * 34)

c2 = Car()
c2.run()
print(c2)