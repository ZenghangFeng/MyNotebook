"""
总结:
1.在 类外 访问类中的行为，需要通过 对象名、 的方式访间.
2.在 类内 访问类中的行为，需要通过 self. 的方式访问。
"""


class Car:

    def run(self):
        print(f'{self} 汽车在跑')

    def work(self):
        print(f'我是work函数，我的self值：{self}')
        self.run()
