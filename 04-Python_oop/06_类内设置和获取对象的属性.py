class Car:
    def __init__(self):
        self.color = 'white'

    def run(self):
        print(f'{self} 汽车在跑')


c1 = Car()
print(c1.color)

c1.color = 'red'
print(c1.color)