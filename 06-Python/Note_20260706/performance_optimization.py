import math
import time


start = time.perf_counter()
"""
# 全局变量 =========================================================
size = 10000
for x in range(size):
    for y in range(size):
        z = math.sqrt(x) + math.sqrt(y)
# 执行耗时: 14.1147 秒
"""
# 局部变量 =========================================================
def main():  # 定义到函数中，以减少全部变量使用
    size = 10000
    for x in range(size):
        for y in range(size):
            z = math.sqrt(x) + math.sqrt(y)

main()
# 执行耗时: 8.8293 秒
# 结论：定义在全局范围内的代码运行速度会比定义在函数中的慢不少

# 局部变量 (LOAD_FAST)：基于数组的快速索引
# 当一个函数被定义时，其内部的局部变量个数就已经固定。因此，解释器会为这些变量分配一个固定大小的 C 语言数组来存储。
# 访问局部变量时，解释器执行 LOAD_FAST 指令，只需通过数组下标就能在 O(1) 时间内直接获取值。这 essentially 就是一次快速的指针运算。
#
# 全局变量 (LOAD_GLOBAL)：基于字典的哈希查找
# 全局变量存储在一个字典 (dict) 对象中。由于模块的全局命名空间可以在程序运行时动态地增删改变量，因此必须用更灵活的字典来存储。
# 访问全局变量时，解释器执行 LOAD_GLOBAL 指令，需要对变量名进行哈希计算，然后在字典中查找。这是一个相对复杂得多的过程。

end = time.perf_counter()
print(f"执行耗时: {end - start:.4f} 秒")

