import math
import time

# 全局变量 =========================================================
start = time.perf_counter()
"""
size = 10000
for x in range(size):
    for y in range(size):
        z = math.sqrt(x) + math.sqrt(y)



"""
# 局部变量 =========================================================
def main():  # 定义到函数中，以减少全部变量使用
    size = 10000
    for x in range(size):
        for y in range(size):
            z = math.sqrt(x) + math.sqrt(y)

main()


end = time.perf_counter()
print(f"执行耗时: {end - start:.4f} 秒")

# 结论：定义在全局范围内的代码运行速度会比定义在函数中的慢不少