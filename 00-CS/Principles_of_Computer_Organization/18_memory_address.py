"""
Python 中的内存地址查看 (runoob 演示)
虽然 Python 屏蔽了底层指针，但 id() 函数仍能让我们一窥地址的概念
"""
import sys

# 基础类型的内存地址
a = 42
b = 42
c = 999
d = 999

print("RUNOOB 内存地址探究: Python id() 演示")
print("=" * 60)
print()

# Python 对小整数（-5~256）做了缓存，相同值的变量指向同一地址
print("小整数缓存现象：")
print(f"  a = 42, id(a) = {id(a)}  (十六进制: 0x{id(a):X})")
print(f"  b = 42, id(b) = {id(b)}  (十六进制: 0x{id(b):X})")
print(f"  a 和 b 指向同一对象: {a is b}")
print()

print("大整数不会缓存：")
print(f"  c = 999, id(c) = {id(c)}  (十六进制: 0x{id(c):X})")
print(f"  d = 999, id(d) = {id(d)}  (十六进制: 0x{id(d):X})")
print(f"  c 和 d 指向同一对象: {c is d}  （通常为 False，取决于 Python 实现）")
print()

# 字符串的地址
text = "Hello RUNOOB"
print(f"字符串 'Hello RUNOOB' 的 id: 0x{id(text):X}")

# 列表的地址及元素地址
lst = [10, 20, 30]
print(f"\n列表 lst = {lst}")
print(f"  列表本身的 id: 0x{id(lst):X}")
print(f"  lst[0] 的 id:  0x{id(lst[0]):X}")
print(f"  lst[1] 的 id:  0x{id(lst[1]):X}")
print(f"  lst[2] 的 id:  0x{id(lst[2]):X}")

print()
print("说明：")
print("  虽然 Python 不暴露物理内存地址，但 id() 返回的是对象的唯一标识符。")
print("  在 CPython 实现中，id() 返回的就是对象在虚拟内存中的地址。")
print("  你可以看到列表元素是分散存储的（不像 C 数组那样连续）。")
print()

# 32 位 vs 64 位的体现
print("=" * 60)
print("Python 解释器的位数信息：")
print(f"  sys.maxsize = {sys.maxsize}")
print(f"  2^31 - 1     = {2**31 - 1}")
print(f"  2^63 - 1     = {2**63 - 1}")
print()
if sys.maxsize > 2**32:
    print(f"  你的 Python 是 64 位的（maxsize > 2^32）")
else:
    print(f"  你的 Python 是 32 位的（maxsize <= 2^32）")