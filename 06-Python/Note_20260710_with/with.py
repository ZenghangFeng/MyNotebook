import time


##########################################################################################
# 基本用法
##########################################################################################
with open('test.txt', 'r') as f:
    data = f.read()
# 文件已自动关闭，无需显式调用 close()


##########################################################################################
# 通过定义类实现 __enter__ 和 __exit__ 方法
##########################################################################################
class ManagedFile:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file   # 返回值赋给 as 后的变量

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        # 返回 True 会压制异常，一般不压制，让异常正常传播
        return False

# 使用
with ManagedFile('test.txt', 'w') as f:
    f.write('Hello, world! ' + time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()))
# 文件自动关闭


##########################################################################################
# Python 标准库 contextlib 提供了更方便的方式来创建上下文管理器。
##########################################################################################
from contextlib import contextmanager

@contextmanager
def managed_file(filename, mode='r'):
    f = open(filename, mode)
    try:
        yield f          # 进入 with 块时返回 f
    finally:
        f.close()        # 退出 with 块时执行清理

with managed_file('test.txt', 'r') as f:
    data_new = f.read()