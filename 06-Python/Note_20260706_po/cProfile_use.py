import cProfile

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# 分析 fibonacci(30) 的执行
cProfile.run('fibonacci(30)')

print('*' * 30 + '分割线' + '*' * 30)

def fib(n):
    if n <= 1:
        return n
    else:
        n_1, n_2 = 1, 0
        idx = 1
        while idx < n:
            n_1, n_2 = n_1 + n_2, n_1
            idx = idx + 1
        return n_1

cProfile.run('fib(30)')

# 报告各列含义：
#
# ncalls: 函数被调用的总次数。
#
# tottime: 函数自身代码的总执行时间，不包含其内部调用的子函数时间。
#
# percall: tottime 除以 ncalls，即每次调用该函数的平均耗时。
#
# cumtime: 累积时间，即该函数及其所有子函数的总执行时间。
#
# percall: cumtime 除以原始调用次数。
#
# filename:lineno(function): 函数所在的文件名、行号和名称。
#
# 从上面的报告可以清晰看到，fibonacci 函数自身耗时 (tottime) 和累积耗时 (cumtime) 几乎一样，说明性能瓶颈就在这个递归函数自身。