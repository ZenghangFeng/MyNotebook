def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        raise ValueError(f"除数 b={b} 不能为 0") from e
    except TypeError as e:
        raise TypeError(f"参数类型错误: a={type(a)}, b={type(b)}") from e

try:
    divide(10, 0)
except ValueError as e:
    print(f"业务错误: {e}")