# ============================================
# 第8讲：从逻辑门到加法器 - Python 演示
# runoob 教程系列
# ============================================

def half_adder(a, b):
    """
    半加器：两个 1 位二进制数相加，不考虑进位输入。

    参数:
        a, b: 各为 0 或 1

    返回:
        (sum_bit, carry_out): 和位与进位

    电路实现:
        sum_bit = a XOR b   （异或门）
        carry_out = a AND b （与门）
    """
    sum_bit = a ^ b      # XOR：相同为 0，不同为 1
    carry_out = a & b    # AND：两者都为 1 时才进位
    return sum_bit, carry_out


def full_adder(a, b, carry_in):
    """
    全加器：两个 1 位二进制数相加，考虑进位输入。

    参数:
        a, b, carry_in: 各为 0 或 1

    返回:
        (sum_bit, carry_out): 和位与进位

    实现技巧——用两个半加器级联：
        第一步：half_adder(a, b) → (s1, c1)
        第二步：half_adder(s1, carry_in) → (sum_bit, c2)
        最终进位：carry_out = c1 OR c2
    """
    s1, c1 = half_adder(a, b)
    sum_bit, c2 = half_adder(s1, carry_in)
    carry_out = c1 | c2   # OR 门：任意一个有进位，最终就有进位
    return sum_bit, carry_out


def adder_4bit(a_bits, b_bits):
    """
    4 位行波进位加法器：4 个全加器级联。

    参数:
        a_bits: 4 位二进制列表，a_bits[0] 为最低位（LSB）
        b_bits: 4 位二进制列表，b_bits[0] 为最低位（LSB）

    返回:
        (sum_bits, final_carry): 4 位和（低位在前）与最终进位

    打印进位传播的完整过程，展示行波进位的工作方式。
    """
    carry_in = 0
    sum_bits = []

    print("=" * 55)
    print("  4 位行波进位加法器 —— 进位传播过程")
    print("=" * 55)
    # 高位在前显示，方便人阅读
    print(f"  加数 A: {list(reversed(a_bits))}  (高位在前)")
    print(f"  加数 B: {list(reversed(b_bits))}  (高位在前)")
    print()

    for i in range(4):
        s, carry_out = full_adder(a_bits[i], b_bits[i], carry_in)
        sum_bits.append(s)

        print(f"  [第 {i} 位] (最低位为第 0 位)")
        print(f"    输入: A[{i}]={a_bits[i]}, B[{i}]={b_bits[i]}, 进位入={carry_in}")
        print(f"    输出: 和位={s}, 进位出={carry_out}")

        if carry_out:
            print(f"    >> 产生进位！Carry={carry_out} 将传递给第 {i+1} 位")

        carry_in = carry_out  # 当前进位成为下一位的进位入
        print()

    print(f"  计算结果: {list(reversed(sum_bits))}  (高位在前)")
    print(f"  最终进位: {carry_in}")
    print("=" * 55)

    return sum_bits, carry_in


def decimal_to_4bit(n):
    """将 0-15 的十进制数转为 4 位二进制列表（低位在前）"""
    return [(n >> i) & 1 for i in range(4)]


def bits_to_decimal(bits):
    """将二进制位列表（低位在前）转为十进制整数"""
    return sum(bit * (2 ** i) for i, bit in enumerate(bits))


# ============================================
# 测试用例
# ============================================

print(">>> runoob 半加器测试 <<<")
for a in (0, 1):
    for b in (0, 1):
        s, c = half_adder(a, b)
        print(f"  half_adder({a}, {b}) => sum={s}, carry={c}")
print()

print(">>> runoob 全加器测试 <<<")
test_cases = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
]
for a, b, cin in test_cases:
    s, c = full_adder(a, b, cin)
    print(f"  full_adder({a}, {b}, carry_in={cin}) => sum={s}, carry={c}")
print()

print(">>> runoob 4 位加法器测试: 7 + 5 <<<")
a_dec, b_dec = 7, 5
a_bits = decimal_to_4bit(a_dec)
b_bits = decimal_to_4bit(b_dec)
result_bits, final_carry = adder_4bit(a_bits, b_bits)
result_dec = bits_to_decimal(result_bits)
print(f"  期望: {a_dec} + {b_dec} = {a_dec + b_dec}")
print(f"  结果: {result_dec} (进位={final_carry})")
print(f"  runoob 验证: {'通过!' if result_dec == a_dec + b_dec else '失败!'}")
print()

print(">>> runoob 额外测试: 10 + 6 (可能溢出) <<<")
a_bits2 = decimal_to_4bit(10)
b_bits2 = decimal_to_4bit(6)
result_bits2, carry2 = adder_4bit(a_bits2, b_bits2)
result_dec2 = bits_to_decimal(result_bits2)
full_result = result_dec2 + (carry2 * 16)
print(f"  期望: 10 + 6 = 16")
print(f"  4 位结果: {result_dec2}, 进位: {carry2}, 完整 5 位值: {full_result}")
print(f"  runoob 验证: {'通过!' if full_result == 16 else '失败!'}")