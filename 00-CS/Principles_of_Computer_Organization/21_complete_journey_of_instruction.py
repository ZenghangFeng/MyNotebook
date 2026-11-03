"""
FullComputer - 完整冯·诺依曼计算机模拟器（runoob 教学演示）

本程序实现了一台完整的（简化版）冯·诺依曼计算机，包含：
- 四层存储: 硬盘 (Disk) → 内存 (Memory) → 缓存 (Cache) → 寄存器 (Register)
- DMA 控制器: 自动将程序从硬盘搬运到内存
- 三种总线: 地址总线、数据总线、控制总线
- 四阶段执行: 取指 (Fetch) → 译码 (Decode) → 执行 (Execute) → 写回 (WriteBack)
- 统计系统: 缓存命中率、指令计数、总线使用次数、执行时间

支持指令集:
  LOAD reg value   - 将立即数加载到寄存器
  STORE reg addr   - 将寄存器值存入内存地址
  ADD dst src      - dst = dst + src
  SUB dst src      - dst = dst - src
  MUL dst src      - dst = dst * src
  CMP a b          - 比较两个寄存器，设置标志位
  JMP addr         - 无条件跳转
  JZ addr          - 如果零标志为真则跳转

运行方式: python fullcomputer.py
"""

import time


# ============================
# 1. 存储系统 - 四层结构
# ============================

class Disk:
    """
    硬盘 - 最底层的持久化存储。
    速度最慢（毫秒级），容量最大，断电不丢失数据。
    """
    def __init__(self):
        self.sectors = {}          # 扇区存储，模拟硬盘扇区
        self.read_latency = 5      # 读延迟（模拟时间单位）
        self.write_latency = 8     # 写延迟

    def load_program(self, program_name):
        """加载存储在硬盘上的程序。返回 None 表示程序不存在。"""
        if program_name in self.sectors:
            print(f"  [硬盘] 找到程序 '{program_name}'，大小: {len(self.sectors[program_name])} 条指令")
            return self.sectors[program_name]
        print(f"  [硬盘] 程序 '{program_name}' 未找到！")
        return None

    def store_program(self, name, program):
        """将程序存入硬盘"""
        self.sectors[name] = program
        print(f"  [硬盘] 已存储程序 '{name}' ({len(program)} 条指令) 到扇区 0x{hash(name) & 0xFFFF:04X}")


class Memory:
    """
    主内存 - 第二层存储。
    速度中等（纳秒级），容量较大，断电丢失数据。
    CPU 只能直接访问内存中的数据（不能直接访问硬盘）。
    """
    def __init__(self, size=65536):
        self.size = size           # 内存大小（字节）
        self.cells = {}            # 内存单元，键为地址，值为数据/指令
        self.read_count = 0        # 读操作计数
        self.write_count = 0       # 写操作计数

    def read(self, addr):
        """从指定地址读取数据"""
        self.read_count += 1
        if addr not in self.cells:
            print(f"    警告: 地址 {addr} 未初始化，返回 0")
            return 0
        return self.cells[addr]

    def write(self, addr, data):
        """向指定地址写入数据"""
        self.write_count += 1
        self.cells[addr] = data

    def load_block(self, start_addr, data_block):
        """
        将一整块数据加载到内存的连续地址区域。
        用于 DMA 批量传输场景。
        """
        for i, item in enumerate(data_block):
            self.cells[start_addr + i] = item
        return len(data_block)


class Cache:
    """
    缓存 (L1 Cache 模拟) - 第三层存储。
    速度极快（与 CPU 同频），容量最小。

    使用简单的直接映射缓存模型:
    - 每个缓存行存储: (内存地址, 数据内容)
    - 缓存命中: 数据已在缓存中，直接返回
    - 缓存未命中: 需要从内存加载，替换一个缓存行
    """
    def __init__(self, num_lines=8):
        self.num_lines = num_lines
        self.lines = {}            # 缓存行: {cache_line_index: (mem_addr, data)}
        self.hit_count = 0
        self.miss_count = 0

    def read(self, mem_addr, memory):
        """
        从缓存读取。若命中则直接返回，若未命中则从内存加载。
        返回: (data, was_hit)
        """
        # 简单映射: 缓存行索引 = 内存地址 % 缓存行数
        line_index = mem_addr % self.num_lines

        if line_index in self.lines:
            cached_addr, cached_data = self.lines[line_index]
            if cached_addr == mem_addr:
                # 缓存命中
                self.hit_count += 1
                print(f"    [缓存] 命中！行 {line_index}, 地址 {mem_addr}, 数据: {cached_data}")
                return cached_data, True

        # 缓存未命中
        self.miss_count += 1
        data = memory.read(mem_addr)
        self.lines[line_index] = (mem_addr, data)
        print(f"    [缓存] 未命中，从内存地址 {mem_addr} 加载数据 {data} → 缓存行 {line_index}")
        return data, False

    @property
    def hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total * 100 if total > 0 else 0.0


class RegisterFile:
    """
    寄存器组 - 最顶层的存储。
    速度最快（一个时钟周期），容量最小。
    通用寄存器 + 特殊寄存器 (PC, IR, MAR, MDR, FLAGS)
    """
    def __init__(self):
        # 通用寄存器
        self.gpr = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        # 特殊寄存器
        self.PC = 0                # 程序计数器: 存放下一条指令的地址
        self.IR = None             # 指令寄存器: 存放当前正在执行的指令
        self.MAR = 0               # 内存地址寄存器: 存放要访问的内存地址
        self.MDR = 0               # 内存数据寄存器: 存放从内存读出/要写入的数据

        # 标志寄存器
        self.FLAGS = {'Z': False}  # 零标志: 上一条运算结果是否为 0

    def __repr__(self):
        gpr_str = ', '.join(f'{k}={v}' for k, v in self.gpr.items())
        return f"GPR[{gpr_str}] PC={self.PC} IR={self.IR} FLAGS={self.FLAGS}"


# ============================
# 2. 总线系统 - 三条总线
# ============================

class SystemBus:
    """
    系统总线 - 包含地址总线、数据总线、控制总线。
    提供统一接口供 CPU、内存、DMA 使用。
    """
    def __init__(self):
        self.address_bus = 0       # 地址总线: 当前传输的地址
        self.data_bus = 0          # 数据总线: 当前传输的数据
        self.control_bus = 'IDLE'  # 控制总线: IDLE, READ, WRITE, INTERRUPT, DMA_REQ, DMA_ACK
        self.transfer_count = 0    # 总线传输计数

    def cpu_read(self, address, memory, cache):
        """
        CPU 通过总线从内存读取数据（走缓存）。
        完整的总线交互过程:
        1. CPU 将地址放到地址总线上
        2. CPU 在控制总线上发出 READ 信号
        3. 内存（通过缓存层）返回数据到数据总线
        4. CPU 从数据总线读取数据
        5. 总线清空
        """
        self.address_bus = address
        self.control_bus = 'READ'
        self.transfer_count += 1
        print(f"  [总线] 地址总线 ← {address}, 控制总线 ← READ")

        # 内存通过缓存层响应
        data, was_hit = cache.read(address, memory)
        self.data_bus = data
        print(f"  [总线] 数据总线 ← {data} {'(缓存命中)' if was_hit else '(来自内存)'}")

        # CPU 读取数据
        result = self.data_bus

        # 总线清空
        self.address_bus = 0
        self.data_bus = 0
        self.control_bus = 'IDLE'
        print(f"  [总线] 总线清空，进入空闲状态")

        return result

    def cpu_write(self, address, data, memory):
        """
        CPU 通过总线向内存写入数据。
        """
        self.address_bus = address
        self.data_bus = data
        self.control_bus = 'WRITE'
        self.transfer_count += 1
        print(f"  [总线] 地址总线 ← {address}, 数据总线 ← {data}, 控制总线 ← WRITE")

        memory.write(address, data)
        print(f"  [总线] 数据 {data} 已写入内存地址 {address}")

        # 总线清空
        self.address_bus = 0
        self.data_bus = 0
        self.control_bus = 'IDLE'
        print(f"  [总线] 总线清空，进入空闲状态")


# ============================
# 3. DMA 控制器
# ============================

class DMAController:
    """
    DMA 控制器 - 负责在硬盘和内存之间直接传输数据。
    CPU 只需告诉 DMA: 源地址、目标地址、传输长度。
    DMA 完成传输后通过中断通知 CPU。
    """
    def __init__(self, bus):
        self.bus = bus
        self.transfer_count = 0    # DMA 传输次数计数
        self.total_bytes = 0       # 总传输字节数

    def transfer_program(self, disk, program_name, memory, load_addr=0):
        """
        通过 DMA 将程序从硬盘加载到内存。
        这是本课程第 20 讲「DMA 方式」的实际应用。
        """
        print(f"\n  {'='*50}")
        print(f"  DMA 传输开始: 硬盘 → 内存")
        print(f"  {'='*50}")
        print(f"  [DMA] 源: 硬盘扇区 (程序 '{program_name}')")
        print(f"  [DMA] 目标: 内存地址 0x{load_addr:04X} 开始")
        print(f"  [DMA] 请求总线控制权...")

        # DMA 占用总线
        self.bus.control_bus = 'DMA_REQ'
        print(f"  [DMA] 获得总线控制权 (控制总线 ← DMA_REQ)")

        program = disk.load_program(program_name)
        if program is None:
            print(f"  [DMA] 错误: 程序不存在，传输中止")
            self.bus.control_bus = 'IDLE'
            return 0

        # 逐条指令传输
        for i, instruction in enumerate(program):
            addr = load_addr + i
            memory.write(addr, instruction)
            self.bus.address_bus = addr
            self.bus.data_bus = instruction
            self.total_bytes += 1
            self.transfer_count += 1
            print(f"  [DMA] 传输 #{i}: 指令 '{instruction}' → 内存地址 {addr} (0x{addr:04X})")

        # DMA 释放总线，发送完成中断
        self.bus.control_bus = 'DMA_ACK'
        print(f"  [DMA] 传输完成！共 {len(program)} 条指令, {self.total_bytes} 次总线传输")
        print(f"  [DMA] 释放总线控制权 (控制总线 ← IDLE)")

        # 中断通知 CPU
        print(f"  [DMA] 发送中断信号通知 CPU: 程序加载完毕")
        self.bus.control_bus = 'INTERRUPT'

        # CPU 收到中断后的处理
        time.sleep(0.001)
        print(f"  [CPU 中断响应] 收到 DMA 完成中断，程序已就绪")
        self.bus.control_bus = 'IDLE'

        return len(program)

    def cache_prefetch(self, memory, cache, start_addr, count):
        """
        DMA 预取: 将内存中即将用到的数据提前加载到缓存。
        模拟现代 CPU 的硬件预取机制。
        """
        print(f"\n  [DMA 预取] 将内存地址 {start_addr}~{start_addr+count-1} 预加载到缓存...")
        for i in range(count):
            addr = start_addr + i
            data = memory.read(addr)
            line_index = addr % cache.num_lines
            cache.lines[line_index] = (addr, data)
        print(f"  [DMA 预取] 完成，{count} 条指令已写入缓存")


# ============================
# 4. ALU - 算术逻辑单元
# ============================

class ALU:
    """
    算术逻辑单元 - 执行所有算术和逻辑运算。
    对应课程第二模块的加法器、逻辑门等内容。
    """
    @staticmethod
    def add(a, b):
        print(f"    [ALU] 执行加法: {a} + {b} = {a + b}")
        return a + b

    @staticmethod
    def sub(a, b):
        print(f"    [ALU] 执行减法: {a} - {b} = {a - b}")
        return a - b

    @staticmethod
    def mul(a, b):
        print(f"    [ALU] 执行乘法: {a} * {b} = {a * b}")
        return a * b

    @staticmethod
    def cmp(a, b):
        result = (a == b)
        print(f"    [ALU] 执行比较: {a} == {b} ? {'是' if result else '否'} (Z={1 if result else 0})")
        return result


# ============================
# 5. FullComputer - 完整计算机
# ============================

class FullComputer:
    """
    完整冯·诺依曼计算机模拟器。

    组件:
    - disk: 硬盘（持久化存储）
    - memory: 主内存
    - cache: L1 缓存
    - registers: 寄存器组 (含 PC, IR, MAR, MDR, FLAGS)
    - alu: 算术逻辑单元
    - bus: 系统总线（地址 + 数据 + 控制）
    - dma: DMA 控制器

    执行流程:
    1. 加载: DMA 将程序从硬盘搬运到内存
    2. 循环取指 → 译码 → 执行 → 写回
    3. 生成完整执行报告
    """

    def __init__(self):
        print("=" * 60)
        print("   FullComputer - 冯·诺依曼计算机初始化中...")
        print("=" * 60)

        # 四层存储系统
        self.disk = Disk()
        self.memory = Memory()
        self.cache = Cache(num_lines=8)
        self.registers = RegisterFile()

        # 计算单元
        self.alu = ALU()

        # 总线系统
        self.bus = SystemBus()

        # DMA 控制器
        self.dma = DMAController(self.bus)

        # 统计信息
        self.instructions_executed = 0
        self.clock_cycles = 0

        print("  组件就绪: 硬盘, 内存(64KB), 缓存(8行L1), 4个通用寄存器")
        print("  组件就绪: ALU, 系统总线, DMA控制器")
        print()

    def store_program(self, name, program):
        """将程序存入硬盘"""
        print(f"[Step 0] 将程序 '{name}' 存入硬盘")
        self.disk.store_program(name, program)
        print()

    def load_program(self, program_name, load_addr=0):
        """
        加载程序: DMA 从硬盘搬运到内存。
        """
        print(f"[Step 1] DMA 加载程序到内存")
        size = self.dma.transfer_program(
            self.disk, program_name, self.memory, load_addr
        )
        print()

        # 预取指令到缓存
        print(f"[Step 2] 缓存预热: 预取前 {min(4, size)} 条指令")
        self.dma.cache_prefetch(self.memory, self.cache, load_addr, min(4, size))
        print()

        # 设置 PC 到程序起始地址
        self.registers.PC = load_addr
        print(f"[Step 3] 设置 PC ← {load_addr} (程序起始地址)")
        print()

    def fetch(self):
        """
        阶段 1: 取指 (Fetch)
        - 将 PC 值放到 MAR
        - 通过地址总线发送地址
        - 通过控制总线发 READ 信号
        - 从数据总线获取指令（经缓存层）
        - 指令存入 IR
        - PC 自增
        """
        print(f"--- 阶段 1: 取指 (Fetch) ---")
        self.clock_cycles += 1

        # MAR ← PC
        self.registers.MAR = self.registers.PC
        print(f"  MAR ← PC = {self.registers.PC}")

        # 通过总线读取内存（走缓存）
        instruction = self.bus.cpu_read(
            self.registers.MAR, self.memory, self.cache
        )

        # MDR ← 数据总线, IR ← MDR
        self.registers.MDR = instruction
        self.registers.IR = instruction
        print(f"  MDR ← {instruction}, IR ← MDR")

        # PC 自增
        old_pc = self.registers.PC
        self.registers.PC += 1
        print(f"  PC ← {old_pc} + 1 = {self.registers.PC}")

        return instruction

    def decode(self):
        """
        阶段 2: 译码 (Decode)
        - 控制器解析 IR 中的指令
        - 分离操作码和操作数
        - 决定执行阶段的操作
        """
        print(f"--- 阶段 2: 译码 (Decode) ---")
        self.clock_cycles += 1

        instruction = self.registers.IR
        parts = instruction.split()
        opcode = parts[0]
        operands = parts[1:] if len(parts) > 1 else []

        print(f"  指令: '{instruction}'")
        print(f"  操作码: {opcode}, 操作数: {operands}")

        return opcode, operands

    def execute(self, opcode, operands):
        """
        阶段 3: 执行 (Execute)
        - 根据操作码执行相应操作
        - ALU 参与算术/逻辑运算
        - 可能会修改寄存器或设置标志位
        """
        print(f"--- 阶段 3: 执行 (Execute) ---")
        self.clock_cycles += 1

        if opcode == 'LOAD':
            # LOAD reg value: 将立即数加载到寄存器
            reg, value = operands[0], int(operands[1])
            self.registers.gpr[reg] = value
            print(f"  操作: 加载立即数 {value} → 寄存器 {reg}")
            print(f"  ALU 不参与（直接加载）")

        elif opcode == 'STORE':
            # STORE reg addr: 将寄存器值存入内存
            reg, addr = operands[0], int(operands[1])
            value = self.registers.gpr[reg]
            print(f"  操作: 准备将寄存器 {reg} 的值 {value} 存入内存地址 {addr}")
            # 数据已准备好，写回阶段执行实际的存储操作
            return ('STORE', reg, addr, value)

        elif opcode == 'ADD':
            # ADD dst src: dst = dst + src
            dst, src = operands[0], operands[1]
            old_val = self.registers.gpr[dst]
            result = self.alu.add(old_val, self.registers.gpr[src])
            self.registers.gpr[dst] = result
            # 设置零标志
            self.registers.FLAGS['Z'] = (result == 0)
            print(f"  操作: {dst} ← {old_val} + {self.registers.gpr.get(src, src)} = {result}")
            print(f"  零标志 Z ← {self.registers.FLAGS['Z']}")

        elif opcode == 'SUB':
            # SUB dst src: dst = dst - src
            dst, src = operands[0], operands[1]
            old_val = self.registers.gpr[dst]
            result = self.alu.sub(old_val, self.registers.gpr[src])
            self.registers.gpr[dst] = result
            self.registers.FLAGS['Z'] = (result == 0)
            print(f"  操作: {dst} ← {old_val} - {self.registers.gpr.get(src, src)} = {result}")

        elif opcode == 'MUL':
            # MUL dst src: dst = dst * src
            dst, src = operands[0], operands[1]
            old_val = self.registers.gpr[dst]
            result = self.alu.mul(old_val, self.registers.gpr[src])
            self.registers.gpr[dst] = result
            self.registers.FLAGS['Z'] = (result == 0)
            print(f"  操作: {dst} ← {old_val} * {self.registers.gpr.get(src, src)} = {result}")

        elif opcode == 'CMP':
            # CMP a b: 比较两寄存器值，设置零标志
            a, b = operands[0], operands[1]
            is_equal = self.alu.cmp(
                self.registers.gpr[a],
                self.registers.gpr[b]
            )
            self.registers.FLAGS['Z'] = is_equal

        elif opcode == 'JMP':
            # JMP addr: 无条件跳转
            target = int(operands[0])
            print(f"  操作: 无条件跳转到地址 {target}")
            old_pc = self.registers.PC
            self.registers.PC = target
            print(f"  PC ← {old_pc} → {target} (跳转)")

        elif opcode == 'JZ':
            # JZ addr: 如果零标志为真则跳转
            target = int(operands[0])
            if self.registers.FLAGS['Z']:
                print(f"  操作: 零标志为真 (Z=1)，跳转到地址 {target}")
                self.registers.PC = target
            else:
                print(f"  操作: 零标志为假 (Z=0)，不跳转，顺序执行")

        elif opcode == 'HALT':
            # HALT: 停机
            print(f"  操作: 停机指令，程序结束")
            return 'HALT'

        else:
            print(f"  未知操作码: {opcode}")

        return None

    def writeback(self):
        """
        阶段 4: 写回 (WriteBack)
        - 如果有需要写回内存的操作（如 STORE），在此阶段完成
        - 通过数据总线传输数据
        """
        print(f"--- 阶段 4: 写回 (WriteBack) ---")
        self.clock_cycles += 1

        # 一般指令的结果已在执行阶段写入寄存器
        # STORE 指令需要在此阶段写内存
        print(f"  寄存器状态: {self.registers.gpr}")
        print(f"  标志寄存器: {self.registers.FLAGS}")

    def writeback_store(self, addr, value):
        """STORE 指令的写回操作——通过总线写入内存"""
        self.bus.cpu_write(addr, value, self.memory)
        print(f"  STORE 完成: {value} → 内存地址 {addr}")

    def run(self):
        """
        运行已加载的程序。
        循环执行 取指 → 译码 → 执行 → 写回 直到程序结束。
        """
        print("=" * 60)
        print("        程序开始执行")
        print("=" * 60)

        while self.registers.PC < max(self.memory.cells.keys(), default=0) + 1:
            # 检查 PC 是否在有效范围
            if self.registers.PC not in self.memory.cells:
                # PC 超出程序范围，可能因为跳转到了末尾，结束
                print(f"\n  程序计数器 PC={self.registers.PC} 超出程序范围，执行结束")
                break

            self.instructions_executed += 1

            print(f"\n{'#' * 55}")
            print(f"### 指令 #{self.instructions_executed} | PC = {self.registers.PC}")
            print(f"{'#' * 55}")

            # 取指
            instruction = self.fetch()

            # 译码
            opcode, operands = self.decode()

            # 执行
            special = self.execute(opcode, operands)

            # 处理 HALT 指令
            if special == 'HALT':
                self.clock_cycles += 1
                break

            # 写回
            if special and special[0] == 'STORE':
                _, reg, addr, value = special
                self.writeback_store(int(addr), value)
            else:
                self.writeback()

            # 安全检查: 防止无限循环
            if self.instructions_executed > 1000:
                print("\n  警告: 已执行超过 1000 条指令，强制停止 (可能死循环)")
                break

        print(f"\n{'=' * 60}")
        print(f"        程序执行完毕")
        print(f"{'=' * 60}")

    def report(self):
        """
        生成完整的执行报告。
        汇总所有统计信息: 缓存命中率、指令数、时钟周期、
        总线传输次数、内存读写次数等。
        """
        print(f"\n{'=' * 65}")
        print(f"      FullComputer 执行报告 —— RUNOOB")
        print(f"{'=' * 65}")
        print()

        # 基本信息
        print(f"  {'─' * 55}")
        print(f"  基本信息")
        print(f"  {'─' * 55}")
        print(f"  执行指令总数:        {self.instructions_executed:>6d} 条")
        print(f"  总时钟周期:          {self.clock_cycles:>6d} 个")
        if self.instructions_executed > 0:
            cpi = self.clock_cycles / self.instructions_executed
            print(f"  平均 CPI (周期/指令):  {cpi:>6.2f}")
        print()

        # 缓存性能
        print(f"  {'─' * 55}")
        print(f"  缓存性能 (L1 Cache)")
        print(f"  {'─' * 55}")
        total_access = self.cache.hit_count + self.cache.miss_count
        print(f"  缓存命中:            {self.cache.hit_count:>6d} 次")
        print(f"  缓存未命中:          {self.cache.miss_count:>6d} 次")
        print(f"  总访问次数:          {total_access:>6d} 次")
        print(f"  命中率:              {self.cache.hit_rate:>6.1f}%")
        print(f"  缓存行总数:          {self.cache.num_lines:>6d} 条")
        print()

        # 内存访问
        print(f"  {'─' * 55}")
        print(f"  内存访问统计")
        print(f"  {'─' * 55}")
        print(f"  内存读操作:          {self.memory.read_count:>6d} 次")
        print(f"  内存写操作:          {self.memory.write_count:>6d} 次")
        print(f"  内存总访问:          {self.memory.read_count + self.memory.write_count:>6d} 次")
        print(f"  内存已用地址数:      {len(self.memory.cells):>6d} 个")
        print()

        # 总线统计
        print(f"  {'─' * 55}")
        print(f"  总线统计")
        print(f"  {'─' * 55}")
        print(f"  CPU 发起的总线传输:  {self.bus.transfer_count:>6d} 次")
        print(f"  DMA 发起的总线传输:  {self.dma.transfer_count:>6d} 次")
        total_bus = self.bus.transfer_count + self.dma.transfer_count
        print(f"  总线传输总次数:      {total_bus:>6d} 次")
        print()

        # 寄存器最终状态
        print(f"  {'─' * 55}")
        print(f"  寄存器最终状态")
        print(f"  {'─' * 55}")
        print(f"  通用寄存器: {self.registers.gpr}")
        print(f"  程序计数器 PC: {self.registers.PC}")
        print(f"  指令寄存器 IR: {self.registers.IR}")
        print(f"  标志寄存器 FLAGS: {self.registers.FLAGS}")
        print()

        # 内存数据转储（仅显示有效地址的数据）
        print(f"  {'─' * 55}")
        print(f"  内存数据转储 (仅显示程序区域)")
        print(f"  {'─' * 55}")
        if self.memory.cells:
            for addr in sorted(self.memory.cells.keys()):
                val = self.memory.cells[addr]
                print(f"  地址 {addr:>4d} (0x{addr:04X}): {val}")
        else:
            print(f"  (内存为空)")
        print()

        print(f"{'=' * 65}")
        print(f"  报告结束。感谢使用 RUNOOB FullComputer 模拟器！")
        print(f"{'=' * 65}")


# ============================
# 6. 主程序 - 端到端演示
# ============================

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  一条指令的完整旅程 —— RUNOOB FullComputer 演示  ".center(52) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 创建计算机
    computer = FullComputer()

    # 定义演示程序: 计算 (10 + 7) * 2 - 3 = ?
    # 用 LOAD 加载初始值，用 ADD/MUL/SUB 完成计算
    demo_program = [
        "LOAD A 10",    # A = 10
        "LOAD B 7",     # B = 7
        "ADD A B",      # A = A + B = 17
        "LOAD C 2",     # C = 2
        "MUL A C",      # A = A * C = 34
        "LOAD D 3",     # D = 3
        "SUB A D",      # A = A - D = 31
        "STORE A 100",  # 将结果存入内存地址 100
        "LOAD B 31",    # B = 31 (预期结果)
        "CMP A B",      # 比较 A 和 B -> 应该相等
        "JZ 12",        # 如果相等(Z=1)，跳转到 HALT
        "LOAD D 0",     # (不执行) D = 0
        "HALT",         # 停机
    ]

    # 步骤 1: 将程序存入硬盘
    computer.store_program("RUNOOB_DEMO", demo_program)

    # 步骤 2: DMA 加载到内存
    computer.load_program("RUNOOB_DEMO", load_addr=0)

    # 步骤 3: 运行程序
    computer.run()

    # 步骤 4: 打印完整执行报告
    computer.report()

    print(f"\n")
    print(f"  知识回顾: 本次演示涵盖了计算机组成原理的全部六个模块:")
    print(f"  模块 1 (二进制):   指令和数据在内存中以二进制编码存储")
    print(f"  模块 2 (逻辑门):   执行阶段的 ADD/SUB/MUL 由 ALU 门电路完成")
    print(f"  模块 3 (冯诺依曼): 程序和数据共享内存, 五大组件协同工作")
    print(f"  模块 4 (CPU 原理):  PC取指→IR译码→ALU执行→写回, 完整四阶段")
    print(f"  模块 5 (存储系统):  硬盘→内存→缓存→寄存器, 四级存储层次")
    print(f"  模块 6 (总线与IO):  三总线交互, DMA程序加载, 中断通知")
    print()