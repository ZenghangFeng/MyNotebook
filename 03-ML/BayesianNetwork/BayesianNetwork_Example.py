# 参考链接：https://www.aidoczh.com/pgmpy/base/base.html

import pandas as pd
import numpy as np
import bnlearn as bn

# ------------------- 1. 模拟生成数据 -------------------
np.random.seed(42)  # 可重现
n = 1000

Cloudy = np.random.choice([0,1], size=n, p=[0.5,0.5])
Sprinkler = np.where(Cloudy==0,
                     np.random.choice([0,1], size=n, p=[0.6,0.4]),
                     np.random.choice([0,1], size=n, p=[0.1,0.9]))
Rain = np.where(Cloudy==0,
                np.random.choice([0,1], size=n, p=[0.8,0.2]),
                np.random.choice([0,1], size=n, p=[0.2,0.8]))
Wet_Grass = np.where((Sprinkler==1) | (Rain==1), 1, 0)

df = pd.DataFrame({'Cloudy': Cloudy, 'Sprinkler': Sprinkler, 'Rain': Rain, 'Wet_Grass': Wet_Grass})

# ------------------- 2. 结构学习 -------------------
model = bn.structure_learning.fit(df)
print("学习到的边：", model['model_edges'])

# ------------------- 3. 参数学习 -------------------
model = bn.parameter_learning.fit(model, df)

# ------------------- 4. 绘图 -------------------
bn.plot(model, node_color='#FFDDAA', node_size=4000)
print("绘图完成！")