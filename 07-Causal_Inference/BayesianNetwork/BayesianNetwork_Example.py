# 参考链接：https://www.lianxh.cn/details/1699.html

import pandas as pd
import numpy as np
import bnlearn as bn
from sklearn.datasets import load_breast_cancer

# ------------------- 1. 模拟生成数据 -------------------
breast_cancer = load_breast_cancer()
df_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
df_bc['target'] = breast_cancer.target
# df = df_bc[['mean concave points', 'worst texture', 'worst concave points', 'worst radius', 'area error', 'worst concavity', 'target']]
df = df_bc[['mean concave points', 'worst texture', 'worst concave points', 'worst concavity', 'target']]

# ------------------- 2. 结构学习 -------------------
# 不同的结构学习方法会得到不同的结果，还需要分析怎样调整参数，使得target作为目标节点
model = bn.structure_learning.fit(df, methodtype='ica-lingam')
print("学习到的边：", model['model_edges'])

# ------------------- 3. 参数学习 -------------------
model = bn.parameter_learning.fit(model, df)

# ------------------- 4. 绘图 -------------------
bn.plot(model, node_color='#FFDDAA', node_size=4000, interactive=False)
print("绘图完成！")