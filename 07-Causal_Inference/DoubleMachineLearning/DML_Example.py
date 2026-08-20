import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML, CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter

# 参考链接：https://www.aidoczh.com/econml/_autosummary/econml.dml.CausalForestDML.html


# ---------- 1. 加载并准备数据 ----------
breast_cancer = load_breast_cancer()
df_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
df_bc['target'] = breast_cancer.target

# t = 'mean concave points'
t = 'worst texture'
T = df_bc[t]
Y = df_bc['target']
X = df_bc.drop([t, 'target'], axis=1)

X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.3, random_state=42
)

# ---------- 2. LinearDML ----------
linear_dml = LinearDML(
    model_y=RandomForestClassifier(n_estimators=100, max_depth=5),
    model_t=RandomForestRegressor(n_estimators=100, max_depth=5),
    discrete_outcome=True,
    cv=5,
    random_state=42
)
linear_dml.fit(Y=Y_train, T=T_train, X=X_train)
cate_linear = linear_dml.effect(X_test)
print(f"LinearDML ATE: {np.mean(cate_linear):.4f}")

# ---------- 3. CausalForestDML ----------
forest_dml = CausalForestDML(
    model_y=RandomForestClassifier(n_estimators=100, max_depth=5),
    model_t=RandomForestRegressor(n_estimators=100, max_depth=5),
    discrete_outcome=True,
    n_estimators=500,
    min_samples_leaf=20,
    random_state=42
)
forest_dml.fit(Y=Y_train, T=T_train, X=X_train, cache_values=True)

# ---------- 4. 分析输出的结果 ----------
# 1. 获取平均处理效应点估计值
average_treatment_effect = forest_dml.ate(X_test)
print(f"Average Treatment Effect (ATE): {average_treatment_effect}")

# 2. 获取包含统计推断的完整结果
ate_summary = forest_dml.ate_inference(X_test)
print(ate_summary)  # 会显示效应值、标准误、z值、p值和置信区间[reference:14]

# 3. 获取每个样本的CATE估计值
cate_estimates = forest_dml.effect(X_test)
plt.figure(figsize=(10, 6))
sns.histplot(cate_estimates, kde=True)
plt.title('Distribution of Conditional Average Treatment Effects (CATE)')
plt.xlabel('Estimated Treatment Effect')
plt.ylabel('Frequency')
plt.show()

# 4. 获取CATE的推断结果（包含标准误和置信区间）
cate_inference = forest_dml.effect_inference(X_test)
# cate_inference 包含了 point_estimate, stderr, z_test, p_value, conf_int 等属性

# 5. 获取特征重要性
importances = forest_dml.feature_importances_
feature_names = forest_dml.cate_feature_names()
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances for Treatment Effect Heterogeneity')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
plt.tight_layout()
plt.show()

# 6. 个体归因：可解释性决策树
# 1. 初始化解释器
intrp = SingleTreeCateInterpreter(include_model_uncertainty=True,
                                  max_depth=3,          # 限制树的深度，保证可读性
                                  min_samples_leaf=10)
# 2. 基于模型和特征进行解释
intrp.interpret(forest_dml, X_test)
# 3. 绘制决策树
plt.figure(figsize=(25, 10))
intrp.plot(feature_names=feature_names, fontsize=12)
plt.show()
#