import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay


############################################################
# 1. 加载数据并训练模型
############################################################
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names  # ['age', 'sex', 'bmi', 'bp', ...]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


############################################################
# 2. 创建画布
############################################################
fig, ax = plt.subplots(figsize=(12, 8))


############################################################
# 3. 绘制多种类型的 PDP
############################################################
# - 'bmi' 的单向 PDP
# - 'bp' 的单向 ICE 图（kind='individual'）
# - ('bmi', 'bp') 的双向交互 PDP
features_to_plot = ["bmi", "bp", ("bmi", "bp")]
kinds = ['average', 'individual', 'average']  # 为每个图分别指定类型

display = PartialDependenceDisplay.from_estimator(
                estimator=model,
                X=X,
                features=features_to_plot,
                feature_names=feature_names,
                kind=kinds,  # 可以传入列表为每个子图指定不同 kind
                n_cols=3,    # 网格列数
                ax=ax
            )

fig.suptitle("Partial Dependence Plots (PDP) and ICE")
plt.tight_layout()
plt.show()