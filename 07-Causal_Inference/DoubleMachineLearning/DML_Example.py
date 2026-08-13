import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML, CausalForestDML

# 参考链接：https://www.aidoczh.com/econml/_autosummary/econml.dml.CausalForestDML.html


# ==================== 1. 加载并准备数据 ====================
breast_cancer = load_breast_cancer()
df_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
df_bc['target'] = breast_cancer.target

T = df_bc['worst area']
Y = df_bc['target']
X = df_bc.drop(['worst area', 'target'], axis=1)

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
forest_dml.fit(Y=Y_train, T=T_train, X=X_train)
cate_forest = forest_dml.effect(X_test)
print(f"CausalForestDML ATE: {np.mean(cate_forest):.4f}")