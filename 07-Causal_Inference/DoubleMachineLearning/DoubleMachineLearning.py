import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV


# ==========================================================================================
# 算法核心实现代码
# ==========================================================================================
def manual_dml_crossfit(X, D, Y,
                        model_y,
                        model_t,
                        n_folds=5,
                        random_state=42):
    """
    手动实现 Double Machine Learning (部分线性模型 + 交叉拟合)

    参数:
    - X: (n_samples, n_features) 高维协变量
    - D: (n_samples,) 处理变量 (Treatment)
    - Y: (n_samples,) 结果变量 (Outcome)
    - model_y: 用于拟合 Y ~ X 的 ML 模型（实例）,sklearn风格的回归器实例 (如 RandomForestRegressor())
    - model_t: 用于拟合 D ~ X 的 ML 模型（实例）,sklearn风格的回归器实例 (如 LassoCV())
    - n_folds: 交叉拟合折数
    - random_state: 随机种子

    返回:
    - theta: 因果效应估计值
    - se: 标准误
    - p_value: 显著性 p 值
    - ci_low, ci_high: 95% 置信区间
    """
    n = len(Y)
    # 初始化残差数组
    residual_y = np.zeros(n)
    residual_t = np.zeros(n)

    # 交叉拟合循环
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(X):
        # 切分数据
        X_train, X_val = X[train_idx], X[val_idx]
        D_train, D_val = D[train_idx], D[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # 1. 拟合 E[Y|X] 的机器学习模型
        # 使用 clone 避免污染原模型实例
        model_y_fold = clone(model_y)
        model_y_fold.fit(X_train, Y_train)
        g_hat = model_y_fold.predict(X_val)  # 预测 Y 的条件期望

        # 2. 拟合 E[D|X] 的机器学习模型
        model_t_fold = clone(model_t)
        model_t_fold.fit(X_train, D_train)
        m_hat = model_t_fold.predict(X_val)  # 预测 D 的条件期望

        # 3. 计算正交残差 (剥离混淆因素)
        residual_y[val_idx] = Y_val - g_hat
        residual_t[val_idx] = D_val - m_hat

    # 4. 用残差做最终 OLS 回归: residual_y = theta * residual_t + error
    # 最小二乘解 theta = sum(res_t * res_y) / sum(res_t^2)
    theta_hat = np.sum(residual_t * residual_y) / np.sum(residual_t ** 2)

    # 5. 计算 Neyman 正交稳健标准误 (不依赖同方差假设)
    n = len(Y)
    # 计算正交得分
    score = residual_t * (residual_y - theta_hat * residual_t)
    # 梯度 (Jacobian) 的期望
    gradient = np.mean(residual_t ** 2)
    # 标准误公式: sqrt( Var(score) / (gradient^2 * n) )
    var_score = np.var(score, ddof=1)  # 样本方差 (无偏)
    se_hat = np.sqrt(var_score / (gradient ** 2 * n))

    # 6. 计算 p 值和 95% 置信区间
    p_value = 2 * (1 - np.abs(theta_hat / se_hat))  # 近似正态检验
    ci_low = theta_hat - 1.96 * se_hat
    ci_high = theta_hat + 1.96 * se_hat

    return {
        "theta": theta_hat,
        "se": se_hat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "residual_y": residual_y,
        "residual_t": residual_t
    }


# ==========================================================================================
# 模拟数据测试
# ==========================================================================================
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# 设置随机种子
np.random.seed(123)
n_samples = 2000
n_features = 10

# 生成协变量 X (多元正态)
X = np.random.normal(0, 1, size=(n_samples, n_features))

# 复杂的非线性混淆结构 (影响 D 也影响 Y)
confounder = X[:, 0] + X[:, 1]**2 + X[:, 2] * X[:, 3]

# 生成处理变量 D (受 X 影响)
D = confounder + np.random.normal(0, 1, size=n_samples)

# 生成结果变量 Y (真实因果效应 theta = 2.0)
theta_true = 2.0
Y = theta_true * D + confounder + np.random.normal(0, 0.5, size=n_samples)

# ---------- 使用手动实现的 DML ----------
# 选择两种不同的机器学习模型 (展示灵活性)
model_y = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_t = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

results = manual_dml_crossfit(X, D, Y, model_y, model_t, n_folds=5)

# 打印结果
print("========== DML 估计结果 ==========")
print(f"真实因果效应 θ: {theta_true}")
print(f"估计因果效应 θ_hat: {results['theta']:.4f}")
print(f"稳健标准误: {results['se']:.4f}")
print(f"p 值: {results['p_value']:.4f}")
print(f"95% 置信区间: [{results['ci_low']:.4f}, {results['ci_high']:.4f}]")


# ==========================================================================================
# 对比：如果没有DML（直接线性回归）会怎样？
# ==========================================================================================
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(np.column_stack([D, X]), Y)
ols_theta = lr.coef_[0]

# 运行 OLS 的标准误 (简略)
resid_ols = Y - lr.predict(np.column_stack([D, X]))
se_ols = np.std(resid_ols) / np.std(D) / np.sqrt(len(Y))

print("\n========== 普通线性回归 (OLS) 对比 ==========")
print(f"OLS 估计的 D 系数: {ols_theta:.4f}")
print(f"OLS 粗糙标准误: {se_ols:.4f}")