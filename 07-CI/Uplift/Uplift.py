import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 兼容不同 NumPy 版本的 trapz / trapezoid
try:
    # NumPy 2.0+ 推荐
    from numpy import trapezoid as trapz
except ImportError:
    # 旧版本回退
    from numpy import trapz


# ------------------- 1. 生成模拟数据 -------------------
def generate_uplift_data(n_samples=10000, seed=42):
    np.random.seed(seed)

    X = np.random.randn(n_samples, 2)
    X1, X2 = X[:, 0], X[:, 1]

    true_uplift = (0.3 * (X1 > 0.5).astype(float) +
                   0.5 * (X2 < -0.5).astype(float) -
                   0.2 * ((X1 + X2) > 1).astype(float))

    p0 = 0.2 + 0.3 / (1 + np.exp(-X1))
    p0 = np.clip(p0, 0.05, 0.95)

    p1 = np.clip(p0 + true_uplift, 0.0, 1.0)

    T = np.random.binomial(1, 0.5, size=n_samples)
    prob = np.where(T == 1, p1, p0)
    Y = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'T': T,
        'Y': Y,
        'true_uplift': true_uplift
    })
    return df


# ------------------- 2. 生成数据并拆分 -------------------
df = generate_uplift_data(n_samples=10000, seed=42)

X = df[['X1', 'X2']].values
T = df['T'].values
Y = df['Y'].values
true_uplift = df['true_uplift'].values

X_train, X_test, T_train, T_test, Y_train, Y_test, true_u_train, true_u_test = train_test_split(
    X, T, Y, true_uplift, test_size=0.3, random_state=42
)

# ------------------- 3. 训练 T‑Learner -------------------
X_train_t1 = X_train[T_train == 1]
Y_train_t1 = Y_train[T_train == 1]

X_train_t0 = X_train[T_train == 0]
Y_train_t0 = Y_train[T_train == 0]

model_t1 = LogisticRegression(C=0.1, solver='liblinear', random_state=42)
model_t1.fit(X_train_t1, Y_train_t1)

model_t0 = LogisticRegression(C=0.1, solver='liblinear', random_state=42)
model_t0.fit(X_train_t0, Y_train_t0)

# ------------------- 4. 对测试集预测 uplift -------------------
prob_t1 = model_t1.predict_proba(X_test)[:, 1]
prob_t0 = model_t0.predict_proba(X_test)[:, 1]
pred_uplift = prob_t1 - prob_t0

# ------------------- 5. 评估模型（Qini 曲线 & AUUC） -------------------
order = np.argsort(pred_uplift)[::-1]
sorted_true_uplift = true_u_test[order]
cumulative_true = np.cumsum(sorted_true_uplift)
total_true = np.sum(true_u_test)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(cumulative_true)), cumulative_true,
         label='T-Learner', color='blue', linewidth=2)
plt.plot(np.arange(len(cumulative_true)),
         np.linspace(0, total_true, len(cumulative_true)),
         label='Random', linestyle='--', color='red', linewidth=2)
plt.xlabel('Number of targeted customers (sorted by predicted uplift)')
plt.ylabel('Cumulative true uplift')
plt.title('Qini Curve - T-Learner Performance')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 使用兼容的 trapz 函数
auuc = trapz(cumulative_true, dx=1)
print(f"AUUC (Area Under Uplift Curve) = {auuc:.2f}")

# ------------------- 6. 业务策略：筛选 Top 20% 高 uplift 用户 -------------------
top_percent = 0.2
top_k = int(top_percent * len(pred_uplift))
top_indices = np.argsort(pred_uplift)[-top_k:]  # 取最高分
avg_true_uplift_top = np.mean(true_u_test[top_indices])
avg_true_uplift_all = np.mean(true_u_test)

print(f"\n全部用户的平均真实 uplift: {avg_true_uplift_all:.4f}")
print(f"Top {top_percent * 100:.0f}% 用户的平均真实 uplift: {avg_true_uplift_top:.4f}")
print(f"提升倍数: {avg_true_uplift_top / avg_true_uplift_all:.2f} 倍")