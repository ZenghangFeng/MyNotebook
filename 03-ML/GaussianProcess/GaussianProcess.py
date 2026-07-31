import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris   # 仅用于加载数据，非算法库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ================== 高斯过程回归（手写实现）==================
class GaussianProcessRegressor:
    def __init__(self, kernel=None, noise=1e-6):
        self.kernel = kernel if kernel is not None else self._rbf_kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    @staticmethod
    def _rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
        """RBF 核函数（平方指数核）"""
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True)
        sq_dist = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sq_dist)

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).flatten()
        K = self.kernel(self.X_train, self.X_train)
        K_noise = K + self.noise * np.eye(K.shape[0])
        self.K_inv = np.linalg.inv(K_noise)   # 若需纯 Python 可替换为自定义求逆
        return self

    def predict(self, X_test, return_std=False):
        X_test = np.asarray(X_test)
        K_star = self.kernel(self.X_train, X_test)
        mean = K_star.T @ self.K_inv @ self.y_train
        K_ss = self.kernel(X_test, X_test)
        diag_var = np.diag(K_ss) - np.sum(K_star.T @ self.K_inv * K_star.T, axis=1)
        diag_var = np.maximum(diag_var, 1e-10)
        if return_std:
            return mean, np.sqrt(diag_var)
        return mean

# ================== 加载 Iris 数据并构造回归任务 ==================
# 加载数据集
iris = load_iris()
X = iris.data          # 形状 (150, 4) ：花萼长、宽，花瓣长、宽
y = iris.target        # 分类标签（此处我们不用）

# 我们选择前三个特征作为输入，预测第四个特征（花瓣宽度）
# 即 X_input = [sepal length, sepal width, petal length] ， y_target = petal width
X_input = X[:, :3]     # (150, 3)
y_target = X[:, 3]     # (150,)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_input, y_target, test_size=0.3, random_state=42
)

# 标准化特征（有助于核函数）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 目标值标准化（可选，使训练更稳定）
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# ================== 训练 GP 模型 ==================
# 自定义核参数（适当调整）
def custom_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    return GaussianProcessRegressor._rbf_kernel(X1, X2, length_scale, sigma_f)

gp = GaussianProcessRegressor(kernel=custom_kernel, noise=0.01)

# 拟合训练数据
gp.fit(X_train_scaled, y_train_scaled)

# 预测测试集（返回均值和标准差）
mean_scaled, std_scaled = gp.predict(X_test_scaled, return_std=True)

# 将预测值转换回原始尺度
mean_pred = mean_scaled * y_std + y_mean
std_pred = std_scaled * y_std

# ================== 评估与可视化 ==================
# 计算 RMSE
rmse = np.sqrt(np.mean((mean_pred - y_test) ** 2))
print(f"测试集 RMSE: {rmse:.4f}")
print(f"平均预测标准差: {np.mean(std_pred):.4f}")

# 绘制真实值 vs 预测值（带误差条）
plt.figure(figsize=(8, 6))
plt.errorbar(y_test, mean_pred, yerr=2*std_pred, fmt='o', alpha=0.6,
             ecolor='gray', capsize=3, label='预测 ± 2σ')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='理想拟合')
plt.xlabel('真实花瓣宽度')
plt.ylabel('预测花瓣宽度')
plt.title('高斯过程回归在鸢尾花数据集上的表现')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制残差分布
residuals = mean_pred - y_test
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=15, edgecolor='k', alpha=0.7)
plt.axvline(0, color='r', linestyle='--')
plt.xlabel('预测残差')
plt.ylabel('频数')
plt.title('残差直方图')
plt.grid(True)
plt.show()