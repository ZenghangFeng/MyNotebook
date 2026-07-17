import numpy as np
import math
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb


# ============================
# 第一部分：数学工具（不变）
# ============================
def norm_pdf(x):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


# ============================
# 第二部分：高斯过程代理模型（不变）
# ============================
class GaussianProcess:
    def __init__(self, length_scale=1.0, sigma_f=1.0, noise=1e-8):
        self.l = length_scale
        self.sf = sigma_f
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def kernel(self, X1, X2):
        sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.sf ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.flatten()
        K = self.kernel(X, X)
        K += self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_star):
        """
        对候选点 X_star 进行预测，返回后验均值 (mu) 和后验标准差 (sigma)
        """
        K_s = self.kernel(self.X_train, X_star)  # (n, m)
        K_ss = self.kernel(X_star, X_star)  # (m, m)

        # 后验均值: mu = K_s^T * K^{-1} * y
        mu = np.dot(K_s.T, np.dot(self.K_inv, self.y_train))  # (m,)

        # 后验方差: var = diag(K_ss) - diag(K_s^T * K^{-1} * K_s)
        temp = np.dot(self.K_inv, K_s)  # (n, m)
        var = np.diag(K_ss) - np.sum(K_s * temp, axis=0)  # (m,)
        var = np.maximum(var, 1e-10)  # 数值保护
        sigma = np.sqrt(var)  # (m,)

        return mu, sigma


# ============================
# 第三部分：采集函数（不变）
# ============================
def expected_improvement(mu, sigma, mu_best, xi=0.01):
    if sigma == 0:
        return 0.0
    imp = mu - mu_best - xi
    z = imp / sigma
    ei = imp * norm_cdf(z) + sigma * norm_pdf(z)
    return ei


# ============================
# 第四部分：修改后的目标函数（使用鸢尾花 + XGBoost）
# ============================
def objective_function(params):
    """
    超参数： [learning_rate, max_depth, subsample]
    """
    lr = params[0]
    depth = int(round(params[1]))  # 转换为整数
    subsample = params[2]

    # 加载鸢尾花数据集
    data = load_iris()
    X, y = data.data, data.target
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建XGBoost分类器
    model = xgb.XGBClassifier(
        learning_rate=lr,
        max_depth=depth,
        subsample=subsample,
        n_estimators=50,  # 固定弱学习器数量
        random_state=42,
        use_label_encoder=False,  # 避免警告
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    return acc


# ============================
# 第五部分：贝叶斯优化主循环（略微调整，支持多维边界）
# ============================
def bayesian_optimization(objective, bounds, n_init=5, n_iter=15, random_seed=42):
    np.random.seed(random_seed)

    # 初始化随机采样
    X_sample = []
    y_sample = []
    print(f"===== 第 1 阶段: 随机探索 ({n_init} 次) =====")
    for i in range(n_init):
        point = np.array([np.random.uniform(low, high) for low, high in bounds])
        score = objective(point)
        X_sample.append(point)
        y_sample.append(score)
        print(f"  评估 {i + 1}: 参数 {np.round(point, 3)} -> 准确率 {score:.4f}")

    X_sample = np.array(X_sample)
    y_sample = np.array(y_sample)

    best_idx = np.argmax(y_sample)
    best_score = y_sample[best_idx]
    best_params = X_sample[best_idx]
    print(f"\n初始最佳: 准确率 {best_score:.4f}, 参数 {np.round(best_params, 3)}")
    print("=" * 50)

    # 贝叶斯迭代
    print(f"\n===== 第 2 阶段: 贝叶斯优化迭代 ({n_iter} 轮) =====")
    for t in range(n_iter):
        gp = GaussianProcess(length_scale=0.5, sigma_f=1.0, noise=1e-8)
        gp.fit(X_sample, y_sample)

        # 生成候选点
        n_candidates = 5000
        candidates = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_candidates, len(bounds))
        )

        mu_candidates, sigma_candidates = gp.predict(candidates)
        ei_values = np.array([
            expected_improvement(mu_candidates[i], sigma_candidates[i], best_score, xi=0.01)
            for i in range(n_candidates)
        ])

        next_idx = np.argmax(ei_values)
        next_point = candidates[next_idx]
        next_ei = ei_values[next_idx]

        actual_score = objective(next_point)

        X_sample = np.vstack([X_sample, next_point.reshape(1, -1)])
        y_sample = np.append(y_sample, actual_score)

        if actual_score > best_score:
            best_score = actual_score
            best_params = next_point
            improvement = " *** 新最佳! ***"
        else:
            improvement = ""

        print(f"  第 {t + 1:2d} 轮: 参数 {np.round(next_point, 3)}, "
              f"准确率 {actual_score:.4f}, EI值 {next_ei:.4f} {improvement}")

    print("\n" + "=" * 50)
    print(f"✅ 优化完成！")
    print(
        f"最优参数: learning_rate={best_params[0]:.4f}, max_depth={int(round(best_params[1]))}, subsample={best_params[2]:.4f}")
    print(f"最高准确率: {best_score:.4f}")
    print(f"总评估次数: {len(X_sample)}")
    return best_params, best_score


# ============================
# 第六部分：运行（修改边界）
# ============================
if __name__ == "__main__":
    # 定义超参数搜索边界 [min, max]
    # 顺序: learning_rate, max_depth, subsample
    parameter_bounds = [
        [0.01, 0.5],  # learning_rate
        [1.0, 10.0],  # max_depth (连续，在函数内取整)
        [0.5, 1.0]  # subsample
    ]

    start_time = time.time()
    best_params, best_score = bayesian_optimization(
        objective=objective_function,
        bounds=parameter_bounds,
        n_init=5,  # 初始随机评估5次
        n_iter=15  # 贝叶斯迭代15次（总共20次评估）
    )
    elapsed = time.time() - start_time
    print(f"\n程序耗时: {elapsed:.2f} 秒")