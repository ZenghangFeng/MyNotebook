import numpy as np


##########################################################################################
# 算法实现
##########################################################################################
class LogisticRegression:
    """
    逻辑回归分类器（二分类）
    使用批量梯度下降优化
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False):
        """
        参数:
        learning_rate : float, 学习率
        num_iterations : int, 迭代次数
        fit_intercept : bool, 是否添加偏置项（截距）
        verbose : bool, 是否打印损失
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = None  # 模型参数

    def _sigmoid(self, z):
        """Sigmoid 函数"""
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _loss(self, h, y):
        """计算平均交叉熵损失"""
        m = y.shape[0]
        # 避免 log(0)
        eps = 1e-15
        h = np.clip(h, eps, 1 - eps)
        loss = - (1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss

    def fit(self, X, y):
        """
        训练逻辑回归模型
        X : ndarray, shape (m, n) 特征矩阵
        y : ndarray, shape (m,) 或 (m,1) 标签 (0 或 1)
        """
        # 确保 y 是列向量
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        m, n = X.shape

        # 添加偏置项（截距）
        if self.fit_intercept:
            X = np.hstack([np.ones((m, 1)), X])
            n += 1

        # 初始化参数
        self.theta = np.zeros((n, 1))

        # 梯度下降
        for i in range(self.num_iterations):
            # 线性组合
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)

            # 计算损失（可选）
            if self.verbose and i % 100 == 0:
                loss = self._loss(h, y)
                print(f"Iteration {i}: loss = {loss:.6f}")

            # 梯度 (1/m) * X^T (h - y)
            gradient = (1 / m) * np.dot(X.T, (h - y))

            # 更新参数
            self.theta -= self.learning_rate * gradient

        return self

    def predict_proba(self, X):
        """
        预测概率 P(y=1|x)
        X : ndarray, shape (m, n)
        返回: 概率数组, shape (m,)
        """
        if self.theta is None:
            raise Exception("模型尚未训练，请先调用 fit 方法")

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        z = np.dot(X, self.theta)
        prob = self._sigmoid(z)
        return prob.flatten()

    def predict(self, X, threshold=0.5):
        """
        预测类别 (0 或 1)
        X : ndarray, shape (m, n)
        threshold : float, 分类阈值
        返回: 预测标签, shape (m,)
        """
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def score(self, X, y):
        """
        计算准确率
        """
        y_pred = self.predict(X)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return np.mean(y_pred.reshape(-1,1) == y)


##########################################################################################
# 算法使用
##########################################################################################
# 生成模拟数据
np.random.seed(42)
m = 200
n = 2
X = np.random.randn(m, n)
true_theta = np.array([1.5, -2.0]).reshape(-1,1)
y = (1 / (1 + np.exp(-(X @ true_theta + 0.5))) > 0.5).astype(int).flatten()

# 训练模型
model = LogisticRegression(learning_rate=0.1, num_iterations=3000, verbose=True)
model.fit(X, y)

# 预测
y_pred = model.predict(X)
accuracy = model.score(X, y)
print(f"训练准确率: {accuracy:.4f}")

# 预测概率
proba = model.predict_proba(X[:5])
print("前5个样本的概率:", proba)