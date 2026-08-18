import numpy as np
from sklearn.datasets import load_iris  # 仅用于获取原始数据数组，不调用任何拟合方法


# -------------------- 1. 加载与预处理数据 --------------------
def load_and_preprocess():
    iris = load_iris()
    X = iris.data  # (150, 4)
    y = iris.target  # (150,)

    # 手动标准化（Z-score），使梯度下降更稳定
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # 添加偏置列（截距项），X 变为 (150, 5)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # 手动打乱数据集并划分训练集(80%)和测试集(20%)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test


# -------------------- 2. 辅助函数：One-hot 编码 --------------------
def one_hot_encoding(y, num_classes=3):
    """将标签向量 (N,) 转换为独热矩阵 (N, K)"""
    N = y.shape[0]
    Y = np.zeros((N, num_classes))
    Y[np.arange(N), y] = 1
    return Y


# -------------------- 3. Softmax 函数（带数值稳定性） --------------------
def softmax(logits):
    """输入 (N, K)，输出同维度的概率矩阵"""
    # 减去每行的最大值，防止 exp 溢出
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


# -------------------- 4. 负对数似然（损失函数） --------------------
def negative_log_likelihood(X, Y, W):
    """计算当前参数下的平均 NLL 损失"""
    N = X.shape[0]
    logits = X @ W  # (N, K)
    probs = softmax(logits)
    # 防止 log(0) 出现无穷，加极小值 1e-15
    loss = -np.sum(Y * np.log(probs + 1e-15)) / N
    return loss


# -------------------- 5. 梯度计算（手推解析式） --------------------
def compute_gradient(X, Y, W):
    """
    损失函数对 W 的梯度:
    dL/dW = (1/N) * X^T (P - Y)
    其中 P = softmax(X @ W)
    """
    N = X.shape[0]
    probs = softmax(X @ W)
    grad = (X.T @ (probs - Y)) / N
    return grad


# -------------------- 6. MLE 训练器（批量梯度下降） --------------------
def train_mle(X_train, y_train, lr=0.1, epochs=5000, verbose=True):
    """
    极大似然估计的核心：通过梯度下降最小化负对数似然
    """
    N, D = X_train.shape
    K = len(np.unique(y_train))

    # 将标签转为独热
    Y_train = one_hot_encoding(y_train, K)

    # 初始化权重（较小随机数）
    W = np.random.randn(D, K) * 0.01

    # 记录训练过程
    losses = []

    for epoch in range(epochs):
        # 计算梯度并更新权重
        grad = compute_gradient(X_train, Y_train, W)
        W -= lr * grad

        # 每 500 轮打印一次损失
        if verbose and epoch % 500 == 0:
            loss = negative_log_likelihood(X_train, Y_train, W)
            losses.append(loss)
            print(f"Epoch {epoch:4d} | NLL Loss: {loss:.6f}")

    return W, losses


# -------------------- 7. 预测与评估 --------------------
def predict(X, W):
    """返回预测类别索引 (0, 1, 2)"""
    logits = X @ W
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# -------------------- 8. 主程序执行 --------------------
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test = load_and_preprocess()

    print("=" * 50)
    print("开始 MLE 训练（手动实现）...")
    print("=" * 50)

    # 训练模型
    W, losses = train_mle(X_train, y_train, lr=0.3, epochs=3000)

    # 预测
    y_pred_train = predict(X_train, W)
    y_pred_test = predict(X_test, W)

    train_acc = accuracy(y_train, y_pred_train)
    test_acc = accuracy(y_test, y_pred_test)

    print("\n" + "=" * 50)
    print(f"训练集准确率: {train_acc * 100:.2f}%")
    print(f"测试集准确率: {test_acc * 100:.2f}%")
    print("=" * 50)

    # 输出最后的权重（即为 MLE 估计出的参数）
    print("\nMLE 估计出的权重矩阵 (5x3):")
    print(W)