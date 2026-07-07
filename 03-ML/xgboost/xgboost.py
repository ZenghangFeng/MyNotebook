import numpy as np
from sklearn.datasets import make_classification  # 仅用于生成示例数据
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class XGBoostTree:
    """XGBoost 风格的回归树（用于二分类的基学习器）"""
    def __init__(self, max_depth=6, min_child_weight=1, gamma=0, reg_lambda=1,
                 min_split_gain=0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.tree = None  # 存储树结构

    def _calc_gain(self, g_sum, h_sum, g_left, h_left, g_right, h_right):
        """计算分裂增益（含正则化）"""
        # 左节点得分
        left_score = (g_left ** 2) / (h_left + self.reg_lambda) if h_left + self.reg_lambda != 0 else 0
        right_score = (g_right ** 2) / (h_right + self.reg_lambda) if h_right + self.reg_lambda != 0 else 0
        parent_score = (g_sum ** 2) / (h_sum + self.reg_lambda) if h_sum + self.reg_lambda != 0 else 0
        gain = 0.5 * (left_score + right_score - parent_score) - self.gamma
        return gain

    def _build_tree(self, X, g, h, depth):
        """递归构建决策树"""
        n_samples = X.shape[0]
        # 停止条件：深度达到最大，或样本数过少，或梯度和太小（可忽略）
        if depth >= self.max_depth or n_samples < 2 * self.min_child_weight:
            weight = - np.sum(g) / (np.sum(h) + self.reg_lambda)
            return {"leaf": True, "weight": weight}

        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        # 遍历所有特征和可能的分裂点（这里采用所有样本值作为候选）
        for feature_idx in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature_idx])
            sorted_X = X[sorted_indices, feature_idx]
            sorted_g = g[sorted_indices]
            sorted_h = h[sorted_indices]

            # 计算累积和
            g_cum = np.cumsum(sorted_g)
            h_cum = np.cumsum(sorted_h)
            total_g = g_cum[-1]
            total_h = h_cum[-1]

            # 尝试每个可能的分裂点（跳过重复值）
            for i in range(0, n_samples - 1):
                if sorted_X[i] == sorted_X[i+1]:
                    continue
                g_left = g_cum[i]
                h_left = h_cum[i]
                g_right = total_g - g_left
                h_right = total_h - h_left

                # 检查左右子节点样本权重是否满足最小约束
                if h_left < self.min_child_weight or h_right < self.min_child_weight:
                    continue

                gain = self._calc_gain(total_g, total_h, g_left, h_left, g_right, h_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = (sorted_X[i] + sorted_X[i+1]) / 2.0
                    best_left_idx = sorted_indices[:i+1]
                    best_right_idx = sorted_indices[i+1:]

        # 如果最佳增益小于阈值，则停止分裂
        if best_gain < self.min_split_gain or best_feature is None:
            weight = - np.sum(g) / (np.sum(h) + self.reg_lambda)
            return {"leaf": True, "weight": weight}

        # 递归构建左右子树
        left_tree = self._build_tree(X[best_left_idx], g[best_left_idx], h[best_left_idx], depth+1)
        right_tree = self._build_tree(X[best_right_idx], g[best_right_idx], h[best_right_idx], depth+1)

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
            "gain": best_gain
        }

    def fit(self, X, g, h):
        """根据给定的梯度和海森矩阵训练树"""
        self.tree = self._build_tree(X, g, h, 0)
        return self

    def _predict_sample(self, node, x):
        """递归预测单个样本"""
        if node["leaf"]:
            return node["weight"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(node["left"], x)
        else:
            return self._predict_sample(node["right"], x)

    def predict(self, X):
        """对数据集预测，返回每个样本的树输出（累加之前的基学习器）"""
        return np.array([self._predict_sample(self.tree, x) for x in X])


class XGBoostClassifier:
    """二分类 XGBoost 分类器"""
    def __init__(self, n_estimators=100, learning_rate=0.3, max_depth=6,
                 min_child_weight=1, gamma=0, reg_lambda=1, min_split_gain=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.trees = []
        self.base_score = 0.0  # 初始预测值（可设为 log(mean(y)/(1-mean(y)))）

    def _sigmoid(self, z):
        """sigmoid函数"""
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        训练模型
        X: 特征矩阵 (n_samples, n_features)
        y: 标签 (0/1)
        """
        y = np.array(y).astype(float)
        self.base_score = np.log(np.mean(y) / (1 - np.mean(y) + 1e-8))  # 初始偏移
        pred = np.full(X.shape[0], self.base_score)  # 当前预测值（log-odds）
        self.trees = []

        for _ in range(self.n_estimators):
            # 计算概率
            prob = self._sigmoid(pred)
            # 一阶梯度 (负梯度)
            g = prob - y
            # 二阶海森 (近似)
            h = prob * (1 - prob)
            # 构建树，拟合负梯度（实际使用g和h）
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                min_split_gain=self.min_split_gain
            )
            tree.fit(X, g, h)
            # 预测当前树的输出
            tree_pred = tree.predict(X)
            # 更新预测值（学习率衰减）
            pred += self.learning_rate * tree_pred
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        """预测概率 P(y=1)"""
        pred = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        prob = self._sigmoid(pred)
        return prob

    def predict(self, X, threshold=0.5):
        """预测类别 (0/1)"""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 生成二分类模拟数据
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 初始化并训练 XGBoost 分类器
    model = XGBoostClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        gamma=0.1,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # 评估
    acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {acc:.4f}")

    # 打印前5个样本的概率预测
    print("前5个样本的预测概率:", proba[:5])