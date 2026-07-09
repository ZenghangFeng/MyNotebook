import math
from collections import Counter

class DecisionTreeNode:
    """CART决策树节点"""
    def __init__(self):
        self.feature_index = None   # 划分特征索引
        self.threshold = None       # 划分阈值（连续特征）
        self.left = None            # 左子树（特征值 <= threshold）
        self.right = None           # 右子树（特征值 > threshold）
        self.value = None           # 叶节点预测类别（非叶节点为None）

class CARTClassifier:
    """
    CART分类树
    参数:
        max_depth: 最大深度
        min_samples_split: 节点最小样本数（小于则停止划分）
    """
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        """计算基尼指数"""
        counter = Counter(y)
        total = len(y)
        if total == 0:
            return 0
        gini = 1.0
        for count in counter.values():
            prob = count / total
            gini -= prob ** 2
        return gini

    def _best_split(self, X, y):
        """
        寻找最优划分特征和阈值
        返回: (最佳特征索引, 最佳阈值, 左标签集, 右标签集)
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        best_left_y = None
        best_right_y = None

        n_features = len(X[0])
        for feature_idx in range(n_features):
            # 提取该特征的所有值，并与标签配对
            pairs = sorted([(sample[feature_idx], label) for sample, label in zip(X, y)])
            # 遍历可能的切分点（相邻不同值的中间值）
            for i in range(len(pairs) - 1):
                if pairs[i][0] == pairs[i+1][0]:
                    continue   # 相同值跳过
                threshold = (pairs[i][0] + pairs[i+1][0]) / 2.0
                # 划分左右标签
                left_y = [label for val, label in pairs if val <= threshold]
                right_y = [label for val, label in pairs if val > threshold]
                # 计算加权基尼
                gini_left = self._gini(left_y)
                gini_right = self._gini(right_y)
                gini_split = (len(left_y) * gini_left + len(right_y) * gini_right) / len(y)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_y = left_y
                    best_right_y = right_y
        return best_feature, best_threshold, best_left_y, best_right_y

    def _build_tree(self, X, y, depth):
        """递归构建树"""
        # 停止条件：样本纯、样本数太少、深度达到最大
        if len(set(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            leaf = DecisionTreeNode()
            leaf.value = Counter(y).most_common(1)[0][0]
            return leaf

        # 寻找最佳划分
        feature, threshold, left_y, right_y = self._best_split(X, y)
        if feature is None:   # 无法划分
            leaf = DecisionTreeNode()
            leaf.value = Counter(y).most_common(1)[0][0]
            return leaf

        # 分割数据
        left_X = [sample for sample in X if sample[feature] <= threshold]
        right_X = [sample for sample in X if sample[feature] > threshold]

        # 创建当前节点
        node = DecisionTreeNode()
        node.feature_index = feature
        node.threshold = threshold
        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)
        return node

    def fit(self, X, y):
        """训练模型"""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("训练数据不能为空")
        self.root = self._build_tree(X, y, 0)

    def _predict_one(self, node, sample):
        """预测单个样本"""
        if node.value is not None:   # 叶节点
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self._predict_one(node.left, sample)
        else:
            return self._predict_one(node.right, sample)

    def predict(self, X):
        """批量预测"""
        return [self._predict_one(self.root, sample) for sample in X]



# 加载鸢尾花数据（CSV格式，可从UCI或本地读取，此处手动构造示例）
import csv
import urllib.request

# 从网络获取鸢尾花数据（因不允许用库，直接用urllib读取）
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = urllib.request.urlopen(url)
data = response.read().decode('utf-8').splitlines()
reader = csv.reader(data)
X, y = [], []
for row in reader:
    if row:
        X.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        y.append(row[4])   # 标签为字符串

# 为简化演示，只取前两个特征（或全部）
X = [[sample[0], sample[1]] for sample in X]   # 仅用两个特征以便可视化

# 划分训练集和测试集（手动划分）
split_ratio = 0.8
n_train = int(len(X) * split_ratio)
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

# 训练CART树
clf = CARTClassifier(max_depth=4, min_samples_split=5)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = sum(1 for pred, true in zip(y_pred, y_test) if pred == true) / len(y_test)
print(f"测试集准确率: {accuracy:.4f}")

# 打印树结构（简单遍历）
def print_tree(node, depth=0):
    if node.value is not None:
        print("  " * depth + f"Leaf: {node.value}")
    else:
        print("  " * depth + f"Feature {node.feature_index} <= {node.threshold:.2f}?")
        print("  " * depth + "Left:")
        print_tree(node.left, depth+1)
        print("  " * depth + "Right:")
        print_tree(node.right, depth+1)

print("决策树结构：")
print_tree(clf.root)