import math
from sklearn.datasets import load_iris


# ---------- 辅助函数 ----------
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def standardize(X):
    n, d = len(X), len(X[0])
    means = [sum(row[i] for row in X) / n for i in range(d)]
    stds = [math.sqrt(sum((row[i] - means[i]) ** 2 for row in X) / n) for i in range(d)]
    return [[(row[i] - means[i]) / (stds[i] + 1e-9) for i in range(d)] for row in X]


# ---------- 手写层次聚类 ----------
class HierarchicalClustering:
    def __init__(self, linkage='ward'):
        self.linkage = linkage
        self.data = None
        self.n_samples = 0
        self.clusters = None  # 当前各簇包含的样本索引
        self.centroids = None
        self.sizes = None
        self.active_ids = None
        self.merge_history = []  # 存储 (c1, c2, distance, c2_samples_snapshot)

    def fit(self, X):
        self.data = X
        self.n_samples = len(X)
        dim = len(X[0])

        # 初始化
        self.clusters = [[i] for i in range(self.n_samples)]
        self.centroids = [X[i][:] for i in range(self.n_samples)]
        self.sizes = [1] * self.n_samples
        self.active_ids = list(range(self.n_samples))
        self.merge_history = []

        while len(self.active_ids) > 1:
            active = self.active_ids
            m = len(active)
            min_dist = float('inf')
            best_i = best_j = -1

            # 寻找最近的两个簇
            for i in range(m):
                for j in range(i + 1, m):
                    c1, c2 = active[i], active[j]
                    dist = self._compute_dist(c1, c2)
                    if dist < min_dist:
                        min_dist = dist
                        best_i, best_j = i, j

            c1, c2 = active[best_i], active[best_j]

            # 保存 c2 的快照（用于后续划分标签）
            c2_snapshot = self.clusters[c2][:]

            # 合并 c2 到 c1
            self.clusters[c1].extend(self.clusters[c2])
            self.clusters[c2] = []

            # 更新质心与大小
            size_c1, size_c2 = self.sizes[c1], self.sizes[c2]
            new_size = size_c1 + size_c2
            new_centroid = [
                (size_c1 * self.centroids[c1][k] + size_c2 * self.centroids[c2][k]) / new_size
                for k in range(dim)
            ]
            self.centroids[c1] = new_centroid
            self.sizes[c1] = new_size

            # 记录历史（包含快照）
            self.merge_history.append((c1, c2, min_dist, c2_snapshot))

            # 从活跃列表中移除 c2
            del active[best_j]

        return self

    def _compute_dist(self, c1, c2):
        if self.linkage == 'ward':
            # 返回欧氏距离（开根号），比较时不影响结果
            return math.sqrt(sum((self.centroids[c1][k] - self.centroids[c2][k]) ** 2
                                 for k in range(len(self.centroids[c1]))))

        points1 = self.clusters[c1]
        points2 = self.clusters[c2]

        if self.linkage == 'single':
            return min(euclidean_distance(self.data[i], self.data[j]) for i in points1 for j in points2)
        elif self.linkage == 'complete':
            return max(euclidean_distance(self.data[i], self.data[j]) for i in points1 for j in points2)
        elif self.linkage == 'average':
            total = 0.0
            cnt = 0
            for i in points1:
                for j in points2:
                    total += euclidean_distance(self.data[i], self.data[j])
                    cnt += 1
            return total / cnt if cnt else 0.0
        else:
            raise ValueError(f"不支持: {self.linkage}")

    def get_labels(self, n_clusters):
        """
        根据合并历史和快照，将样本划分为 n_clusters 个簇
        """
        if n_clusters < 1 or n_clusters > self.n_samples:
            raise ValueError("n_clusters 越界")

        # 并查集
        parent = list(range(self.n_samples))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # 应用前 (N - n_clusters) 次合并
        for idx in range(self.n_samples - n_clusters):
            c1, c2, dist, c2_samples = self.merge_history[idx]
            # 将 c2 中的所有样本，合并到 c1 的第一个样本（作为代表）上
            if self.clusters[c1]:  # 正常情况下非空
                rep = self.clusters[c1][0]  # c1的根代表样本
                for sample_idx in c2_samples:
                    union(rep, sample_idx)

        # 分配标签
        root_to_label = {}
        labels = [0] * self.n_samples
        for i in range(self.n_samples):
            root = find(i)
            if root not in root_to_label:
                root_to_label[root] = len(root_to_label)
            labels[i] = root_to_label[root]

        return labels


# ---------- 主程序：鸢尾花数据集示例 ----------
if __name__ == "__main__":
    # 1. 加载数据（仅用于获取原始数据）
    iris = load_iris()
    X_raw = iris.data.tolist()
    y_true = iris.target

    # 2. 标准化（Ward法强烈建议标准化，其他方法也可提升效果）
    X_std = standardize(X_raw)

    # 3. 训练层次聚类（使用 Ward 法）
    model = HierarchicalClustering(linkage='ward')
    model.fit(X_std)

    # 4. 获取 3 个聚类（鸢尾花已知有3类）
    n_clusters = 3
    y_pred = model.get_labels(n_clusters)

    # 5. 输出结果对比
    print("=" * 60)
    print(f"使用连接准则: {model.linkage}")
    print(f"总样本数: {len(y_pred)}，聚类数: {n_clusters}")
    print("-" * 60)
    print("前100个样本的预测标签 vs 真实标签：")
    for i in range(100):
        print(f"{y_pred[i]:2d}  vs  {y_true[i]:2d}   |", end=" ")
        if (i + 1) % 10 == 0:
            print()

    print("\n" + "-" * 60)
    print("合并历史（最后5次合并）:")
    for idx, (c1, c2, dist, _) in enumerate(model.merge_history[-5:]):
        print(f"  第{len(model.merge_history) - 5 + idx + 1}次合并: 簇{c1} 与 簇{c2}，距离 = {dist:.4f}")