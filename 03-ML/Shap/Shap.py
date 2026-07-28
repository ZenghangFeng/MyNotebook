import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
import math
import warnings

warnings.filterwarnings("ignore")


# ==============================
# 1. 数据加载（使用 GitHub CSV）
# ==============================
def load_california_housing_from_csv():
    """从 GitHub 加载加州房价数据，并构造与 fetch_california_housing 一致的 X, y"""
    url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
    df = pd.read_csv(url)

    # 构造特征（与原始数据集一致）
    X = pd.DataFrame()
    X['MedInc'] = df['median_income']
    X['HouseAge'] = df['housing_median_age']
    X['AveRooms'] = df['total_rooms'] / df['households']
    X['AveBedrms'] = df['total_bedrooms'] / df['households']
    X['Population'] = df['population']
    X['AveOccup'] = df['population'] / df['households']
    X['Latitude'] = df['latitude']
    X['Longitude'] = df['longitude']

    # 目标变量（转换为十万美元单位）
    y = df['median_house_value'] / 100000.0

    # 移除缺失值（若有）
    if X.isnull().values.any():
        clean_idx = ~X.isnull().any(axis=1)
        X = X[clean_idx]
        y = y[clean_idx]

    # 返回一个类似 Bunch 的对象以兼容原代码
    feature_names = list(X.columns)
    return Bunch(data=X.values, target=y.values, feature_names=feature_names,
                 DESCR="California Housing dataset from GitHub")


# 加载数据
data = load_california_housing_from_csv()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("数据形状：", X.shape)
print("特征名称：", X.columns.tolist())

# ==============================
# 2. 划分训练集 / 测试集
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（便于模型训练）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 3. 训练一个黑盒模型（随机森林）
# ==============================
model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

print("训练集 R²：", model.score(X_train_scaled, y_train))
print("测试集 R²：", model.score(X_test_scaled, y_test))


# ==============================
# 4. 自实现 KernelSHAP 类
# ==============================
class KernelSHAP:
    def __init__(self, model, background_data, nsamples=1000):
        """
        model: 训练好的黑盒模型（必须有 predict 方法）
        background_data: 背景数据集（ndarray），用于填充被忽略的特征
        nsamples: 生成掩码的数量（越大近似越准，计算越慢）
        """
        self.model = model
        self.background = np.array(background_data)
        self.nsamples = nsamples
        self.M = background_data.shape[1]  # 特征个数

    def _generate_masks(self):
        """生成随机二值掩码，并包含全0和全1两个特殊掩码"""
        masks = []
        # 随机生成 nsamples-2 个掩码
        for _ in range(self.nsamples - 2):
            mask = np.random.choice([0, 1], size=self.M, p=[0.5, 0.5])
            # 保证每个掩码既有0又有1，避免无效样本
            while np.sum(mask) == 0 or np.sum(mask) == self.M:
                mask = np.random.choice([0, 1], size=self.M, p=[0.5, 0.5])
            masks.append(mask)
        # 加入全0和全1掩码
        masks.append(np.zeros(self.M, dtype=int))
        masks.append(np.ones(self.M, dtype=int))
        return np.array(masks)

    def _weight(self, mask):
        """计算给定掩码的权重"""
        z = int(np.sum(mask))
        if z == 0 or z == self.M:
            return 1e6  # 给全0/全1极大权重，确保回归通过这两个点
        return (self.M - 1) / (z * (self.M - z) * math.comb(self.M, z))

    def _generate_perturbations(self, x_explain, masks):
        """根据掩码生成扰动样本"""
        n_masks = len(masks)
        perturbed = np.zeros((n_masks, self.M))
        for i, mask in enumerate(masks):
            for j in range(self.M):
                if mask[j] == 1:
                    perturbed[i, j] = x_explain[j]
                else:
                    # 从背景数据中随机抽取一个样本的该特征值
                    idx = np.random.randint(0, len(self.background))
                    perturbed[i, j] = self.background[idx, j]
        return perturbed

    def shap_values(self, x_explain):
        """
        计算单个样本 x_explain 的 SHAP 值
        返回: shap_values (array), base_value (float)
        """
        # 1. 生成掩码
        masks = self._generate_masks()
        n_masks = len(masks)

        # 2. 生成扰动样本
        X_perturb = self._generate_perturbations(x_explain, masks)

        # 3. 模型预测
        preds = self.model.predict(X_perturb)

        # 4. 构建加权线性回归
        weights = np.array([self._weight(mask) for mask in masks])

        # 设计矩阵：第一列为1（截距），其余为掩码值
        X_design = np.hstack([np.ones((n_masks, 1)), masks])

        # 加权最小二乘：beta = (X^T W X)^{-1} X^T W y
        W = np.diag(weights)
        XtWX = X_design.T @ W @ X_design
        XtWy = X_design.T @ W @ preds
        # 添加极小正则项避免奇异矩阵
        lambda_reg = 1e-6
        beta = np.linalg.inv(XtWX + lambda_reg * np.eye(XtWX.shape[0])) @ XtWy

        base_value = beta[0]
        shap_vals = beta[1:]
        return shap_vals, base_value


# ==============================
# 5. 使用自实现 KernelSHAP 解释一个样本
# ==============================
# 固定随机种子以保证结果可重复
np.random.seed(42)

# 选择测试集第一个样本
x_sample = X_test_scaled[0]

# 背景数据选取训练集的前 100 个样本（可随机抽取）
background = X_train_scaled[:100]

# 创建解释器（nsamples 可调，这里为了速度设为 300）
explainer = KernelSHAP(model, background, nsamples=300)

# 计算该样本的 SHAP 值
shap_vals, base_value = explainer.shap_values(x_sample)

# ==============================
# 6. 输出结果
# ==============================
prediction = model.predict([x_sample])[0]
print("\n" + "=" * 50)
print("样本解释结果")
print("=" * 50)
print(f"模型预测值: {prediction:.4f}")
print(f"基线值（训练集平均预测）: {base_value:.4f}")
print("\n各特征 SHAP 值（对预测的贡献）:")
for name, val in zip(X.columns, shap_vals):
    print(f"  {name:>10}: {val: .4f}")
print("\nSHAP 值之和 + 基线 =", np.sum(shap_vals) + base_value)
print("预测值 ≈ 基线 + SHAP 之和（应接近模型预测值）")