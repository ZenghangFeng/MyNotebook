import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool

# 1. 生成模拟数据
np.random.seed(42)
n_samples = 1000
numeric_feature_1 = np.random.normal(0, 1, n_samples)
numeric_feature_2 = np.random.uniform(-5, 5, n_samples)
categorical_feature = np.random.choice(['A', 'B', 'C'], n_samples)
target = (2 * numeric_feature_1 - 0.5 * numeric_feature_2 +
          1.5 * (categorical_feature == 'B') - 0.8 * (categorical_feature == 'C') +
          np.random.normal(0, 0.5, n_samples))

data = pd.DataFrame({
    'numeric_1': numeric_feature_1,
    'numeric_2': numeric_feature_2,
    'cat_feature': categorical_feature,
    'target': target
})

# 2. 划分训练集和测试集
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 指定类别特征
cat_features = ['cat_feature']

# 4. 训练模型
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    cat_features=cat_features,
    verbose=100,
    random_seed=42
)
model.fit(X_train, y_train)

# 5. 模型评估
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n测试集 RMSE: {rmse:.4f}")
print(f"测试集 R²: {r2:.4f}")

# 6. 特征重要性
feature_importance = model.get_feature_importance()
print("\n特征重要性：")
for name, importance in zip(X_train.columns, feature_importance):
    print(f"{name}: {importance:.4f}")