import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causal_curve import GPS_Classifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# 参考链接：https://causal-curve.readthedocs.io/en/latest/GPS_Classifier.html

# ==================== 1. 加载并准备数据 ====================
breast_cancer = load_breast_cancer()
df_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
df_bc['target'] = breast_cancer.target

# -------------------- 2. 初始化并训练 GPS_Classifier --------------------
gps = GPS_Classifier(treatment_grid_num=200, random_seed=42)

# t = 'mean concave points'
t = 'worst texture'
gps.fit(T = df_bc[t], X = df_bc[[c for c in df_bc.columns if c not in [t, 'target']]], y = df_bc['target'])
gps_results = gps.calculate_CDRC(0.95)

# -------------------- 3. 可视化 --------------------
treatment_grid = gps_results['Treatment']
point_estimate = gps_results['Causal_Odds_Ratio']
lower_bound = gps_results['Lower_CI']
upper_bound = gps_results['Upper_CI']

plt.figure(figsize=(10, 6))

plt.plot(treatment_grid,
         point_estimate,
         color='blue',
         label='Causal Effect Estimate (Probability)')

plt.fill_between(treatment_grid,
                 lower_bound,
                 upper_bound,
                 color='blue',
                 alpha=0.2,
                 label='95% Confidence Interval')

plt.xlabel('Treatment Level')
plt.ylabel('Potential Outcome Probability')
plt.title('Causal Dose-Response Curve (Binary Outcome)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

"""
# -------------------- 5. 特定处理值下的因果效应推断 --------------------
treatment_point = np.array([0.5])
point_effect = gps.point_estimate(treatment_point)
point_effect_interval = gps.point_estimate_interval(treatment_point, ci=0.95)

print(f"\n当 Treatment = 0.5 时，估计的潜在结果概率为: {point_effect[0]:.4f}")
print(f"其 95% 置信区间为: [{point_effect_interval[0][0]:.4f}, {point_effect_interval[0][1]:.4f}]")
"""