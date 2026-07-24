import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# ===== 新增：解决中文显示问题 =====
# 设置支持中文的字体（按优先级顺序）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# =================================

# ---------- 1. 生成模拟数据 ----------
np.random.seed(123)  # 保证结果可复现
n = 600  # 样本量

# 协变量（模拟观察性研究中的混杂因素）
age = np.random.normal(30, 5, n)                # 年龄
gender = np.random.binomial(1, 0.5, n)          # 性别 (0男/1女)
income = np.random.normal(50, 15, n)            # 收入（千元）
hypertension = np.random.binomial(1, 0.2, n)    # 高血压史 (0/1)

# 将协变量组合成DataFrame
X = pd.DataFrame({
    'age': age,
    'gender': gender,
    'income': income,
    'hypertension': hypertension
})

# 生成处理变量（例如：是否吸烟），取决于协变量，存在选择性偏差
logit_prob = -2.0 + 0.03 * age + 0.5 * gender + 0.01 * income + 1.2 * hypertension
prob_treat = 1 / (1 + np.exp(-logit_prob))
treatment = np.random.binomial(1, prob_treat)   # 1=处理组，0=对照组

# 生成结局变量（例如：出生体重，单位kg），处理效应为 -0.5（吸烟导致体重降低）
true_effect = -0.5
outcome = 3.5 + true_effect * treatment + 0.02 * age - 0.1 * gender + 0.005 * income - 0.3 * hypertension + np.random.normal(0, 0.3, n)

# 合并所有数据
data = pd.DataFrame({
    'treatment': treatment,
    'outcome': outcome,
    'age': age,
    'gender': gender,
    'income': income,
    'hypertension': hypertension
})

print("数据概览：")
print(data.head())
print(f"处理组人数：{sum(treatment)}，对照组人数：{n - sum(treatment)}\n")

# ---------- 2. 估计倾向值评分（Logistic回归） ----------
covariates = ['age', 'gender', 'income', 'hypertension']
X_ps = data[covariates]
X_ps = sm.add_constant(X_ps)  # 添加截距项
logit_model = sm.Logit(data['treatment'], X_ps)
result = logit_model.fit(disp=0)  # disp=0 不输出迭代过程
data['propensity_score'] = result.predict(X_ps)

print("倾向值评分估计完成。评分范围：{:.4f} ~ {:.4f}".format(
    data['propensity_score'].min(), data['propensity_score'].max()
))

# ---------- 3. 匹配（1:1 最近邻无放回匹配，卡钳=0.02） ----------
treated = data[data['treatment'] == 1].sort_values('propensity_score').reset_index(drop=True)
control = data[data['treatment'] == 0].sort_values('propensity_score').reset_index(drop=True)

# 初始化匹配结果
matched_treated = []
matched_control = []
used_control = set()
caliper = 0.02

for i, t_row in treated.iterrows():
    # 在对照组中寻找未使用的个体，计算倾向值差
    best_idx = None
    best_dist = np.inf
    for c_idx, c_row in control.iterrows():
        if c_idx in used_control:
            continue
        dist = abs(t_row['propensity_score'] - c_row['propensity_score'])
        if dist < caliper and dist < best_dist:
            best_dist = dist
            best_idx = c_idx
    if best_idx is not None:
        matched_treated.append(i)
        matched_control.append(best_idx)
        used_control.add(best_idx)

# 构建匹配后的数据集
matched_treated_df = treated.loc[matched_treated].copy()
matched_control_df = control.loc[matched_control].copy()
matched_data = pd.concat([matched_treated_df, matched_control_df], ignore_index=True)

print(f"匹配成功样本对：{len(matched_treated)}")

# ---------- 4. 均衡性检验（标准化差异） ----------
def std_diff(df_treated, df_control, vars):
    """计算各协变量的标准化差异"""
    diffs = {}
    for var in vars:
        mean_t = df_treated[var].mean()
        mean_c = df_control[var].mean()
        var_t = df_treated[var].var()
        var_c = df_control[var].var()
        pooled_std = np.sqrt((var_t + var_c) / 2)
        if pooled_std == 0:
            diffs[var] = 0
        else:
            diffs[var] = (mean_t - mean_c) / pooled_std
    return diffs

# 匹配前处理组与对照组（按原始数据分组）
pre_treated = data[data['treatment'] == 1]
pre_control = data[data['treatment'] == 0]
pre_diff = std_diff(pre_treated, pre_control, covariates)

# 匹配后
post_treated = matched_data[matched_data['treatment'] == 1]
post_control = matched_data[matched_data['treatment'] == 0]
post_diff = std_diff(post_treated, post_control, covariates)

# 打印标准化差异
print("\n协变量标准化差异（绝对值<0.1表示均衡良好）：")
print("变量     匹配前    匹配后")
for var in covariates:
    print(f"{var:10s} {pre_diff[var]:8.3f}  {post_diff[var]:8.3f}")

# 可视化
fig, ax = plt.subplots(figsize=(8, 4))
x_pos = np.arange(len(covariates))
ax.barh(x_pos - 0.2, [pre_diff[v] for v in covariates], height=0.4, label='匹配前')
ax.barh(x_pos + 0.2, [post_diff[v] for v in covariates], height=0.4, label='匹配后')
ax.axvline(x=0.1, color='red', linestyle='--', label='阈值±0.1')
ax.axvline(x=-0.1, color='red', linestyle='--')
ax.set_yticks(x_pos)
ax.set_yticklabels(covariates)
ax.set_xlabel('标准化差异')
ax.set_title('倾向值匹配前后的协变量均衡性')
ax.legend()
plt.tight_layout()
plt.show()

# ---------- 5. 估计处理效应 ----------
# 匹配前原始差异（未经调整）
crude_effect = pre_treated['outcome'].mean() - pre_control['outcome'].mean()
# 匹配后的处理效应（匹配样本）
matched_effect = post_treated['outcome'].mean() - post_control['outcome'].mean()

print("\n处理效应估计：")
print(f"匹配前平均处理效应（ATE）：{crude_effect:.4f}（存在混杂偏倚）")
print(f"匹配后平均处理效应（ATE）：{matched_effect:.4f}（更接近真实效应 {true_effect}）")