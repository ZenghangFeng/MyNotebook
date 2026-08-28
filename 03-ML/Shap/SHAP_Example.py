import base64
import time
from io import BytesIO
import numpy as np
import xgboost
import graphviz
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML, CausalForestDML
from xgboost import XGBRegressor, XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real


# 绘图的中文显示设置 ===================================================================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块


# ==========================================================================================
# ==================== 1. 加载并准备数据 ====================
# ==========================================================================================
breast_cancer = load_breast_cancer()
df_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
df_bc['target'] = breast_cancer.target

y = df_bc['target']
x = df_bc.drop(['target'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0, shuffle=True)

model_path = f"./xgb_clf" + ".json"
"""
# ==========================================================================================
# 定义、训练、评估分类模型
# ==========================================================================================
xgb_reg_model = XGBClassifier(random_state=42)  # 增加随机种子保证基模型稳定
# 定义贝叶斯优化的超参数搜索空间 ----------------------------
# 注意：将离散列表转为连续/整数区间，有利于贝叶斯代理模型学习
search_spaces = {
            'n_estimators':Integer(100, 150),
            'max_depth': Integer(3, 8),               # 原 [3,4,5,6,7,8]
            'min_child_weight': Integer(1, 5),        # 原 [1,3,5]，现允许探索2,4等中间值
            'gamma': Real(0, 0.2),                    # 原 [0, 0.1, 0.2]
            'subsample': Real(0.6, 1.0),              # 原 [0.6, 0.8, 1.0]
            'colsample_bytree': Real(0.6, 1.0),       # 原 [0.6, 0.8, 1.0]
            'reg_alpha': Real(0, 1),                  # 原 [0, 0.1, 1]
            'reg_lambda': Real(0, 1)                  # 原 [0, 0.1, 1]
        }
# 使用贝叶斯优化搜索进行调参 --------------------------------
# n_iter=50：进行50轮贝叶斯迭代（一般50~100轮即可收敛，远少于网格搜索的数千种组合）
bayes_search = BayesSearchCV(
            estimator=xgb_reg_model,
            search_spaces=search_spaces,
            n_iter=100,  # 贝叶斯优化迭代次数
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            random_state=0  # 控制采样的随机性
        )
start = time.perf_counter()
print('-------------------开始训练回归模型（贝叶斯优化调参）---------------------')
bayes_search.fit(x_train, y_train)
end = time.perf_counter()
print(f"执行耗时: {end - start:.4f} 秒")
# 输出最佳参数与最佳交叉验证 ---------------------------------
print("Best parameters found: ", bayes_search.best_params_)
print("Best cross-validation AUC score: ", bayes_search.best_score_)

# 模型评估 ===============================================
best_model = bayes_search.best_estimator_
best_model.fit(x_train, y_train)
y_test_pre = best_model.predict_proba(x_test)
auc_score = roc_auc_score(y_test, y_test_pre[:,1])
print("Test set AUC score: ", auc_score)

# 保存模型推荐保存为json格式（原逻辑）
best_model.save_model(model_path)
print("------------------当前使用的特征个数是 {}--------------".format(x_train.shape[1]))
"""

# ==========================================================================================
# SHAP分析
# ==========================================================================================
# 计算SHAP值，并绘图 ================================================================
model = XGBClassifier()
model.load_model(model_path)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)
# xgboost.plot_tree(model, tree_idx=0)  # num_trees=0 表示绘制第一棵树
# plt.show()


fontsize = 12
labelpad = 10
shap.summary_plot(shap_values, x, plot_type="bar", max_display=10, show=False)
plt.xlabel("SHAP Value", fontsize=fontsize, labelpad=labelpad)
pic_name = 'SHAP贡献度排序条形图'
plt.title(label=pic_name, pad=5, fontsize=fontsize)
plt.show()

shap.summary_plot(shap_values, x, max_display=10, show=False)
plt.xlabel("MEAN(|SHAP Value|)", fontsize=fontsize, labelpad=labelpad)
pic_name = 'SHAP相关性小提琴图'
plt.title(label=pic_name, pad=5, fontsize=fontsize)
plt.show()

# col = 'mean concave points'
col = 'worst texture'
pic_name = f'{col} 的SHAP依赖图'
shap.dependence_plot(col, shap_values, x, interaction_index=None, show=False)
plt.xlabel(xlabel=col, fontsize=fontsize)
plt.ylabel(ylabel="SHAP Value for " + col, fontsize=fontsize)
plt.title(label=pic_name, pad=5, fontsize=fontsize)
plt.show()

"""
fig = plt.gcf()
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# 保存并关闭图形
buf = BytesIO()
plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
plt.close(fig)  # 关闭正确的图形对象
buf.seek(0)

img_binary = buf.read()
img_base64 = base64.b64encode(img_binary).decode('utf-8')
"""