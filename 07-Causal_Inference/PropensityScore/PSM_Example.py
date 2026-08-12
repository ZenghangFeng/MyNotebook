import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsRegressor


df = pd.read_csv('data/management_training.csv')

# 单变量线性回归
linear_model = smf.ols("engagement_score ~ intervention", data=df).fit()
print(linear_model.summary())
print('-'*40 + '分割线' + '-'*40)

# 多变量线性回归
model = smf.ols("""engagement_score ~ intervention 
        + tenure + last_engagement_score + department_score
        + n_of_reports + C(gender) + C(role)""", data=df).fit()
print("ATE:", model.params["intervention"])
print("95% CI:", model.conf_int().loc["intervention", :].values.T)
print('-'*40 + '分割线' + '-'*40)


# ==========================================================================================
# 倾向评分
# ==========================================================================================
ps_model = smf.logit("""intervention ~ 
        tenure + last_engagement_score + department_score
        + C(n_of_reports) + C(gender) + C(role)""", data=df).fit(disp=0)

data_ps = df.assign(
    propensity_score=ps_model.predict(df),
)
# data_ps[["intervention", "engagement_score", "propensity_score"]].head()


# ==========================================================================================
# 倾向评分匹配
# ==========================================================================================
T = "intervention"
X = "propensity_score"
Y = "engagement_score"

treated = data_ps.query(f"{T}==1")
untreated = data_ps.query(f"{T}==0")

mt0 = KNeighborsRegressor(n_neighbors=1).fit(untreated[[X]],untreated[Y])
mt1 = KNeighborsRegressor(n_neighbors=1).fit(treated[[X]], treated[Y])

predicted = pd.concat([
    # find matches for the treated looking at the untreated knn model
    treated.assign(match=mt0.predict(treated[[X]])),
    # find matches for the untreated looking at the treated knn model
    untreated.assign(match=mt1.predict(untreated[[X]]))
])

# predicted.head()


# ==========================================================================================
# 计算平均处理效应
# ==========================================================================================
ATE = np.mean((predicted[Y] - predicted["match"]) * predicted[T]
               + (predicted["match"] - predicted[Y]) * (1 - predicted[T]))
print("ATE:", ATE)
