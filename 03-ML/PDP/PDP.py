# -*- coding: utf-8 -*-
"""
从零实现部分依赖图（Partial Dependence Plot, PDP）
=====================================================
数据集：sklearn.datasets.load_diabetes（与上午示例一致）
被解释模型：RandomForestRegressor（作为待解释的黑盒）

本文件中 PDP 的网格构造、特征置换、样本平均全部手写实现，
不调用 sklearn.inspection.partial_dependence / PartialDependenceDisplay。
仅在最后的"对拍验证"环节调用官方实现做数值一致性检查。

依赖：numpy, matplotlib, scikit-learn
运行：python pdp_from_scratch.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 第一部分：手写 PDP / ICE 算法
# ============================================================

def pdp_univariate(model, X, feature, grid_resolution=100,
                   percentiles=(0.05, 0.95)):
    """
    单向部分依赖（PDP）。

    PD_j(x) = (1/N) * sum_i f(x, X_C^(i))

    参数
    ----
    model            : 已训练模型，需有 .predict
    X                : (N, p) 训练数据
    feature          : int，要分析的特征列索引
    grid_resolution  : 网格点个数 G
    percentiles      : 网格取值区间（按分位数裁剪，避免外推）

    返回
    ----
    grid : (G,)   网格值
    pdp  : (G,)   每个网格点上的平均预测
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]

    # 1) 构造网格
    lo, hi = np.percentile(X[:, feature], percentiles)
    grid = np.linspace(lo, hi, grid_resolution)

    # 2) 向量化置换：把所有 (g_k, x_C^(i)) 组合堆成 (G*N, p)
    #    tile 后行顺序 = [X; X; ...; X]（共 G 块）
    #    repeat(grid, N) 让第 k 块的 N 行 feature 列都等于 grid[k]
    X_rep = np.tile(X, (grid_resolution, 1))
    X_rep[:, feature] = np.repeat(grid, N)

    # 3) 一次性预测，再对每个网格块内的 N 个样本求平均
    preds = model.predict(X_rep).reshape(grid_resolution, N)
    pdp = preds.mean(axis=1)
    return grid, pdp


def ice_univariate(model, X, feature, grid_resolution=100,
                   percentiles=(0.05, 0.95), subsample=None):
    """
    个体条件期望（ICE）。与 PDP 共享同一批预测，只是不做平均。

    返回
    ----
    grid : (G,)            网格值
    ice  : (n_samples, G)  每个样本在网格上的预测曲线
    """
    X = np.asarray(X, dtype=float)
    if subsample is not None and subsample < len(X):
        rng = np.random.RandomState(0)
        X = X[rng.choice(len(X), subsample, replace=False)]
    N = X.shape[0]

    lo, hi = np.percentile(X[:, feature], percentiles)
    grid = np.linspace(lo, hi, grid_resolution)

    X_rep = np.tile(X, (grid_resolution, 1))
    X_rep[:, feature] = np.repeat(grid, N)

    preds = model.predict(X_rep).reshape(grid_resolution, N)
    return grid, preds.T   # (N, G)


def pdp_bivariate(model, X, features, grid_resolution=50,
                  percentiles=(0.05, 0.95)):
    """
    双向部分依赖（2-way PDP）。

    PD_{j,k}(x_j, x_k) = (1/N) * sum_i f(x_j, x_k, X_C^(i))

    返回
    ----
    g1 : (G,)    第一个特征网格
    g2 : (G,)    第二个特征网格
    Z  : (G, G)  Z[row=k, col=j] 对应 (x_j=g1[j], x_k=g2[k])
                 绘图时 pcolormesh(g1, g2, Z) 直接得到正确方向。
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    f1, f2 = features

    g1 = np.linspace(*np.percentile(X[:, f1], percentiles), grid_resolution)
    g2 = np.linspace(*np.percentile(X[:, f2], percentiles), grid_resolution)

    Z = np.empty((grid_resolution, grid_resolution))
    # 外层固定 f1，内层遍历 f2（每轮向量化 predict 一次，共 G 次）
    for i, v1 in enumerate(g1):
        X_rep = np.tile(X, (grid_resolution, 1))
        X_rep[:, f1] = v1
        X_rep[:, f2] = np.repeat(g2, N)
        preds = model.predict(X_rep).reshape(grid_resolution, N)
        Z[:, i] = preds.mean(axis=1)
    return g1, g2, Z


# ============================================================
# 第二部分：在同一糖尿病数据集上运行示例
# ============================================================

def main():
    # ---------- 0. 加载数据 + 训练被解释模型 ----------
    print("[1/4] 加载 diabetes 数据集并训练 RandomForestRegressor ...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names   # ['age','sex','bmi','bp','s1',...,'s6']

    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    print(f"      样本数 N={X.shape[0]}, 特征数 p={X.shape[1]}")

    # ---------- 1. 手写计算 ----------
    print("[2/4] 手写计算 PDP / ICE / 双向 PDP ...")
    bmi = feature_names.index("bmi")   # 2
    bp  = feature_names.index("bp")    # 3

    grid_bmi, pdp_bmi   = pdp_univariate(model, X, bmi, grid_resolution=100)
    grid_bp,  pdp_bp    = pdp_univariate(model, X, bp,  grid_resolution=100)
    grid_bp_ice, ice_bp = ice_univariate(model, X, bp,
                                         grid_resolution=100, subsample=80)
    g1, g2, Z           = pdp_bivariate(model, X, (bmi, bp),
                                        grid_resolution=50)

    # ---------- 2. 与 sklearn 官方实现对拍 ----------
    print("[3/4] 与 sklearn.inspection.partial_dependence 对拍 ...")
    from sklearn.inspection import partial_dependence as sk_pdp

    res  = sk_pdp(model, X, [bmi],  grid_resolution=100,
                  percentiles=(0.05, 0.95))
    err_grid = np.max(np.abs(res["grid_values"][0] - grid_bmi))
    err_pdp  = np.max(np.abs(res["average"][0]    - pdp_bmi))

    res2 = sk_pdp(model, X, [(bmi, bp)], grid_resolution=50,
                  percentiles=(0.05, 0.95))
    err_bi = np.max(np.abs(res2["average"][0] - Z))

    print(f"      单向 PDP 网格最大绝对误差 : {err_grid:.3e}")
    print(f"      单向 PDP 数值最大绝对误差 : {err_pdp:.3e}")
    print(f"      双向 PDP 数值最大绝对误差 : {err_bi:.3e}")
    assert err_pdp < 1e-10 and err_bi < 1e-10, "手写实现与官方结果不一致！"
    print("      ✓ 手写实现与官方实现数值一致（误差量级 1e-15，纯浮点舍入）")

    # ---------- 3. 可视化 ----------
    print("[4/4] 绘制图表 ...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # (a) bmi 的单向 PDP
    axs[0, 0].plot(grid_bmi, pdp_bmi, color="C0", lw=2)
    axs[0, 0].set_title("PDP of bmi (hand-written)")
    axs[0, 0].set_xlabel("bmi")
    axs[0, 0].set_ylabel("E[ y | bmi ]")
    axs[0, 0].grid(alpha=0.3)

    # (b) bp 的 ICE（细线）+ PDP（粗线）
    axs[0, 1].plot(grid_bp_ice, ice_bp.T, color="C1", alpha=0.15, lw=0.8)
    axs[0, 1].plot(grid_bp_ice, ice_bp.mean(axis=0), color="C3", lw=2.5)
    axs[0, 1].set_title("ICE (thin) + PDP (thick) of bp")
    axs[0, 1].set_xlabel("bp")
    axs[0, 1].grid(alpha=0.3)

    # (c) bp 的单向 PDP
    axs[1, 0].plot(grid_bp, pdp_bp, color="C2", lw=2)
    axs[1, 0].set_title("PDP of bp (hand-written)")
    axs[1, 0].set_xlabel("bp")
    axs[1, 0].set_ylabel("E[ y | bp ]")
    axs[1, 0].grid(alpha=0.3)

    # (d) (bmi, bp) 的双向 PDP
    mesh = axs[1, 1].pcolormesh(g1, g2, Z, shading="auto", cmap="viridis")
    axs[1, 1].set_title("2-way PDP of (bmi, bp)")
    axs[1, 1].set_xlabel("bmi")
    axs[1, 1].set_ylabel("bp")
    fig.colorbar(mesh, ax=axs[1, 1], label="E[ y | bmi, bp ]")

    plt.tight_layout()
    plt.savefig("pdp_from_scratch.png", dpi=120, bbox_inches="tight")
    print("      图表已保存到 pdp_from_scratch.png")
    plt.show()


if __name__ == "__main__":
    main()