import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# y=β_0+β_1*X_1+⋯+β_n*X_n+ε
nsize = 100

# 产生0到10的等差数列，作为自变量x
x = np.linspace(0, 10, nsize)

# 为x添加一列全为1的常量,即X_0等于1
X = sm.add_constant(x)

# 设置模型中的β_0, β_1
beta = np.array([1, 10])

# 数据中加上误差项，生成一个长度为k的正态分布样本。
e = np.random.normal(size=nsize)

# 生成 y
y = np.dot(X, beta) + e

# 使用OLS()进行回归
model = sm.OLS(y, X)

# 获取回归结果result
result = model.fit()

# 打印回归信息
print(result.params)

