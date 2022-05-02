"""
生成数据函数
"""
import numpy as np
import numpy.random as npr
from math import pi

def init_data(m, n):
    """
    创建数据
    :param m 生成矩阵x的行数
    :param n + 1 生成矩阵x的列数,多项式的最大次数
    :return: 参数 w
    """
    x0 = np.arange(-1, 1, 2 / m).T
    gauss_noise = npr.normal(0, 0.1, (m,))
    y = np.sin(pi * x0) + gauss_noise
    x = np.zeros((m, n + 1))
    for i in range(m):
        x[i][1] = x0[i]
        for j in range(n + 1):
            x[i][j] = x[i][1] ** j
    return x, y