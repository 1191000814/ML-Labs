"""
最小二乘法
"""

import numpy as np
from init_data import init_data
import matplotlib.pyplot as plt

def least_square(x, y, lamb):
    """
    :param x: 自变量
    :param y: 因变量
    :param lamb: 正则项系数(为 0 表示无正则项)
    :return 系数向量 w
    """
    print("x:", x)
    print("y:", y)
    if lamb == 0:
        w = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    else:
        w = np.linalg.pinv(x.dot(x.T).dot(x) + lamb * x).dot(x.dot(x.T)).dot(y)
    print("w:", w)
    return w


if __name__ == '__main__':
    x, y = init_data(10, 9)
    w = least_square(x, y, 0)
    x1 = x[:, 1]
    print(x1)
    plt.scatter(x1, y.T)
    y_pre = x.dot(w.T)
    print("y_pre:", y_pre)
    plt.plot(x1, y_pre)
    plt.show()