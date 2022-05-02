"""
梯度下降法
"""
import numpy as np
from init_data import init_data
import matplotlib.pyplot as plt

# 梯度下降法
def gradient_descent(x, y, lamb, step):
    """
    :param x: 自变量s1
    :param y: 因变量
    :param lamb: 正则项系数
    :param step 步长
    :return 系数向量 w
    """
    w_new = np.zeros((x.shape[1], )) # 先随机生成一个w向量(m, 1)
    w_old = w_new # 上一次预测的w生成的y值
    grad = x.T.dot(x).dot(w_new) - x.T.dot(y) + lamb * w_new  # 梯度
    grad_l = np.sqrt(grad.dot(grad)) # 单位梯度
    w_new -= grad * step / grad_l
    loss_new = loss(y, x, w_new)
    time = 0 # 执行的次数
    normal_time = 0 # 正常前进的次数
    while True:
        if time > 100000: # 最多运行的次数,不设置loss的精度
            break
        grad = x.T.dot(x).dot(w_new) - x.T.dot(y) + lamb * w_new # 梯度
        grad_l = np.sqrt(grad.dot(grad))  # 梯度向量的长度
        loss_old = loss_new # 将上一次loss更新
        w_new -= grad * step / grad_l  # 取梯度除以梯度向量的长度(变成单位梯度)
        loss_new = loss(y, x, w_new)  # 修改新的loss值
        # print("w=", w_new, " loss=", loss(y, x, w_new), " step=", step, " time:", time, "normal_time:", normal_time)
        if loss_new > loss_old: # loss值增大了,先退回原处,再将步长减半
            w_new = w_old
            step /= 1.5
        else:
            normal_time += 1
            if normal_time >= 1000: # 正常前进n步,就把步长乘以2
                step *= 2
                normal_time = 0
        time += 1 # 更新time
    print("loss=", loss_new, " step=", step)
    return w_new


# 共轭梯度法
def gradient_conjugate(x, y, lamb, precision):
    """
    共轭梯度法
    :param x: 自变量
    :param y: 因变量
    :param lamb: 正则项系数
    :param precision 精度
    :return 系数向量
    """
    w = np.zeros((x.shape[1],)) # 设置w的初始值
    A = x.T.dot(x) + lamb * np.eye(x.shape[1]) # (n+1)*(n+1)维
    b = x.T.dot(y) # (n+1)*1, p和b都是算法中需要的中间变量
    r = b - A.dot(w)  # 残度,即梯度的相反数 (n+1)*1维
    p = r  # 初始化p = r = b-aX
    t = 0
    while True:
        if (r.T.dot(r) < precision) | (t > 1000000):
            break
        step = r.T.dot(r) / (p.T.dot(A).dot(p)) # 步长
        w += p * step # w沿前进下降一个步长
        r_new = r - step * A.dot(p) # 更新残量r
        b = r_new.T.dot(r_new) / r.T.dot(r) # 更新b
        p = r_new + b * p # 更新p
        r = r_new
        t += 1
    print("loss=", loss(y, x, w))
    return w


def gradient(x, y, w, lamb):
    """
    :return: 梯度
    """
    return x.T.dot(x).dot(w) - x.T.dot(y) + lamb * w


def loss(y_true, x, w):
    """
    loss惩罚函数
    :param y_true: y的真实值
    :param x 自变量
    :param w 系数列向量
    :return: 惩罚值(误差值)
    """
    return (y_true - x.dot(w)).T.dot(y_true - x.dot(w))


if __name__ == '__main__':
    x, y = init_data(20, 9)
    # w = gradient_descent(x, y, 0.1, 1e-3)
    w = gradient_conjugate(x, y, 0.1, 1e-3)
    plt.scatter(x[:, 1], y)
    y1 = x.dot(w)
    plt.plot(x[:, 1], y1)
    print(x[:, 1])
    print(y1)
    plt.show()