"""
逻辑回归
"""
import matplotlib.pyplot as plt
import numpy as np

def init_data(m, bayes=0.0):
    """
    初始化数据
    """
    cov = np.mat([[1, bayes], [bayes, 1]]) # 协方差矩阵
    mean0 = [0, 0] # 均值
    x = np.ones((m, 3)) # 自变量
    x[:, 1:] = np.random.multivariate_normal(mean0, cov, m) # 初始化自变量为多维高斯分布
    y = np.ones((m, 1))
    for i in range(m):
        # 以直线x + y= 0划分区域
        if x[i][1] + x[i][2] > 0:
            y[i] = 1
        else:
            y[i] = 0
    return x, y


def gradient_descent(m, alpha=1e-3, lamb=0.0, max_iter=10000, bayes=0.0):
    """
    梯度下降法
    :return 系数向量 w
    """
    x, y = init_data(m, bayes)
    print("x= ", x)
    print("y= ", y)
    w_new = np.zeros((3, 1))  # 先随机生成一个w向量(3, 1)
    w_old = w_new  # 上一次预测的w生成的y值
    grad = gradient(x, y, w_new, lamb)  # 梯度
    print(grad.shape)
    grad_l = np.sqrt(grad.T.dot(grad))  # 单位梯度
    w_new -= grad * alpha / grad_l
    loss_new = loss(x, y, w_new, lamb)

    # 正式循环
    for time in range(max_iter): # 最多运行的次数,不设置loss的精度
        grad = gradient(x, y, w_new, lamb)  # 梯度
        grad_l = np.sqrt(grad.T.dot(grad))  # 梯度向量的长度
        loss_old = loss_new  # 将上一次loss更新
        w_new -= grad * alpha / grad_l  # 取梯度除以梯度向量的长度(变成单位梯度)
        loss_new = loss(x, y, w_new, lamb)  # 修改新的loss值
        # print("w=", w_new, " loss=", loss(x, y, w_new, lamb), " step=", step, " time:", time, "normal_time:", normal_time)
        if loss_new > loss_old:  # loss值增大了,先退回原处,再将步长减半
             w_new = w_old
             alpha /= 1.5

    # 最终结果
    print("loss=", loss_new, " w=", w_new, " step=", alpha)

    right = 0
    # 计算准确度
    for i in range(m):
        if ((y[i] == 1) & (w_new.T.dot(x[i]) > 0)) | ((y[i] == 0) & (w_new.T.dot(x[i]) < 0)):
            right += 1
    print("正确率是", right / m)

    # 下面是绘图操作
    ls0_x1 = []
    ls0_x2 = []
    ls1_x1 = []
    ls1_x2 = []
    x1_axis = np.linspace(-5, 5, 10)
    x2_axis = - w_new[1] / w_new[2] * x1_axis - w_new[0] / w_new[2]
    for i in range(m):
        if y[i] == 1:
            ls1_x1.append(x[i][1])
            ls1_x2.append(x[i][2])
        else:
            ls0_x1.append(x[i][1])
            ls0_x2.append(x[i][2])

    plt.plot(x1_axis, x2_axis)
    plt.scatter(ls0_x1, ls0_x2)
    plt.scatter(ls1_x1, ls1_x2)
    plt.show()


def gradient(x, y, w, lamb=0.0):
    """
    :return: 梯度
    """
    return x.T.dot(sigmoid(x.dot(w)) - y) + lamb * w


def loss(x, y, w, lamb=0.0):
    """
    :return: 损失函数值
    """
    m = len(y)
    lo = 0 # 损失值
    for i in range(m):
        lo += (-y[i] * np.log(sigmoid(w.T.dot(x[i])))) - (- y[i] + 1) * np.log(sigmoid(1 - w.T.dot(x[i])))
    return lo + lamb * 0.5 * w.T.dot(w)


def sigmoid(z):
    """
    sigmoid函数
    """
    return 1 / (np.exp(-z) + 1)


if __name__ == '__main__':
    gradient_descent(100, bayes=1, lamb=1)