import numpy as np
import matplotlib.pyplot as plt

# 参数设置
dxs = [0.1, 0.05, 0.025]
dt = 0.01 # ADI为恒稳格式 对时间步长无要求
x_max = y_max = 1
t_max = 3
t_steps = int(t_max / dt)
sigma = 1
NEs = []  # 存放数值误差
T = np.linspace(0, t_max - dt, t_steps)


def thomas(a, b, c, d):  # 三对角追赶法求解
    n = len(b)
    # 拷贝以避免修改原始数据
    ac, bc, cc, dc = map(list, (a, b, c, d))

    for i in range(1, n):
        m = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - m * cc[i - 1]
        dc[i] = dc[i] - m * dc[i - 1]

    x = [0] * n
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return np.array(x)


def f(t, x, y):
    return 20 + 80 * (y - np.exp(-0.5 * sigma * np.pi ** 2 * t) * np.sin(np.pi * x / 2) * np.sin(np.pi * y / 2))


def fx(t, x):
    return 20 + 80 * (1 - np.exp(-0.5 * sigma * np.pi ** 2 * t) * np.sin(np.pi * x / 2))


def fy(t, y):
    return 20 + 80 * (y - np.exp(-0.5 * sigma * np.pi ** 2 * t) * np.sin(np.pi * y / 2))


def delta_x(X, k, i, j):
    return X[k, j, i + 1] - 2 * X[k, j, i] + X[k, j, i - 1]


def delta_y(X, k, i, j):
    return X[k, j + 1, i] - 2 * X[k, j, i] + X[k, j - 1, i]


def max1(X):
    m = 0
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            if np.abs(X[j, i]) > m:
                m = np.abs(X[j, i])
    return m


for dx in dxs:
    a = dt * sigma / (2 * dx ** 2)
    Ehs = []  # 存放计算误差
    # 划分求解域
    x_steps = int(x_max / dx)
    u = np.zeros((t_steps + 1, x_steps + 1, x_steps + 1))
    v = np.zeros((t_steps, x_steps + 1, x_steps + 1))

    # 设置追赶法向量
    diag1 = (1 + 2 * a) * np.ones(x_steps - 1)
    diag2 = (-a) * np.ones(x_steps - 2)

    # 设置初始条件
    for i in range(x_steps + 1):
        for j in range(x_steps + 1):
            u[0, j, i] = f(0, i * dx, j * dx)

    # 设置边界条件
    for k in range(1, t_steps + 1):
        for i in range(x_steps + 1):
            u[k, 0, i] = 20
            u[k, x_steps, i] = fx(k * dt, i * dx)
        for j in range(x_steps + 1):
            u[k, j, 0] = 20 + 80 * j * dx
            u[k, j, x_steps] = fy(k * dt, j * dx)

    for k in range(t_steps):
        for j in range(1, x_steps):
            v[k, j, 0] = (u[k + 1, j, 0] + u[k, j, 0]) / 2 - a * (delta_y(u, k + 1, 0, j) - delta_y(u, k, 0, j)) / 4
            v[k, j, x_steps] = ((u[k + 1, j, x_steps,] + u[k, j, x_steps]) / 2
                                - a * (delta_y(u, k + 1, x_steps, j) - delta_y(u, k, x_steps, j)) / 4)

    # 采用ADI格式求解 使用三对角阵追赶法
    for k in range(t_steps):
        for j in range(1, x_steps):  # 列不变求行
            b = np.zeros(x_steps - 1)
            for n in range(1, x_steps):
                b[n - 1] = u[k, j, n] + a * delta_y(u, k, n, j)
            b[0] += a * v[k, j, 0]
            b[x_steps - 2] += a * v[k, j, x_steps]
            v[k, j, 1: x_steps] = thomas(diag2, diag1, diag2, b)

        for i in range(1, x_steps):  # 行不变求列
            c = np.zeros(x_steps - 1)
            for n in range(1, x_steps):
                c[n - 1] = v[k, n, i] + a * delta_x(v, k, i, n)
            c[0] += a * u[k + 1, 0, i]
            c[x_steps - 2] += a * u[k + 1, x_steps, i]
            u[k + 1, 1: x_steps, i] = thomas(diag2, diag1, diag2, c)

    # 求出数值误差（取中点的位置）
    NE = np.abs(u[200, int(x_steps / 2), int(x_steps / 2)] - f(2, 0.5, 0.5))
    NEs.append(NE)

    # 求出计算误差
    for k in range(t_steps):
        Eh = max1(u[k + 1] - u[k])
        Ehs.append(Eh)

    # 绘制计算误差图像
    plt.figure()
    plt.semilogy(T, Ehs)
    plt.title("T ~ logEh Δx=%s" % dx)
    plt.xlabel("T")
    plt.ylabel("logEh")
    plt.show()

plt.figure()
plt.loglog(dxs, NEs)
plt.title("logε ~ logh")
plt.xlabel("Δx")
plt.ylabel("Numerical Error ε")
plt.show()