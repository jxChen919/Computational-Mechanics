import numpy as np
import matplotlib.pyplot as plt

# 记录所有点的坐标
coord_list = []
with open(r'C:\Users\10564\Desktop\computational  mechanics\problem1_coord.txt') as f1:
    all_coord = f1.readlines()
    for i in range(len(all_coord)):
        temp1 = []
        for element in all_coord[i].split():
            temp1.append(float(element))
        coord_list.append(temp1)

# 记录所有三角形对应的节点
elem_list = []
with open(r'C:\Users\10564\Desktop\computational  mechanics\problem1_element.txt') as f2:
    all_elem = f2.readlines()
    for i in range(len(all_elem)):
        temp2 = []
        for element in all_elem[i].split():
            temp2.append(int(element))
        elem_list.append(temp2)

coords = np.array(coord_list)
elems = np.array(elem_list)


# 全局变量
K_glo = np.zeros((2 * len(coords), 2 * len(coords)))
F = np.zeros((2 * len(coords), 1))


def findLine(coord, direction, value):  # 注意 这里找的是direction = value 的轴
    bound = []
    for i in range(len(coord)):
        if direction == 'x':
            if value - 1e-5 <= coord[i, 0] <= value + 1e-5:
                bound.append(i)
        if direction == 'y':
            if value - 1e-5 <= coord[i, 1] <= value + 1e-5:
                bound.append(i)
    return np.array(bound)


def findAt(coord, x, y):
    point = []
    for i in range(len(coord)):
        if x - 1e-5 <= coord[i, 0] <= x + 1e-5 and y - 1e-5 <= coord[i, 1] <= y + 1e-5:
            point.append(i)
    return np.array(point)


# 材料性质矩阵
def matrix_D(type):  # type为0/1分别对应材料1/材料2
    if type == 0:
        E0 = E1 / (1 - v1 ** 2)
        v0 = v1 / (1 - v1)
    elif type == 1:
        E0 = E2 / (1 - v2 ** 2)
        v0 = v2 / (1 - v2)

    D = E0 / (1 - v0 ** 2) * np.array([[1, v0, 0],
                                       [v0, 1, 0],
                                       [0, 0, (1 - v0) / 2]])
    return D


# 几何矩阵
def matrix_B(i, j, k):
    i = i - 1
    j = j - 1
    k = k - 1

    b = np.array([coords[j, 1] - coords[k, 1], coords[k, 1] - coords[i, 1], coords[i, 1] - coords[j, 1]])
    c = np.array([-coords[j, 0] + coords[k, 0], -coords[k, 0] + coords[i, 0], -coords[i, 0] + coords[j, 0]])

    G = []
    for m in [i, j, k]:
        G.append([1, coords[m, 0], coords[m, 1]])

    G = np.array(G)
    delta = abs(np.linalg.det(G)) / 2

    B = 1 / (2 * delta) * np.array([[b[0], 0, b[1], 0, b[2], 0],
                                    [0, c[0], 0, c[1], 0, c[2]],
                                    [c[0], b[0], c[1], b[1], c[2], b[2]]])
    return B


# 刚度矩阵
def matrix_K(i, j, k):
    i = i - 1
    j = j - 1
    k = k - 1

    b = np.array([coords[j, 1] - coords[k, 1], coords[k, 1] - coords[i, 1], coords[i, 1] - coords[j, 1]])
    c = np.array([-coords[j, 0] + coords[k, 0], -coords[k, 0] + coords[i, 0], -coords[i, 0] + coords[j, 0]])

    G = []
    for m in [i, j, k]:
        G.append([1, coords[m, 0], coords[m, 1]])

    G = np.array(G)
    delta = abs(np.linalg.det(G)) / 2

    B = 1 / (2 * delta) * np.array([[b[0], 0, b[1], 0, b[2], 0],
                                    [0, c[0], 0, c[1], 0, c[2]],
                                    [c[0], b[0], c[1], b[1], c[2], b[2]]])

    x_center = (coords[i, 0] + coords[j, 0] + coords[k, 0]) / 3
    if x_center > 0:
        K = delta * t * B.T @ matrix_D(0) @ B
    else:
        K = delta * t * B.T @ matrix_D(1) @ B

    return K


def assemble(i, j, k):
    # 调用局部单元矩阵
    K_unit = matrix_K(i, j, k)

    global K_glo
    count_m = 0
    for m in [i - 1, j - 1, k - 1]:
        count_n = 0
        for n in [i - 1, j - 1, k - 1]:
            K_glo[2 * m: 2 * m + 2, 2 * n: 2 * n + 2] += K_unit[2 * count_m: 2 * count_m + 2, 2 * count_n: 2 * count_n + 2]
            count_n += 1
        count_m += 1


# 施加边界条件 常见的有固定位移(mode == 0)ux=0(mode == 1)uy=0(mode == 2)
def disp_boundary(boundary, mode):
    global F
    global K_glo
    if mode == 0:  # 采用直接法 把非对角元素置为0
        for i in boundary:
            for j in range(len(coords)):
                if j == i:
                    F[2 * j] = 0
                    F[2 * j + 1] = 0
                    K_glo[2 * j, 2 * j] += alpha
                    K_glo[2 * j + 1, 2 * j + 1] += alpha
                    for k in range(len(coords)):
                        if k != i:
                            K_glo[2 * j, 2 * k] = 0
                            K_glo[2 * j, 2 * k + 1] = 0
                            K_glo[2 * j + 1, 2 * k] = 0
                            K_glo[2 * j + 1, 2 * k + 1] = 0
                else:
                    K_glo[2 * j, 2 * i] = 0
                    K_glo[2 * j + 1, 2 * i] = 0
                    K_glo[2 * j, 2 * i + 1] = 0
                    K_glo[2 * j + 1, 2 * i + 1] = 0

    if mode == 2:
        for i in boundary:
            F[2 * i + 1] = 0
            for j in range(2 * len(coords)):
                if j != 2 * i + 1:
                    K_glo[j, 2 * i + 1] = 0
                    K_glo[2 * i + 1, j] = 0

    if mode == 1:
        for i in boundary:
            F[2 * i] = 0
            for j in range(2 * len(coords)):
                if j != 2 * i:
                    K_glo[j, 2 * i] = 0
                    K_glo[2 * i, j] = 0


# 施加应力边界条件
def load_boundary(boundary, direction, mode, load):  # 点载荷(mode==0) 面载荷(mode==1) 变力载荷(mode==2)
    # 在load_boundary前对边界节点排序
    global F
    if mode == 0:
        for i in boundary:
            if direction == 'x':
                F[2 * i] += load
            if direction == 'y':
                F[2 * i + 1] += load

    if mode == 1:
        for i in range(len(boundary) - 1):
            l = np.linalg.norm(coords[boundary[i + 1]] - coords[boundary[i]])
            if direction == 'x':
                F[2 * boundary[i]] += t * l * load / 2
                F[2 * boundary[i + 1]] += t * l * load / 2
            if direction == 'y':
                F[2 * boundary[i] + 1] += t * l * load * 1 / 2
                F[2 * boundary[i + 1] + 1] += t * l * load * 1 / 2


def linear_force_boundary(boundary, direction, start, end):
    global F
    df = np.linspace(start, end, len(boundary))
    print(df)
    for i in range(len(boundary) - 1):
        l = 0.375
        if direction == 'x':
            if end == 0:
                F[2 * boundary[i]] += t * l * (df[i] + df[i + 1]) / 4
                F[2 * boundary[i + 1]] += t * l * (df[i] + df[i + 1]) / 4
            if start == 0:
                F[2 * boundary[i]] += t * l * (df[i] + df[i + 1]) / 4
                F[2 * boundary[i + 1]] += t * l * (df[i] + df[i + 1]) / 4
        if direction == 'y':
            F[2 * boundary[i] + 1] += t * l * (df[i] + df[i + 1]) / 3
            F[2 * boundary[i + 1] + 1] += t * l * (df[i] + df[i + 1]) / 6


def gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    # 构建增广矩阵
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # 前向消元
    for i in range(n):
        # 选择主元行
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if np.abs(Ab[max_row, i]) < 1e-12:
            raise ValueError("矩阵为奇异矩阵（无解或无唯一解）")
        Ab[[i, max_row]] = Ab[[max_row, i]]  # 交换行

        # 消元
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # 回代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x


def inv_solution(A,b):
    return np.dot(np.linalg.inv(A), b)


# 计算第<index>个单元的mise应力(从0编号)
def get_mises(index):
    disp_e = []  # 存放单个单元位移的列表
    i = elems[index, 0]
    j = elems[index, 1]
    k = elems[index, 2]
    for n in [i - 1, j - 1, k - 1]:
        disp_e.append(d[2 * n])
        disp_e.append(d[2 * n + 1])
    disp_e = np.array(disp_e)
    center = get_center(index)
    if center[0] > 0:
        sigma_e = matrix_D(0) @ matrix_B(i, j, k) @ disp_e
    else:
        sigma_e = matrix_D(1) @ matrix_B(i, j, k) @ disp_e
    mise = np.sqrt((sigma_e[0] + sigma_e[1]) ** 2 - 3 * (sigma_e[0] * sigma_e[1] - sigma_e[2] ** 2))
    return mises


def get_center(index):  # 此函数用于找到第<index>个element的重心(从0编号)
    i = elems[index, 0]
    j = elems[index, 1]
    k = elems[index, 2]
    sum_x = 0
    sum_y = 0
    for n in [i - 1, j - 1, k - 1]:
        sum_x += coords[n, 0]
        sum_y += coords[n, 1]
    return np.array([sum_x / 3, sum_y / 3])


def get_elem(index):  # 此函数用于第<index - 1>个高斯积分点有关的所有element
    output = []
    for i in range(len(elems)):
        for j in elems[i]:
            if j == index:
                output.append(i)
    return np.array(output)


def get_stress(index):  # 此函数用于第<index>个高斯积分点有关的所有element
    index += 1
    ele = get_elem(index)
    sigma_point = np.zeros((3, 1))
    for n in ele:
        disp_e = []
        i = elems[n, 0]
        j = elems[n, 1]
        k = elems[n, 2]
        for m in [i - 1, j - 1, k - 1]:
            disp_e.append(d[2 * m])
            disp_e.append(d[2 * m + 1])
        disp_e = np.array(disp_e)
        sigma_e = matrix_D(1) @ matrix_B(i, j, k) @ disp_e
        sigma_point += sigma_e
    sigma_point = np.array(sigma_point) / len(ele)
    return sigma_point


# 参数设置
E1 = 205e9  # 弹性模量
v1 = 0.3  # 泊松比
E2 = 100e9
v2 = 0.25
t = 1  # 平板厚度
load1 = -300
load2 = 100
alpha = 1e14

# 组装总体刚度矩阵
for i in range(len(elems)):
    assemble(elems[i, 0], elems[i, 1], elems[i, 2])

# 查找边界
boundary_a = findLine(coords, 'x', -3)
boundary_b = findLine(coords, 'x', 3)
point_a = findAt(coords, -3, 0)
point_b = findAt(coords, 0, 1)
point_c = findAt(coords, -3, 3)
point_d = findAt(coords,3, 3)
point_e = findAt(coords, 0, 0)


# 设置边界条件
disp_boundary(point_c, 0)
disp_boundary(point_d, 0)
linear_force_boundary(boundary_a, direction='x', start=load2, end=0)
linear_force_boundary(boundary_b, direction='x', start=0, end=-load2)
load_boundary(point_e, direction='y', mode=0, load=load1)


# 计算得到结果
d = np.linalg.solve(K_glo, F)
print('A点位移：', d[2 * point_a[0]], d[2 * point_a[0] + 1])
print('B点位移：', d[2 * point_b[0]], d[2 * point_b[0] + 1])
print('A点应力：', get_stress(point_a[0])[0], get_stress(point_a[0])[1])
print('B点应力：', get_stress(point_b[0])[0], get_stress(point_b[0])[1])

max = 0
index = 0
for i in range(len(coords)):
    if abs(d[2 * i]) > max:
        max = abs(d[2 * i])
        index = i
print(max)
print(coords[index])

# 后处理
# mises = []
# for i in range(len(elems)):
#     mises.append(get_mises(i))
# mises = np.array(mises)
# print('最大应力单元：', np.where(mises == np.max(mises)))
# print('最大应力:', mises[72])
# print('最大应力坐标：', get_center(72))

# 查看刚度矩阵
# plt.spy(K_glo, precision=1e-5)
# plt.title("刚度矩阵稀疏模式")
# plt.show()
