import numpy as np
import math
import matplotlib.pyplot as plt

# 记录所有点的坐标
coord_list = []
with open (r'C:\Users\10564\Desktop\computational  mechanics\mesh_8_coord.txt') as f1:
    all_coord = f1.readlines()
    for i in range (len(all_coord)):
        temp1 = []
        for element in all_coord[i].split():
            temp1.append(float(element))
        coord_list.append(temp1)

# 记录所有三角形对应的节点
elem_list = []
with open (r'C:\Users\10564\Desktop\computational  mechanics\mesh_8_element.txt') as f2:
    all_elem = f2.readlines()
    for i in range (len(all_elem)):
        temp2 = []
        for element in all_elem[i].split():
            temp2.append(int(element))
        elem_list.append(temp2)

coord = np.array(coord_list)
elem = np.array(elem_list)

# 查找所有边界点并记录
bound_a = []
for i in range(len(coord)):
    if 0.5-1e-5 <= math.sqrt(coord[i, 0]**2 + coord[i, 1]**2) <= 0.5+1e-5:
        bound_a.append(i)

bound_b = []
for i in range(len(coord)):
    if 1.5-1e-5 <= math.sqrt(coord[i, 0]**2 + coord[i, 1]**2) <= 1.5+1e-5:
        bound_b.append(i)

bound_a = np.array(bound_a)
bound_b = np.array(bound_b)

# 额外布置题目的边界条件点
# bound_a = np.array([0, 1])
# bound_b = np.array([4, 5])

# 参数常量设置
k = 1
Ta = 400
Tb = 300
q = 100

# 创建单元
def single_unit(i, j, k):
    i = i - 1
    j = j - 1
    k = k - 1

    b = np.array([coord[j, 1] - coord[k, 1], coord[k, 1] - coord[i, 1], coord[i, 1] - coord[j, 1]])
    c = np.array([-coord[j, 0] + coord[k, 0], -coord[k, 0] + coord[i, 0], -coord[i, 0] + coord[j, 0]])
    B = np.zeros((2,3),dtype=float)
    for m in range(3):
        B[0, m] = b[m]
        B[1, m] = c[m]


    G = []
    for m in [i, j, k]:
        G.append([1, coord[m, 0], coord[m, 1]])

    G = np.array(G)
    delta = abs(np.linalg.det(G))

    K = (1/(4 * delta)) * np.dot(B.T, B)

    return K


K_glo = np.zeros((len(coord),len(coord)))
# 组合
def assemble(i, j, k):
    # 调用局部单元矩阵
    K_unit = single_unit(i, j, k)

    global K_glo
    count_m = 0
    for m in [i - 1, j - 1, k - 1]:
        count_n = 0
        for n in [i - 1, j - 1, k - 1]:
            K_glo[m, n] += K_unit[count_m, count_n]
            count_n += 1
        count_m += 1

# 接下来是主函数
# 组装K和Q
for i in range(len(elem)):
    assemble(elem[i, 0], elem[i, 1], elem[i, 2])

Q = np.zeros(len(coord))

# 施加边界条件
def set_boundary(situ,mode):
    global Q
    global K_glo
    if situ == 'b':
        if mode == 1:
            # 上边界I型
            for i in bound_b:
                for j in range(len(coord)):
                    if j == i:
                        Q[j] = K_glo[j, i] * Tb
                        for k in range(len(coord)):
                            if k != i:
                                K_glo[j, k] = 0
                    else:
                        Q[j] -= K_glo[j, i] * Tb
                        K_glo[j, i] = 0
        if mode == 2:
            # 上边界II型
            l = math.sqrt((coord[bound_b[0], 0] - coord[bound_b[1], 0]) ** 2 + (coord[bound_b[0], 1] - coord[bound_b[1], 1]) ** 2) * 0.5
            for i in range(len(bound_b) - 1):
                Q[bound_b[i]] += -q * l / 2
                Q[bound_b[i + 1]] += -q * l / 2

    if situ == 'a':
        if mode == 1:
            # 下边界I型
            for i in bound_a:
                for j in range(len(coord)):
                    if j == i:
                        Q[j] = K_glo[j, i] * Ta
                        for k in range(len(coord)):
                            if k != i:
                                K_glo[j, k] = 0
                    else:
                        Q[j] -= K_glo[j, i] * Ta
                        K_glo[j, i] = 0
        if mode == 2:
            # 下边界II型
            l = math.sqrt((coord[bound_a[0], 0] - coord[bound_a[1], 0]) ** 2 + (coord[bound_a[0], 1] - coord[bound_a[1], 1]) ** 2) * 0.5
            for i in range(len(bound_a) - 1):
                Q[bound_a[i]] += -q * l / 2
                Q[bound_a[i + 1]] += -q * l / 2

set_boundary('a',1)
set_boundary('b',2)

d = np.dot(np.linalg.inv(K_glo),Q)
print(d)

def picture():
    simu = []
    displace = []
    # 取横坐标为0的点作图
    for i in range(len(coord)):
        if -1e-8 <= coord[i, 0] <= 1e-8:
            simu.append(d[i])
            displace.append(coord[i, 1])

    simu = np.array(simu)
    displace = np.array(displace)
    analy = np.zeros(len(displace))

    # 计算解析解
    for i in range(len(displace)):
        analy[i] = 400 - 150 * math.log(2 * displace[i])

    plt.subplot(221)
    plt.plot(displace,simu)
    plt.plot(displace,analy)
    plt.legend(['numerical','analytical'])
    plt.title('T-r relations(8*8*2)')
    plt.xlabel('T')
    plt.ylabel('r')

    plt.subplot(224)
    plt.scatter(simu,analy)
    plt.title('numer-analy comparison')
    x = np.linspace(300,400,50)
    plt.plot(x,x)
    plt.xlabel('numerical')
    plt.ylabel('analytical')
    plt.show()

picture()