import numpy as np
import math

# 创建数组x,y,z来存放节点坐标
x = np.array([58,48,31,17,0,58,48,36,0,58,58,0,0,18,0]).astype(float)
y = np.array([38,38,38,38,38,38,38,38,38,17,17,17,17,0,0]).astype(float)
z = np.array([0,0,0,22,24,42,42,70,75,42,0,0,24,72,37.5]).astype(float)
# 将堆成节点坐标也写进数组
for i in range(13):
    x = np.append(x,x[i])
    y = np.append(y,-y[i])
    z = np.append(z,z[i])

# 参数常量设置
Iz = 0.003
Iy = 0.003
J = 0.006
A = 0.2
E = 100.
G = 38.5

# 定义一个异常类 方便debug
class InputError(Exception):
    def __init__(self,message):
        self.message = message

# 列出两节点中间梁的单元刚度方程
def single_unit(i,j):
    i = i - 1
    j = j - 1
    l = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
    K = np.zeros((12,12),dtype=float)
    # 我们要对坐标向量做一个规定，即前六项分别为i节点的xyz方向位移和xyz方向转角，后六项为j的对应值
    # 写入扭转刚度
    for m in [4,10]:
        for n in [4,10]:
            if m == n :
                K[m-1,n-1] += (G*J/l)
            else:
                K[m-1,n-1] += -(G*J/l)
    # 写入拉伸刚度
    for m in [1,7]:
        for n in [1,7]:
            if m == n:
                K[m-1,n-1] += (E*A/l)
            else:
                K[m-1,n-1] += -(E*A/l)

    k = np.array([[12, 6*l, -12, 6*l],
                  [6*l, 4*l*l, -6*l, 2*l*l],
                  [-12, -6*l, 12, -6*l],
                  [6*l, 2*l*l, -6*l, 4*l*l]]).astype(float)
    # 写入弯曲刚度
    count_m = 0
    for m in [2, 6, 8, 12]:
        count_n = 0
        count_m += 1
        for n in [2, 6, 8, 12]:
            count_n += 1
            K[m - 1, n - 1] += (E * Iz / l * l * l) * k[count_m - 1, count_n - 1]

    count_m = 0
    for m in [3, 5, 9, 11]:
        count_n = 0
        count_m += 1
        for n in [3, 5, 9, 11]:
            count_n += 1
            K[m - 1, n - 1] += (E * Iy / l * l * l) * k[count_m - 1, count_n - 1]

    return K

# 列出两节点中间梁的坐标转换方程
def coord_trans(i,j):
    i = i - 1
    j = j - 1
    local_x = np.array([x[j] - x[i], y[j] - y[i], z[j] - z[i]])
    # 需要把杆分为横杆和竖杆 便于讨论
    if local_x[2] == 0:
        local_y = [-local_x[1], local_x[0], 0]
        local_z = [0, 0, 1]
    elif local_x[1] == 0:
        local_y = [local_x[2], 0, -local_x[0]]
        local_z = [0, 1, 0]
    elif local_x[0] == 0:
        local_y = [0, -local_x[2], local_x[1]]
        local_z = [1, 0, 0]
    elif j == 13:   # 这里有些杆比较特殊 独立讨论一下
        if i == 7 or i == 8:
            rf = np.array([x[j] - x[i + 15], y[j] - y[i + 15], z[j] - z[i + 15]])
        if i == 22 or i == 23:
            rf = np.array([x[j] - x[i - 15], y[j] - y[i - 15], z[j] - z[i - 15]])
        local_z = np.cross(rf, local_x)
        local_y = np.cross(local_z, local_x)
    else:
        raise InputError("可能输入了错误的节点编号")

    loc = np.array([local_x, local_y, local_z])

    global_x = np.array([1, 0, 0]).astype(float)
    global_y = np.array([0, 1, 0]).astype(float)
    global_z = np.array([0, 0, 1]).astype(float)
    glo = np.array([global_x, global_y, global_z])

    t = np.zeros((3, 3))
    for m in range(3):
        for n in range(3):
            t[m, n] = np.dot(glo[m], loc[n])/(np.linalg.norm(glo[m]) * np.linalg.norm(loc[n]))

    T = np.zeros((12, 12),dtype=float)
    for m in range(3):
        for n in range(3):
            T[m, n] = t[m, n]
            T[m + 3, n + 3] = t[m, n]
            T[m + 6, n + 6] = t[m, n]
            T[m + 9, n + 9] = t[m, n]

    return T

K_glo = np.zeros((6*28,6*28))
# 将杆件组装起来
def assemble(i,j):
    # 对局部坐标下单元矩阵坐标转换得到整体坐标下单元矩阵
    K_unit = np.dot(np.dot(coord_trans(i, j).T, single_unit(i, j)), coord_trans(i, j))

    m = i - 1
    n = j - 1
    # 调用总体刚度矩阵，并进行组装
    global K_glo
    for k in range(6):
        K_glo[6 * m + k, 6 * m + k] += K_unit[k, k]
        K_glo[6 * m + k, 6 * n + k] += K_unit[k, 6 + k]
        K_glo[6 * n + k, 6 * m + k] += K_unit[6 + k, k]
        K_glo[6 * n + k, 6 * n + k] += K_unit[6 + k, 6 + k]


# 接下来写主函数
# 记录所有的杆件,对称的杆件共
list = [[1, 2],[2, 3],[3, 4],[4, 5],[5, 9],[1, 11],[10, 11],[6, 10],[6, 7],[7, 8],[8, 9],
        [11, 12],[2, 7],[5, 13],[12, 13],[5, 15],[9, 15],[8, 14],[9, 14],
        [11, 26],[10, 25],[8, 23],[9, 24],[13, 28]]

# for i in range(len(list)):
#     if i <= 14:
#         assemble(list[i,0],list[i,1])
#         assemble(list[i,0]+15,list[i,1]+15)
#     elif i <= 17:
#         assemble(list[i,0],list[i,1])
#         assemble(list[i,0]+15,list[i,1])
#     else:
#         assemble(list[i, 0], list[i, 1])

for i in range(len(list)):
    if i <= 14:
        list.append([list[i][0]+15,list[i][1]+15])
    elif i <= 17:
        list.append([list[i][0]+15,list[i][1]])

list = np.array(list)

# 总体系统组装
for i in range(len(list)):
    assemble(list[i, 0], list[i, 1])

P = np.zeros(6*28,dtype=float)
P[0] = -3194.
P[1] = -856.

# 施加边界条件（删除被约束的点所对应的行和列）
cons = []
for i in [11,12,26,27]:
    for j in range(6):
        cons.append(6 * (i - 1) + j)

K_cons = np.delete(K_glo, cons, axis=0)
P_constrain = np.delete(P, cons, axis=0)
K_constrain = np.delete(K_cons, cons, axis=1)

q = np.dot(np.linalg.inv(K_constrain), P_constrain)

# 输出所求节点的挠度和转角
for i in [1, 2, 6, 7, 10]:
    print("节点%d :" %i)
    for j in range(6):
        print(q[6 * (i - 1) + j])

q_whole = q
for i in [11, 12, 26, 27]:
    q_whole = np.insert(q_whole, 6 * (i - 1), [0,0,0,0,0,0])

M_record = []
for i in range(len(list)):
    K_unit = np.dot(np.dot(coord_trans(list[i,0], list[i,1]).T, single_unit(list[i,0], list[i,1])), coord_trans(list[i,0], list[i,1]))
    q_unit = np.zeros(12)
    for j in range(6):
        q_unit[j] = q_whole[6 * (list[i,0] - 1) + j]
        q_unit[6 + j] = q_whole[6 * (list[i,1] - 1) + j]
    M = np.dot(K_unit, q_unit)
    m1 = math.sqrt(M[3] ** 2 + M[4] ** 2 + M[5] ** 2)
    m2 = math.sqrt(M[9] ** 2 + M[10] ** 2 + M[11] ** 2)

    M_record.append([list[i, 0], m1])
    M_record.append([list[i, 1], m2])


M_record = np.array(M_record)
j = 0
for i in range(1, len(M_record)):
    if M_record[i, 1] >= M_record[j, 1]:
        j = i
print(M_record[j])





