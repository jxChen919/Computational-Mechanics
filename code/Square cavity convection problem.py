import numpy as np
import matplotlib.pyplot as plt


def initialize(nx, ny):
    # 定义交错网格变量
    u = np.zeros((nx + 2, ny + 1))  # u(i-1/2, j)
    v = np.zeros((nx + 1, ny + 2))  # v(i, j-1/2)
    p = np.zeros((nx + 1, ny + 1))  # p(i, j)
    return u, v, p


def apply_boundary_conditions(u, v, lid_velocity=1.0):
    # 顶盖速度（u方向）
    u[:, -1] = -1
    u[:, 0] = 0
    u[0, :] = -u[1, :]
    u[-1, :] = -u[-2, :]

    v[:, 0] = -v[:, 1]
    v[:, -1] = -v[:, -2]
    v[0, :] = 0
    v[-1, :] = 0
    return u, v


def compute_rhs(u, v, dx, dy, dt, nx, ny):
    rhs = np.zeros((nx + 1, ny + 1))
    for i in range(0, nx + 1):
        for j in range(0, ny + 1):
            du_dx = (u[i + 1, j] - u[i, j]) / dx
            dv_dy = (v[i, j + 1] - v[i, j]) / dy
            rhs[i, j] = (du_dx + dv_dy) / dt
    return rhs


def pressure_poisson(p, rhs, dx, dy, tol=1e-2, max_iter=50):
    pn = p.copy()
    for _ in range(max_iter):
        p_old = pn.copy()
        pn[1:-1, 1:-1] = (((p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2 +
                           (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 -
                           rhs[1:-1, 1:-1] * dx**2 * dy**2) /
                          (2 * (dx**2 + dy**2)))

        pn[:, 0] = pn[:, 1]
        pn[:, -1] = pn[:, -2]
        pn[0, :] = pn[1, :]
        pn[-1, :] = pn[-2, :]

        if np.linalg.norm(pn - p_old, ord=2) < tol:
            break
    return pn


def tentative_velocity(u, v, p, dx, dy, dt, Re):
    un = u.copy()
    vn = v.copy()

    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            v_avg = 0.25 * (v[i - 1, j + 1] + v[i, j + 1] + v[i - 1, j] + v[i, j])
            du2dx = (u[i + 1, j]**2 - u[i - 1, j]**2) / (2 * dx)
            duvdy = (u[i, j + 1] * v_avg - u[i, j - 1] * v_avg) / (2 * dy)
            d2udx2 = (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2
            d2udy2 = (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy**2
            un[i, j] = u[i, j] + dt * (-du2dx - duvdy + (1 / Re) * (d2udx2 + d2udy2))

    for i in range(1, v.shape[0] - 1):
        for j in range(1, v.shape[1] - 1):
            u_avg = 0.25 * (u[i + 1, j - 1] + u[i, j - 1] + u[i + 1, j] + u[i, j])
            dv2dy = (v[i, j + 1]**2 - v[i, j - 1]**2) / (2 * dy)
            duvdx = (v[i + 1, j] * u_avg - v[i - 1, j] * u_avg) / (2 * dx)
            d2vdx2 = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2
            d2vdy2 = (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy**2
            vn[i, j] = v[i, j] + dt * (-duvdx - dv2dy + (1 / Re) * (d2vdx2 + d2vdy2))

    return un, vn


def correct_velocity(u, v, p, dx, dy, dt):
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            u[i, j] -= dt * (p[i, j] - p[i - 1, j]) / dx

    for i in range(1, v.shape[0] - 1):
        for j in range(1, v.shape[1] - 1):
            v[i, j] -= dt * (p[i, j] - p[i, j - 1]) / dy

    return u, v


def plot_streamlines(u, v, nx, ny, Re):
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    X, Y = np.meshgrid(x, y)

    u_center = 0.5 * (u[1:, :] + u[:-1, :])  # shape: (nx, ny)
    v_center = 0.5 * (v[:, 1:] + v[:, :-1])  # shape: (nx, ny)


    plt.figure(figsize=(6, 6))
    plt.streamplot(X, Y, u_center.T, v_center.T, density=2)
    plt.title("Streamlines Re=%s, Grid=%sx%s" % (Re, nx, ny))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.show()


def plot_profiles(u, v, nx, ny):
    dy = 1.0 / ny
    dx = 1.0 / nx
    y = np.linspace(0, 1, ny + 1)
    x = np.linspace(0, 1, nx + 1)

    u_center = 0.5 * (u[nx // 2, :] + u[nx // 2 + 1, :])
    v_center = 0.5 * (v[:, ny // 2] + v[:, ny // 2 + 1])

    plt.figure()
    plt.plot(u_center, y, label="u(x=0.5)")
    plt.xlabel("u"); plt.ylabel("y"); plt.grid(); plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x, v_center, label="v(y=0.5)")
    plt.xlabel("x"); plt.ylabel("v"); plt.grid(); plt.legend()
    plt.show()


def cavity_flow(Re, nx, ny, dt, nt):
    dx = 1.0 / nx
    dy = 1.0 / ny

    u, v, p = initialize(nx, ny)

    for step in range(nt):
        u, v = apply_boundary_conditions(u, v)
        ut, vt = tentative_velocity(u, v, p, dx, dy, dt, Re)
        rhs = compute_rhs(ut, vt, dx, dy, dt, nx, ny)
        p = pressure_poisson(p, rhs, dx, dy)
        u, v = correct_velocity(ut, vt, p, dx, dy, dt)

        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}/{nt}")

    return u, v, p


# 主运行入口
if __name__ == "__main__":
    Res = [100, 400, 1000]
    grids = [40, 80, 160]

    for nx in grids:
        ny = nx
        for Re in Res:
            print(f"\nRunning: Re={Re}, Grid={nx}x{ny}")
            u, v, p = cavity_flow(Re=Re, nx=nx, ny=ny, dt=0.0005, nt=40000)
            plot_streamlines(u, v, nx, ny, Re)
            if Re == 1000:
                plot_profiles(u, v, nx, ny)
