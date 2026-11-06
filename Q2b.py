
import numpy as np

K = 0.49
C = 0.2174
rho = 2.7

alpha = K / (rho * C)


def crank_nicolson(dx, dt, total_time):

    L = 10.0
    nx = int(L / dx) + 1
    nt = int(round(total_time / dt))
    m = nx - 2

    r = alpha * dt / dx**2

    A = np.zeros((m, m))
    B = np.zeros((m, m))

    for i in range(m):

        if i == 0 or i == m - 1:
            A[i, i] = 1 + r / 2
            B[i, i] = 1 - r / 2
        else:
            A[i, i] = 1 + r
            B[i, i] = 1 - r

        if i > 0:
            A[i, i-1] = -r / 2
            B[i, i-1] = r / 2

        if i < m - 1:
            A[i, i+1] = -r / 2
            B[i, i+1] = r / 2

    T_internal = np.full(m, 50.0)

    history = [T_internal.copy()]

    for step in range(nt):

        rhs = B @ T_internal
        rhs[0] -= alpha * dt / dx

        T_internal = np.linalg.solve(A, rhs)

        history.append(T_internal.copy())

    history = np.array(history)

    T_full = np.zeros((nt + 1, nx))
    T_full[:, 1:-1] = history

    T_full[:, 0] = T_full[:, 1] - dx
    T_full[:, -1] = T_full[:, -2]

    time = np.arange(nt + 1) * dt
    x = np.arange(nx) * dx

    return x, time, T_full


if __name__ == "__main__":

    dx = 2.0
    dt = 0.1
    total_time = 30.0

    x, time, T = crank_nicolson(dx, dt, total_time)

    print("Thermal diffusivity:", alpha, "cm^2/s")
    print("\nTemperature distribution at t = 30 s:")

    for i in range(len(x)):
        print(f"x = {x[i]:.1f} cm : T = {T[-1, i]:.4f} °C")

    print("\nQ2(b) completed successfully.")
