
import numpy as np
import matplotlib.pyplot as plt

K = 0.49
dx = 10.0
dy = 10.0

n = 16
A = np.zeros((n, n))
B = np.zeros(n)

def index(i, j):
    return j * 4 + i

for j in range(4):
    for i in range(4):
        p = index(i, j)
        A[p, p] = 4.0

        if i == 0:
            A[p, index(1, j)] -= 2.0
        else:
            A[p, index(i-1, j)] -= 1.0

        if 0 < i < 3:
            A[p, index(i+1, j)] -= 1.0
        elif i == 3:
            B[p] += 25.0 * j

        if j == 0:
            A[p, index(i, 1)] -= 2.0
        else:
            A[p, index(i, j-1)] -= 1.0

        if 0 < j < 3:
            A[p, index(i, j+1)] -= 1.0
        elif j == 3:
            B[p] += 25.0 * i

T_nodes = np.linalg.solve(A, B)

T = np.zeros((5, 5))
T[:4, :4] = T_nodes.reshape(4, 4)
T[4, :] = [0, 25, 50, 75, 100]
T[:, 4] = [0, 25, 50, 75, 100]

x = np.arange(0, 41, dx)
y = np.arange(0, 41, dy)

X, Y = np.meshgrid(x, y)

dTdy, dTdx = np.gradient(T, dy, dx)

qx = -K * dTdx
qy = -K * dTdy

qx[:, 0] = 0
qy[0, :] = 0

plt.figure(figsize=(8, 6))

contour = plt.contourf(X, Y, T, levels=20, cmap='jet')
plt.colorbar(contour, label='Temperature (°C)')
plt.contour(X, Y, T, colors='black', linewidths=0.5)

plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Temperature Distribution in the Plate')
plt.tight_layout()
plt.savefig('Q1c_temperature_profile.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 6))

plt.quiver(
    X[:4, :4], Y[:4, :4],
    qx[:4, :4], qy[:4, :4],
    angles='xy', scale_units='xy', scale=0.15,
    width=0.004, color='blue'
)

plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Heat Flux at Internal and Insulated Nodes')
plt.xlim(-3, 35)
plt.ylim(-3, 35)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Q1c_heat_flux.png', dpi=300)
plt.show()

print("Temperature Distribution (°C):")
print(np.round(T_nodes.reshape(4, 4), 4))

print("\nHeat Flux in X Direction:")
print(np.round(qx[:4, :4], 4))

print("\nHeat Flux in Y Direction:")
print(np.round(qy[:4, :4], 4))

print("\nBoth plots saved successfully.")
