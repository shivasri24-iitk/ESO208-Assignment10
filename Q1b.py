
import numpy as np

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
            A[p, index(i - 1, j)] -= 1.0

        if i > 0 and i < 3:
            A[p, index(i + 1, j)] -= 1.0
        elif i == 3:
            B[p] += 25.0 * j

        if j == 0:
            A[p, index(i, 1)] -= 2.0
        else:
            A[p, index(i, j - 1)] -= 1.0

        if j > 0 and j < 3:
            A[p, index(i, j + 1)] -= 1.0
        elif j == 3:
            B[p] += 25.0 * i


def gauss_elimination(A, B):
    A = A.copy()
    B = B.copy()
    n = len(B)

    for k in range(n):
        pivot = k + np.argmax(np.abs(A[k:, k]))

        if abs(A[pivot, k]) < 1e-12:
            raise ValueError("Singular matrix")

        A[[k, pivot]] = A[[pivot, k]]
        B[[k, pivot]] = B[[pivot, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            B[i] -= factor * B[k]

    X = np.zeros(n)

    for i in range(n - 1, -1, -1):
        X[i] = (B[i] - np.dot(A[i, i+1:], X[i+1:])) / A[i, i]

    return X


T = gauss_elimination(A, B)

print("Temperature Distribution (°C):")
print(np.round(T.reshape(4, 4), 4))

print("\nIndividual Node Temperatures:")

for j in range(4):
    for i in range(4):
        print(f"T[{i},{j}] = {T[index(i,j)]:.4f} °C")

print("\nMaximum Residual Error:")
print(np.max(np.abs(A @ T - B)))
