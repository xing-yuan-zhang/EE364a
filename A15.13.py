import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from zero_crossings_data import *

C = np.zeros((n, B))
S = np.zeros((n, B))
for j in range(B):
    C[:, j] = np.cos(2 * np.pi * (f_min + j) * np.arange(1, n + 1) / n)
    S[:, j] = np.sin(2 * np.pi * (f_min + j) * np.arange(1, n + 1) / n)
A = np.hstack((C, S))

x = cp.Variable(2 * B)
obj = cp.norm(A @ x)
constraints = [cp.multiply(s, A @ x) >= 0, s.T @ (A @ x) == n]
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve()
y_hat = A @ x.value

print(("Recovery error: {}".format(
    np.linalg.norm(y- y_hat) / np.linalg.norm(y)))
)

plt.figure()
plt.plot(np.arange(0, n), y, label="original")
plt.plot(np.arange(0, n), y_hat, label="recovered")
plt.xlim([0, n])
plt.legend(loc="lower left")
plt.show()