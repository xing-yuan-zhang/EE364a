import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
m, n = 300, 100
A = np.random.rand(m, n)
b = A.dot(np.ones(n)) / 2
c =-np.random.rand(n)

x = cp.Variable(n)
objective = cp.Minimize(c @ x)
constraints = [0 <= x, x <= 1, A @ x <= b]
cp.Problem(objective, constraints).solve()

L = objective.value
x_rlx = x.value

N = 100
t = np.linspace(0, 1, N).reshape(N, 1)
maxviol = np.zeros((N, 1))
obj = np.zeros((N, 1))
U = float("inf")
t_best = float("nan")

for i in range(N):
    x = np.array(x_rlx >= t[i])
    obj[i] = c @ x
    maxviol[i] = max(A @ x- b)
    if maxviol[i] <= 0 and obj[i] < U:
        U = obj[i].item()
        x_best = x
        t_best = t[i]

print(L)
print(U)

plt.figure(1)
plt.subplot(211)
plt.plot(t[maxviol <= 0], maxviol[maxviol <= 0], "b")
plt.plot(t[maxviol > 0], maxviol[maxviol > 0], "r")
plt.ylabel("max violation")
plt.xlabel("threshold")

plt.subplot(212)
plt.plot(t[maxviol <= 0], obj[maxviol <= 0], "b")
plt.plot(t[maxviol > 0], obj[maxviol > 0], "r")
plt.plot(t, objective.value * np.ones((N, 1)), "g")
plt.ylabel("objective")
plt.xlabel("threshold")
plt.show()