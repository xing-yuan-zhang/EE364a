import cvxpy as cp
import numpy as np

n = 4
Atot = 10000
alpha = np.array([1e-5, 1e-2, 1e-2, 1e-2])
M = np.array([0.1, 5, 10, 10])
Amax = np.array([40, 40, 40, 20])

a = cp.Variable(n, pos=True)
S = cp.Variable(n, pos=True)
Sin = cp.Variable( pos=True)

constrs = []
constrs.append(cp.prod(a) == Atot)
constrs.append(a <= Amax)
constrs.append(S[0] == a[0] * Sin)
for i in range(1, n):
    constrs.append(S[i] == a[i] * S[i-1])
constrs.append(S <= M)

Nsquare = []
Nsquare.append(a[0] ** 2 * alpha[0] ** 2)
for i in range(1, n):
    Nsquare.append(a[i] ** 2 * (Nsquare[-1] + alpha[i] ** 2))
objective = cp.Maximize(S[-1] / cp.sqrt(Nsquare[-1]))
problem = cp.Problem(objective, constrs)
problem.solve(gp=True)

print("D", problem.value)
print("a", a.value)
print("S", S.value)
print("Sin", Sin.value)