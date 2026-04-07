import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

k = 201
t =-3 + 6 * np.arange(k) / (k- 1)
y = np.exp(t)

Tpowers = np.vstack((np.ones(k), t, t**2)).T
a = cvx.Variable(3)
b = cvx.Variable(2)
gamma = cvx.Parameter(nonneg=True)
lhs = cvx.abs(Tpowers @ a- (y[:, np.newaxis] * Tpowers) @ cvx.hstack([1, b]))
rhs = gamma * Tpowers @ cvx.hstack([1, b])
problem = cvx.Problem(cvx.Minimize(0), [lhs <= rhs])

l, u = 0, np.exp(3)
bisection_tol = 1e-3
while u- l >= bisection_tol:
    gamma.value = (l + u) / 2
    problem.solve()
    if problem.status == 'optimal':
        u = gamma.value
        a_opt = a.value
        b_opt = b.value
        objval_opt = gamma.value
    else:
        l = gamma.value

y_fit = (Tpowers @ a_opt / (Tpowers @ np.hstack((1, b_opt))))

plt.figure()
plt.plot(t, y, 'b', t, y_fit, 'r+')
plt.xlabel('t')
plt.ylabel('y')
plt.show()

plt.figure()
plt.plot(t, y_fit- y)
plt.xlabel('t')
plt.ylabel('err')
plt.show()