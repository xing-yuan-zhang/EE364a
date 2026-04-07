import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(1)
n = 20
pbar = np.ones((n, 1)) * 0.03
pbar += np.r_[np.random.rand(n- 1, 1), np.zeros((1, 1))] * 0.12
S = np.random.randn(n, n)
S = S.T @ S
S = S / max(np.abs(np.diag(S))) * 0.2
S[:,-1] = np.zeros(n)
S[-1, :] = np.zeros(n)

x = cp.Variable(shape=(n, 1))
risk = cp.quad_form(x, S)
constraints = [cp.sum(x) == 1, pbar.T @ x == sum(pbar) / n]

x_unif = np.ones((n, 1)) / n
print("Risk for uniform: %.2f" % (np.sqrt(x_unif.T @ S @ x_unif) * 100))

cp.Problem(cp.Minimize(risk), constraints).solve()
print("Risk for unconstrained: %.2f" % (np.sqrt(risk.value.item()) * 100))

cp.Problem(cp.Minimize(risk), constraints + [x >= 0]).solve()
print("Risk for long only: %.2f" % (np.sqrt(risk.value.item()) * 100))

cp.Problem(
    cp.Minimize(risk),
    constraints + [cp.sum(cp.neg(x)) <= 0.5]
).solve()
print("Risk for limit on short: %.2f" % (np.sqrt(risk.value.item()) * 100))

gamma = cp.Parameter(nonneg=True)
expec_return = pbar.T @ x
prob = cp.Problem(cp.Maximize(expec_return- gamma * risk), [])
N = 128

gamma_vals = np.logspace(-1, 5, num=N)
return_vec1 = np.zeros((N, 1))
risk_vec1 = np.zeros((N, 1))
constraints = [cp.sum(x) == 1, x >= 0]
prob = cp.Problem(prob.objective, constraints)
for i in range(N):
    gamma.value = gamma_vals[i]
    prob.solve()
    return_vec1[i] = expec_return.value
    risk_vec1[i] = risk.value

plt.figure()
plt.plot(np.sqrt(risk_vec1) * 100, return_vec1 * 100, label="Long only")

return_vec2 = np.zeros((N, 1))
risk_vec2 = np.zeros((N, 1))
constraints = [cp.sum(x) == 1, cp.sum(cp.neg(x)) <= 0.5]
prob = cp.Problem(prob.objective, constraints)
for i in range(N):
    gamma.value = gamma_vals[i]
    prob.solve()
    return_vec2[i] = expec_return.value
    risk_vec2[i] = risk.value

plt.plot(np.sqrt(risk_vec2) * 100, return_vec2 * 100, label="Limit on short")
plt.legend()
plt.xlabel("Risk in %")
plt.ylabel("Return in %")
plt.savefig("simple_portfolio.eps")
plt.show()