import numpy as np
import cvxpy as cvx

np.random.seed(15)

n = 20
M = 25
K = 100
c_true = np.random.randn(n)
X = np.random.randn(n,K)
y = np.dot(np.transpose(X), c_true) + 0.1*(np.sqrt(n))*np.random.randn(K)

sort_ind = np.argsort(y.T)
y = np.sort(y.T)
y = y.T
X = X[:, sort_ind.T]
D = (y[M-1]+y[M])/2.0
y = y[list(range(M))]

z = cvx.Variable(K-M)
c = cvx.Variable(n)
objective = cvx.Minimize(
    cvx.sum_squares(y-X[:, :M].T @ c)
    + cvx.sum_squares(z-X[:,M:].T @ c)
)
constraints = [D <= z]
prob = cvx.Problem(objective, constraints)
prob.solve()
c_cens = c.value

objective = cvx.Minimize(cvx.sum_squares(y-X[:, :M].T @ c))
prob = cvx.Problem(objective, [])
prob.solve()
c_ls = c.value
cens_relerr = np.linalg.norm(c_cens-c_true)/np.linalg.norm(c_true)
ls_relerr = np.linalg.norm(c_ls-c_true)/np.linalg.norm(c_true)

print("c_true is:")
print(c_true.T)
print("c_cens is:")
print(np.squeeze(np.asarray(c_cens.T)))
print("c_ls is:")
print(np.squeeze(np.asarray(c_ls.T)))
print("The relative error when we use the censored data is:")
print(cens_relerr)
print("The relative error when we do not use the censored data is:")
print(ls_relerr)