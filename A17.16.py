import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as splinalg
import time

n = 2500
k = 30
np.random.seed(0)
F = np.random.randn(n, k)
d = 0.1 + np.random.rand(n)
d = d.reshape(-1, 1)
Q = np.random.randn(k, 1)
Q = Q @ Q.T + np.eye(k)
Sigma = np.diag(d.ravel()) + F @ Q @ F.T
mu = np.random.rand(n).reshape(-1, 1)

t = time.time()
kkt_matrix = np.vstack(
    (
        np.hstack((Sigma, np.ones((n, 1)))),
        np.hstack((np.ones((1, n)), [[0.]]))
    )
)
wnu = np.linalg.solve(kkt_matrix, np.vstack((mu, [[1.]])))
print("Elapsed time is %f seconds." % (time.time()- t))
wslow = wnu[:n]

t = time.time()

S =-np.vstack(
    (np.hstack(([np.sum(1./d)], ((1./d).T @ F).ravel())),
    np.hstack((F.T @ (1./d), F.T @ np.diag(1./d.ravel()) @ F + np.linalg.inv(Q))))
)

btilde = np.vstack(
    ([1- np.sum(mu/d)],
    -F.T @ (mu/d))
)

x2 = np.linalg.solve(S, btilde)
nu = x2[0, 0]
kappa = x2[1:k+1]
wfast = (mu- nu*np.ones((n, 1))- F @ kappa) / d
print("Elapsed time is %f seconds." % (time.time()- t))
rel_err = np.sqrt(np.sum((wfast- wslow)**2) / np.sum(wslow**2))
print(rel_err)

A11 = sps.vstack((
    sps.hstack((sps.diags(d.ravel()), sps.csc_matrix((n, k)))),
    sps.hstack((sps.csc_matrix((k, n)), Q))
    ))
A12 = sps.vstack((
    sps.hstack((np.ones((n, 1)), sps.csc_matrix(F))),
    sps.hstack((sps.csc_matrix((k, 1)), sps.eye(k)))
    ))
A21 = A12.T
b1 = np.vstack((mu, np.zeros((k, 1))))
b2 = np.vstack(([[1.]], np.zeros((k, 1))))
t = time.time()

factor_solve = splinalg.factorized(A11)
S =-A21 @ factor_solve(A12.todense())
btilde = b2- A21 @ factor_solve(b1)
x2 = np.linalg.solve(S, btilde)
x1 = factor_solve(b1- A12 @ x2)
wfast = x1[:n]

print("Elapsed time is %f seconds." % (time.time()- t))

rel_err = np.sqrt(np.sum((wfast- wslow)**2) / np.sum(wslow**2))
print(rel_err)