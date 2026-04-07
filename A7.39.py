import cvxpy as cp
import numpy as np
from poisson_ar_data import x

u = cp.Variable()
v = cp.Variable()
log_likelihood = 0
for t in range(1,len(x)):
    log_likelihood += -cp.exp(u + v*x[t-1]) + x[t]*(u + v*x[t-1])
prob = cp.Problem(cp.Maximize(log_likelihood))
prob.solve()

print("nu:", np.exp(u.value))
print("omega:", np.exp(v.value))