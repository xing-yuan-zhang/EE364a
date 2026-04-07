import cvxpy as cp
import numpy as np
from opt_schedule_data import n, alpha, m, P

s, f = cp.Variable(n), cp.Variable(n)
T = cp.Parameter()
obj = cp.Minimize(cp.sum(cp.multiply(alpha, cp.inv_pos(f- s- m))))
constr = [0 <= s, s <= f, f <= T] + [s[j] >= f[i] for i, j in P]
prob = cp.Problem(obj, constr)

T_ = np.linspace(10, 30)
C_ = [prob.solve() for T.value in T_]