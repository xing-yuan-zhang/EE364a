import cvxpy as cp
from cvxpy import *
import numpy as np
import itertools

kappa = 0.5
p = np.array([4, 2, 2, 1])
d = np.array([20, 5, 10, 15])
s = np.array([30, 10, 5, 0])
d_tld = np.array([10, 25, 5, 15])
s_tld = np.array([5, 20, 15, 20])

V = [(1,1), (2,1), (2,2), (3,1), (3,3), (4,1), (4,2), (4,3), (4,4)]
S_all = itertools.product((1,2,3,4), repeat = 2)
V_not = list(set(S_all)- set(V))
V = [(i-1,j-1) for i,j in V]
V_not = [(i-1,j-1) for i,j in V_not]

B = Variable((4,4), nonneg = True)
B_tld = Variable((4,4), nonneg = True)
t = Variable(4)
ones = np.ones(4)
obj = p.T@(B.T@ones + B_tld.T@ones) + kappa*norm1(t)
constr = [B@ones == d, B_tld@ones == d_tld,
    B.T@ones <= s- t, B_tld.T@ones <= s_tld + t]
for i,j in V_not:
    constr += [B[i,j] == 0, B_tld[i,j] == 0]
prob = Problem(Minimize(obj), constr)
prob.solve()

print("Optimal cost:", prob.value)
print("Optimal shipment cost:", kappa*np.linalg.norm(t.value, ord=1))
print("Optimal shipment vector:\n", t.value)
print("Optimal policy for bank 1:\n", B.value)
print("Optimal policy for bank 2:\n", B_tld.value)

prob_nos = Problem(Minimize(obj), constr + [t == 0])
prob_nos.solve()

print("Problem status without shipment:", prob_nos.status)