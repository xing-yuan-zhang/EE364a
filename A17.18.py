import cvxpy as cp
import numpy as np

m = 200
r = 1.05
p_given = np.array([1., 1., 0.06, 0.03, 0.02, 0.01])
F = 0.9
C = 1.15

S = np.linspace(0.5, 2, m)
Vt = np.vstack((
    r * np.ones(m),
    S,
    np.maximum(0, S- 1.1),
    np.maximum(0, S- 1.2),
    np.maximum(0, 0.8- S),
    np.maximum(0, 0.7- S),
    np.clip(S, a_min=F, a_max=C)
))

p = cp.Variable(7)
y = cp.Variable(m)

constraints = [
    p[:6] == p_given,
    Vt @ y == p,
    y >= 0
]

prob_min = cp.Problem(cp.Minimize(p[-1]), constraints)
prob_max = cp.Problem(cp.Maximize(p[-1]), constraints)
p_min = prob_min.solve()
p_max = prob_max.solve()
print(f"Lower Bound: {p_min}")
print(f"Upper Bound: {p_max}")