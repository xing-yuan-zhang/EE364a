import numpy as np
import cvxpy as cp
import dccp
from moving_obstacle_data import T, h, m, S, dmin, obstacles

p = cp.Variable((T, 2))
v = cp.Variable((T, 2))
f = cp.Variable((T, 2))

obj = cp.Minimize(cp.sum(cp.norm(f, axis=1)))
constr = [
    cp.norm(f, axis=1) <= 1,
    p[1:] == p[:-1] + h * v[:-1],
    v[1:] == v[:-1] + (h / m) * f[:-1],
    p[0] == 0, p[-1] == [1, 0],
    v[0] == [S, 0], v[-1] == [S, 0]
] + [
    cp.norm(p- ob, axis=1) >= dmin
    for ob in obstacles
]

prob = cp.Problem(obj, constr)

f.value = np.zeros(f.shape)
v.value = np.zeros(v.shape)
v.value[:, 0] = S
p.value = np.zeros(p.shape)
p.value[:, 0] = np.linspace(0, 1, T)
prob.solve(method='dccp')
print('Value:', obj.value)