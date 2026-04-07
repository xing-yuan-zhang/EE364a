import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from veh_speed_sched_data import *

t = cvx.Variable(n)
obj = cvx.sum(a*cvx.multiply(cvx.square(d), cvx.inv_pos(t)) + b*d + c*t)
cons = [t <= d/smin, t >= d/smax]
cons += [tau_min[i] <= cvx.sum(t[:i+1]) for i in range(n)]
cons += [tau_max[i] >= cvx.sum(t[:i+1]) for i in range(n)]
cvx.Problem(cvx.Minimize(obj), cons).solve()
s = d / t.value

print("Optimal Fuel Consumption: ", obj.value, " kg")

plt.step(np.arange(n), s, where="post")
plt.xlabel("Waypoints, $i$")
plt.ylabel("Speed, $s_i$")
plt.savefig("veh_speed_sched.eps")
plt.show()