import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from various_obj_regulator_data import *

u_ = cp.Variable((m,T))
x_ = cp.Variable((n,T+1))

objs = [
(cp.Minimize(cp.sum_squares(u_)), r"(a) $\sum\|u_t\|_2^2$"),
(cp.Minimize(cp.sum(cp.norm(u_,2,axis=0))), r"(b) $\sum\|u_t\|_2$"),
(cp.Minimize(cp.max(cp.norm(u_,axis=0))), r"(c) $\max\|u_t\|_2$"),
(cp.Minimize(cp.sum(cp.norm(u_,1,axis=0))), r"(d) $\sum\|u_t\|_1$")
]

plt.rcParams[’text.usetex’] = True
plt.figure(figsize=(15,5))

for i,obj in enumerate(objs):
    const = [x_[:,-1] == np.zeros(n)]
    const.append(x_[:,0] == x_init)
    for t in range(1,T+1):
        const.append(x_[:,t] == A@x_[:,t-1] + B@u_[:,t-1])
    prob = cp.Problem(obj[0],const)
    prob.solve()
    plt.subplot(2,4,i+1)
    plt.plot(u_.value.T)
    if i == 0:
        plt.ylabel("$u_t$")
    plt.title(obj[1])
    plt.grid()
    plt.xlabel("t")
    plt.subplot(2,4,i+5)
    plt.xlabel("t")
    plt.plot(np.linalg.norm(u_.value,axis=0),c="black",label=r"$\|u\|_2$")
    if i == 2:
        plt.ylim(ymax = .12,ymin=0)
    if i == 0:
        plt.ylabel(r"$\|u_t\|_2$")
    plt.grid()
plt.tight_layout()
plt.show()