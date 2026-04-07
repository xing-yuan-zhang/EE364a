import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
T = 96
t = np.linspace(1, T, num=T).reshape(T,1)
p = np.exp(-np.cos((t-15)*2*np.pi/T)+0.01*np.random.randn(T,1))
u = 2*np.exp(-0.6*np.cos((t+40)*np.pi/T) \
             - 0.7*np.cos(t*4*np.pi/T)+0.01*np.random.randn(T,1))

plt.figure(1)
plt.plot(t/4, p)
plt.plot(t/4, u, 'r')
plt.legend(['p', 'u'])
plt.show()

q = cp.Variable(shape=(T, 1))
c = cp.Variable(shape=(T, 1))
D = cp.Parameter(nonneg=True)
C = cp.Parameter(nonneg=True)
Q = cp.Parameter(nonneg=True)
obj = p.T @ (u + c)
cons = [c >=-D,
        c <= C,
        q >= 0,
        q <= Q,
        q[1:] == q[:T-1] + c[:T-1],
        q[0] == q[T-1] + c[T-1],
        u + c >= 0]
prob = cp.Problem(cp.Minimize(obj), cons)

Q.value = 35
C.value, D.value = 3, 3
pstar = prob.solve()
ts = np.linspace(1, T, num=T) / 4

plt.figure(2)
plt.subplot(3, 1, 1)
plt.plot(ts, u)
plt.plot(ts, c.value)
plt.legend(['ut', 'ct'])
plt.ylabel('ut, ct')
plt.subplot(3, 1, 2)
plt.plot(ts, p)
plt.ylabel('pt')
plt.subplot(3, 1, 3)
plt.plot(ts, q.value)
plt.xlabel('t')
plt.ylabel('qt')
plt.show()

N = 31
Qs = np.linspace(0, 150, num=N)
Cs = [1, 3]
Ds = [1, 3]
costs = np.zeros((len(Cs), N))
for j, (C.value, D.value) in enumerate(zip(Cs, Ds)):
    for i, Q.value in enumerate(Qs):
        costs[j][i] = prob.solve()

plt.figure(3)
plt.plot(Qs, costs[0], 'g--')
plt.plot(Qs, costs[1], 'b.-')
plt.legend(['cost 1', 'cost 2'])
plt.xlabel('Q')
plt.ylabel('cost')
plt.show()