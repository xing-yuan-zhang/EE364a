import cvxpy as cp
import numpy as np

Q =np.array([[1,-0.5], [-0.5,2]])
f =np.array([-1,0])
A =np.array([[1,2], [1,-4],[5, 76]])
b =np.array([-2,-3, 1])
x =cp.Variable(2)

obj =cp.quad_form(x,Q) + f@ x
cons =[A @x <=b]
p_star= cp.Problem(cp.Minimize(obj),cons).solve()
lambdas= cons[0].dual_value

print(p_star)
print(x.value)
print(lambdas)
print(A @ x.value- b)
print(2 * Q @ x.value + f + A.T @ lambdas)

arr_i = np.array([0,-1, 1])
delta = 0.1
pa_table = np.zeros((9, 4))
count = 0
for i in arr_i:
    for j in arr_i:
        p_pred = p_star- (lambdas[0] * i + lambdas[1] * j) * delta
        cons = [A @ x <= b + delta * np.array([i, j, 0])]
        p_exact = cp.Problem(cp.Minimize(obj), cons).solve()
        pa_table[count, :] = np.array([i * delta, j * delta, p_pred, p_exact])
        count += 1

print(pa_table)