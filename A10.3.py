import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from lp_acent_infeas import lp_acent_infeas

def lp_acent_infeas(A,b,c,x_0):
    alpha = 0.1
    beta = 0.5
    epsilon = 1e-5
    max_iters = 50

    m = b.size
    n = x_0.size

    nu = np.zeros(m)
    x = x_0
    residuals = []

    for i in range(max_iters):
        H = np.diag(pow(x,-2))
        r_dual = c- pow(x,-1) + np.dot(A.T, nu)
        r_pri = np.dot(A, x)- b
        r = np.append(r_dual, r_pri)

        r_norm = la.norm(r)
        residuals = residuals + [r_norm];
        print("Iter %i. Residuals %f."%(i, r_norm))
        if r_norm <= epsilon and la.norm(np.dot(A, x)- b) <= epsilon:
            break

    d_nu = la.solve(np.dot(np.dot(A, np.diag(pow(x, 2))), A.T), \
                    r_pri- np.dot(A, np.dot(np.diag(pow(x, 2)), r_dual)))
    d_x = np.dot(-np.diag(pow(x, 2)), (r_dual + np.dot(A.T, d_nu)))

    t = 1
    while np.min(x + t * d_x) <= 0:
        t = beta * t
    r_dual_new = c- np.power(x + t * d_x,-1) + A.T.dot(nu + t * d_nu)
    r_pri_new = np.dot(A, x + t * d_x)- b

    while la.norm(np.append(r_dual_new, r_pri_new)) > (1- alpha * t) * r_norm:
        t = beta * t;
    r_dual_new = c- pow(x + t * d_x,-1) + np.dot(A.T, nu + t * d_nu)
    r_pri_new = np.dot(A, x + t * d_x)- b

    x = x + t * d_x
    nu = nu + t * d_nu
    if i == max_iters- 1:
        print("ERROR: max_iters reached.")
        return (None, None, i, np.array(residuals))
    else:
        return (x, nu, i, np.array(residuals))

m = 100
n = 200

A = np.random.rand(m,n)
p = np.random.rand(n)
b = A.dot(p)
c = np.random.rand(n)
x0 =np.random.rand(n)

x_star,nu_star, it,residuals =lp_acent_infeas(A,b,c,x0)
plt.figure()
plt.semilogy(residuals)
plt.xlim([1,it])
plt.xlabel("Newtonsteps")
plt.ylabel("normof residuals")
plt.show()