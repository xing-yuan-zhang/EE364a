import numpy as np
from lp_acent_infeas import lp_acent_infeas

def stdlp_infeas(A,b,c,mu,tol):
    max_iters=35
    _,n = A.shape
    x0 = np.ones(n)
    t = 1.
    gaps = []
    inniters = []
    for k in range(max_iters):
        print("Outer iteration %d" % k)
        x_star, nu_star, inner_iters, _ = lp_acent_infeas(A,b,t*c,x0)
        x0 = x_star
        inniters += [inner_iters]
        gap = n/t
        gaps += [gap]
        if x_star is None:
            print("Error")
            break
        if (gap < tol):
            break
        t = min(t*mu, (n+1.)/tol)

lambda_star =-1.0/(t*x_star)
return x_star, nu_star, lambda_star, np.array([inniters, gaps])