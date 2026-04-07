import cvxpy as cvx
import numpy as np

x1 = cvx.Variable()
x2 = cvx.Variable()
constraints = [2*x1 + x2 >= 1, x1 + 3*x2 >= 1, x1 >= 0, x2 >= 0]

objective = cvx.Minimize(x1+x2)
prob = cvx.Problem(objective, constraints)
prob.solve()
print('With obj. x1+x2, p* is %.2f, optimal x1 is %.2f, optimal x2 is %.2f' \
    % (prob.value, x1.value, x2.value))

objective = cvx.Minimize(-x1-x2)
prob = cvx.Problem(objective, constraints)
prob.solve()
print('With obj.-x1-x2, status is ' + prob.status)

objective = cvx.Minimize(x1)
prob = cvx.Problem(objective, constraints)
prob.solve()
print('With obj. x1, p* is %.2f, x1* is %.2f, x2* is %.2f' \
    % (prob.value, x1.value, x2.value))

objective = cvx.Minimize(cvx.maximum(x1, x2))
prob = cvx.Problem(objective, constraints)
prob.solve()
print('With obj. max(x1,x2), p* is %.2f, x1* is %.2f, x2* is %.2f' \
    % (prob.value, x1.value, x2.value))

objective = cvx.Minimize(cvx.square(x1) + 9*cvx.square(x2))
prob = cvx.Problem(objective, constraints)
prob.solve()
print('With obj. x1^2+9x2^2, p* is %.2f, x1* is %.2f, x2* is %.2f' \
    % (prob.value, x1.value, x2.value))