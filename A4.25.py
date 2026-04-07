import cvxpy as cp

p = cp.Variable((2, 2, 2, 2), nonneg=True)
constrs = [cp.sum(p) == 1,
cp.sum(p[1, :, :, :]) == 0.9,
cp.sum(p[:, 1, :, :]) == 0.9,
cp.sum(p[:, :, 1, :]) == 0.1]
constrs += [p[1, 0, 1, 0] + p[1, 1, 1, 0] == 0.7 * cp.sum(p[:, :, 1, :])]
constrs += [p[0, 1, 0, 1] + p[1, 1, 0, 1] == 0.6 * cp.sum(p[:, 1, 0, :])]
P = cp.sum(p[:, :, :, 1])

for direction in [-1, 1]:
    prob = cp.Problem(cp.Maximize(direction*P), constrs)
    prob.solve()
    
print(P.value)