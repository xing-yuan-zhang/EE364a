import cvxpy as cp
from image_colorization_data import *

R = cp.Variable((m, n))
G = cp.Variable((m, n))
B = cp.Variable((m, n))
tv = cp.tv(R, B, G)

objective = cp.Minimize(tv)
constraints = [
    0.299*R + 0.587*G + 0.114*B == M,
    R[known_ind] == R_known,
    G[known_ind] == G_known,
    B[known_ind] == B_known,
    0 <= R, 0 <= G, 0 <= B,
    1 >= R, 1 >= G, 1 >= B,
]

problem = cp.Problem(objective, constraints)
problem.solve()

print("Optimal Total Variation: ", problem.value)
save_img('flower_reconstructed.png', R.value, G.value, B.value)