import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from microgrid_data import *

p_batt = cvx.Variable(N)
p_buy = cvx.Variable(N)
p_sell = cvx.Variable(N)
p_grid = p_buy- p_sell
q = cvx.Variable(N)

payments = (0.25) * p_buy.T * R_buy
income = (0.25) * p_sell.T * R_sell
obj = cvx.Minimize(payments- income)

constraints = []

constraints += [p_ld == p_grid + p_batt + p_pv]

constraints += [p_buy >= 0, p_sell >= 0]

constraints += [q <= Q, q >= 0, p_batt <= D, p_batt >=-C]

constraints += [q[0] == q[N- 1]- (0.25) * p_batt[N- 1]]
for i in range(1, N):
    constraints += [q[i] == q[i- 1]- (0.25) * p_batt[i- 1]]

prob = cvx.Problem(obj, constraints)
prob.solve()

print("Minimum Cost: ", obj.value)

q = q.value
p_grid = p_grid.value
p_batt = p_batt.value

plt.figure(figsize=fig_size)
plt.plot(p_grid)
plt.ylabel("Power (kW)")
plt.title("p_grid (kW)", fontsize=19)
plt.xticks(xtick_vals, xtick_labels)
plt.axvline(partial_peak_start, linestyle="--", color="black")
plt.axvline(peak_start, linestyle="--", color="black")
plt.axvline(peak_end, linestyle="--", color="black")
plt.axvline(partial_peak_end, linestyle="--", color="black")
plt.axhline(0, color="black")
plt.savefig("microgrid_p_grid_plot.eps")
plt.show()

plt.figure(figsize=fig_size)
plt.plot(p_batt)
plt.ylabel("Power (kW)")
plt.title("p_batt (kW)", fontsize=19)
plt.xticks(xtick_vals, xtick_labels)
plt.axvline(partial_peak_start, linestyle="--", color="black")
plt.axvline(peak_start, linestyle="--", color="black")
plt.axvline(peak_end, linestyle="--", color="black")
plt.axvline(partial_peak_end, linestyle="--", color="black")
plt.axhline(D, linestyle="--", color="black")
plt.axhline(-C, linestyle="--", color="black")
plt.axhline(0, color="black")
plt.savefig("microgrid_p_batt_plot.eps")
plt.show()

plt.figure(figsize=fig_size)
plt.plot(q)
plt.ylabel("Energy (kWh)")
plt.title("Battery Charge (kWh)", fontsize=19)
plt.xticks(xtick_vals, xtick_labels)
plt.axvline(partial_peak_start, linestyle="--", color="black")
plt.axvline(peak_start, linestyle="--", color="black")
plt.axvline(peak_end, linestyle="--", color="black")
plt.axvline(partial_peak_end, linestyle="--", color="black")
plt.axhline(Q, linestyle="--", color="black")
plt.ylim(bottom=0)
plt.savefig("microgrid_batt_energy_plot.eps")
plt.show()

dual_vals = -constraints[0].dual_value
LMP = 4 * dual_vals

plt.figure(figsize=(19, 5))
plt.plot(R_buy, "--", label="Buy Price", linewidth=2)
plt.plot(R_sell, "--", label="Sell Price", linewidth=2)
plt.plot(LMP, linewidth=2, label="LMP")
plt.xlabel("Time")
plt.ylabel("Price ($/kWh)")
plt.title("Locational Marginal Price", fontsize=18)
plt.legend()
plt.xticks(xtick_vals, xtick_labels)
plt.savefig("microgrid_LMP_plot.eps")
plt.show()

load_cost = p_ld @ dual_vals
batt_cost = p_batt @ dual_vals
PV_cost = p_pv @ dual_vals
grid_costs = p_grid @ dual_vals
print("Battery cost: %.2f" % (batt_cost))
print("PV cost: %.2f" % (PV_cost))
print("Effective grid cost: %.2f" % (grid_costs))
print("Load cost: %.2f" % (load_cost))
print("Battery + PV + grid cost: %.2f" % (grid_costs + batt_cost + PV_cost))