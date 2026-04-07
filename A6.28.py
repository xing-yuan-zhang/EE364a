import cvxpy as cvx
import numpy as np
from ranking_est_data import *

def regression_loss_cvx(X, pi, theta):
    pi = np.array(pi)
    X = np.array(X)
    N = X.shape[0]
    Y = X @ theta.T
    return cvx.norm(cvx.vec(Y- pi), 1) / (2 * N)

def avg_ranking_distance(pi_1, pi_2):
    return np.sum(np.abs(pi_1- pi_2))/ (2 * pi_1.shape[0])

theta = cvx.Variable((K, d))
obj = regression_loss_cvx(X_train, pi_train, theta)
problem = cvx.Problem(cvx.Minimize(obj), [])
problem.solve()

train_loss = obj.value
test_loss = regression_loss_cvx(X_test, pi_test, theta).value
predicted_ranking_train = Pi(X_train.dot(theta.value.T))
predicted_ranking_test = Pi(X_test.dot(theta.value.T))
train_error = avg_ranking_distance(pi_train, predicted_ranking_train)
test_error = avg_ranking_distance(pi_test, predicted_ranking_test)
print("train loss: {:.3f}\ttrain error: {:.3f}".format(train_loss, train_error))
print("test loss: {:.3f}\ttest error: {:.3f}".format(test_loss, test_error))

n_print = 10

print("True and Predicted Test Rankings and their distance:")
for pi_pred, pi_true in zip(predicted_ranking_test[:n_print], pi_test[:n_print]):
    print("[",("{:2d}, " * len(pi_pred))[:-2].format(*pi_pred) + "], [" +
        ("{:2d}, " * len(pi_true))[:-2].format(*pi_true) +
         "], {}".format(avg_ranking_distance(pi_pred.reshape(1,-1),
                                             pi_true.reshape(1,-1)))
        )

print("True and Predicted Train Rankings and their distance:")
for pi_pred, pi_true in zip(predicted_ranking_train[:n_print], pi_train[:n_print]):
    print("[",("{:2d}, " * len(pi_pred))[:-2].format(*pi_pred) + "], [" +
        ("{:2d}, " * len(pi_true))[:-2].format(*pi_true) +
         "], {}".format(avg_ranking_distance(pi_pred.reshape(1,-1),
                                             pi_true.reshape(1,-1)))
        )