"""
Trying to see some results now
"""
import numpy as np

from minimisers import *

parameters = [200, data, pred]

print("\n1D Minimisation")
minim1, f_min1, iters = parabolic_method(nll_1d, (0.7), (np.pi/4 - 0.05), [np.pi/4, 2.4], 0, parameters, 1e-6)
minim2, f_min2, iters = parabolic_method(nll_1d, (np.pi/4 + 0.05), (0.9), [np.pi/4, 2.4], 0, parameters, 1e-6)
err_plus, err_minus = nll_plusmin_error(minim1, parameters)
error1 = error_curv(nll_1d, [minim1], parameters)
error2 = error_curv(nll_1d, [minim2], parameters)
print("\nParabolic method:")
print(f"{round(minim1, 3)} +/- {round(error1[0], 3)}")
print(f"{round(minim2, 3)} +/- {round(error2[0], 3)}")
print(f"Error using NLL +/ 0.5: {err_plus}, {err_minus}")
print(iters)


print("\n2D Minimisation:")
theta_u, mdiff_u, count_univ = univariate_method(nll_2d, [[0.7, np.pi/4 - 0.1], [2.2, 2.45]], 1e-6)
min_univ = [theta_u[-1], mdiff_u[-1]]
err_univ = error_curv(nll_2d, min_univ, parameters)
print("\nUnivariate method:")
print(f"{round(min_univ[0], 3)} +/- {round(err_univ[0], 3)}")
print(f"{round(min_univ[1], 3)} +/- {round(err_univ[1], 3)}")
print(count_univ)

min_grad, iter_grad = gradient_method(nll_2d, [0.7, 2.3], 1e-6)
err_grad = error_curv(nll_2d, min_grad, parameters)
print("\nGradient method:")
print(f"{round(min_grad[0], 3)} +/- {round(err_grad[0], 3)}")
print(f"{round(min_grad[1], 3)} +/- {round(err_grad[1], 3)}")
print(iter_grad)

min_newt, iter_newt = newtons_method(nll_2d, [0.7, 2.3], 1e-6)
err_newt = error_curv(nll_2d, min_newt, parameters)
print("\nNewton method:")
print(f"{round(min_newt[0], 3)} +/- {round(err_newt[0], 3)}")
print(f"{round(min_newt[1], 3)} +/- {round(err_newt[1], 3)}")
print(iter_newt)

min_quas, iter_quas = quasi_newton_method(nll_2d, [0.7, 2.3], 1e-6, update_method = "broyden")
err_quas = error_curv(nll_2d, min_quas, parameters)
print("\nQuasi-Newton method:")
print(f"{round(min_quas[0], 3)} +/- {round(err_quas[0], 3)}")
print(f"{round(min_quas[1], 3)} +/- {round(err_quas[1], 3)}")
print(iter_quas)


print("\n\n3D Minimisation:")
print("\nGradient:")
min_3, it_3 = gradient_method(nll_3d_alpha, [0.777, 2.338, 0.5], 1e-6)
err_grad = error_curv(nll_3d_alpha, min_3, parameters)
print("Minimum at:")
print(min_3)
print("Error:")
print(err_grad)

print("\nQuasi-Newton method:")
min_3, it_3 = quasi_newton_method(nll_3d_alpha, [0.777, 2.338, 0.5], 1e-6)
err_grad = error_curv(nll_3d_alpha, min_3, parameters)
print("Minimum at:")
print(min_3)
print("Error:")
print(err_grad)

print("\nNewton method in 3D had issues with evaluating the function troughout its path")


print("\n\n4D Minimisation:")
print("\nGradient:")
min_4, it_4 = gradient_method(nll_4d, [0.701, 2.398, 0.683, 0.015], 1e-5)
err_grad = error_curv(nll_4d, min_4, parameters)
print("Minimum at:")
print(min_4)
print("Error:")
print(err_grad)

print("\nNewton method")
min_4, it_4 = newtons_method(nll_4d, [0.701, 2.398, 0.683, 0.015], 1e-5)
err_newt = error_curv(nll_4d, min_4, parameters)
print("Minimum at:")
print(min_4)
print("Error:")
print(err_newt)

print("\nQuasi-Newton method:")
min_4, it_4 = quasi_newton_method(nll_4d, [0.701, 2.398, 0.683, 0.015], 1e-5)
err_quas = error_curv(nll_4d, min_4, parameters)
print("Minimum at:")
print(min_4)
print("Error:")
print(err_quas)